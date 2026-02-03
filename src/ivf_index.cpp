/// @file ivf_index.cpp
/// @brief Implementation of IVF Index for scalable SAQ search.

#include "index/ivf_index.h"
#include "index/fast_scan/fast_scan.h"
#include "saq/simd_kernels.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>

namespace saq {

// ============================================================================
// FlatInitializer Implementation
// ============================================================================

bool FlatInitializer::AddCentroids(const float* centroids, uint32_t k,
                                    uint32_t dim) {
  if (centroids == nullptr || k == 0 || dim == 0) {
    return false;
  }

  num_clusters_ = k;
  dim_ = dim;
  centroids_.assign(centroids, centroids + static_cast<size_t>(k) * dim);
  return true;
}

const float* FlatInitializer::GetCentroid(uint32_t id) const {
  if (id >= num_clusters_) {
    return nullptr;
  }
  return centroids_.data() + static_cast<size_t>(id) * dim_;
}

void FlatInitializer::FindNearestClusters(
    const float* query, uint32_t dim, uint32_t nprobe,
    std::vector<ClusterCandidate>& candidates) const {
  if (query == nullptr || dim != dim_) {
    candidates.clear();
    return;
  }

  nprobe = std::min(nprobe, num_clusters_);
  std::vector<ClusterCandidate> all_candidates(num_clusters_);

  // Compute all distances using SIMD batch operation
  std::vector<float> distances(num_clusters_);
  simd::L2DistancesBatch(query, centroids_.data(), num_clusters_, dim_, 
                          distances.data());

  for (uint32_t i = 0; i < num_clusters_; ++i) {
    all_candidates[i].id = i;
    all_candidates[i].distance = distances[i];
  }

  std::partial_sort(all_candidates.begin(),
                    all_candidates.begin() + nprobe,
                    all_candidates.end());

  candidates.assign(all_candidates.begin(),
                    all_candidates.begin() + nprobe);
}


std::vector<uint8_t> FlatInitializer::Serialize() const {
  // Format: [num_clusters][dim][centroids_data]
  size_t data_size = 2 * sizeof(uint32_t) +
                     centroids_.size() * sizeof(float);
  std::vector<uint8_t> buffer(data_size);
  uint8_t* ptr = buffer.data();

  std::memcpy(ptr, &num_clusters_, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(ptr, &dim_, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(ptr, centroids_.data(), centroids_.size() * sizeof(float));

  return buffer;
}

bool FlatInitializer::Deserialize(const std::vector<uint8_t>& data) {
  if (data.size() < 2 * sizeof(uint32_t)) {
    return false;
  }

  const uint8_t* ptr = data.data();
  std::memcpy(&num_clusters_, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(&dim_, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);

  size_t expected_size = 2 * sizeof(uint32_t) +
                         static_cast<size_t>(num_clusters_) * dim_ * sizeof(float);
  if (data.size() != expected_size) {
    return false;
  }

  centroids_.resize(static_cast<size_t>(num_clusters_) * dim_);
  std::memcpy(centroids_.data(), ptr, centroids_.size() * sizeof(float));

  return true;
}

// ============================================================================
// HNSWInitializer Implementation
// ============================================================================

/// @brief Simple HNSW graph structure for centroid search.
///
/// This is a simplified HNSW implementation focused on small graphs
/// (centroid search typically involves 1K-20K nodes).
struct HNSWInitializer::HNSWGraph {
  struct Node {
    std::vector<std::vector<uint32_t>> neighbors;  // neighbors per level
    uint32_t max_level = 0;
  };

  std::vector<Node> nodes;
  uint32_t entry_point = 0;
  uint32_t max_level = 0;
  uint32_t m = 16;
  uint32_t m_max = 16;
  uint32_t m_max_0 = 32;
  uint32_t ef_construction = 200;
  float level_mult = 0.0f;
  std::mt19937 rng;

  const float* data = nullptr;
  uint32_t dim = 0;

  HNSWGraph(uint32_t m_param, uint32_t ef_con, uint32_t seed)
      : m(m_param), m_max(m_param), m_max_0(2 * m_param),
        ef_construction(ef_con), rng(seed) {
    level_mult = 1.0f / std::log(static_cast<float>(m));
  }

  uint32_t RandomLevel() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    return static_cast<uint32_t>(-std::log(r) * level_mult);
  }

  float Distance(uint32_t id, const float* query) const {
    return simd::L2DistanceSquared(data + static_cast<size_t>(id) * dim, query, dim);
  }

  void Insert(uint32_t id) {
    uint32_t new_level = RandomLevel();

    if (nodes.empty()) {
      Node n;
      n.max_level = new_level;
      n.neighbors.resize(new_level + 1);
      nodes.push_back(std::move(n));
      entry_point = 0;
      max_level = new_level;
      return;
    }

    // Extend node structure
    while (nodes.size() <= id) {
      nodes.emplace_back();
    }
    nodes[id].max_level = new_level;
    nodes[id].neighbors.resize(new_level + 1);

    const float* query = data + static_cast<size_t>(id) * dim;
    uint32_t ep = entry_point;

    // Navigate from top to insertion level
    for (int level = static_cast<int>(max_level);
         level > static_cast<int>(new_level); --level) {
      ep = GreedySearch(query, ep, level);
    }

    // Insert at each level
    for (int level = std::min(new_level, max_level);
         level >= 0; --level) {
      auto neighbors = SearchLevel(query, ep, ef_construction, level);

      // Select M neighbors
      uint32_t max_m = (level == 0) ? m_max_0 : m_max;
      if (neighbors.size() > max_m) {
        neighbors.resize(max_m);
      }

      for (auto& [dist, nid] : neighbors) {
        if (nid != id) {
          nodes[id].neighbors[level].push_back(nid);
          nodes[nid].neighbors[level].push_back(id);

          // Prune if necessary
          if (nodes[nid].neighbors[level].size() > max_m) {
            PruneNeighbors(nid, level, max_m);
          }
        }
      }

      if (!neighbors.empty()) {
        ep = neighbors[0].second;
      }
    }

    if (new_level > max_level) {
      entry_point = id;
      max_level = new_level;
    }
  }

  uint32_t GreedySearch(const float* query, uint32_t ep, uint32_t level) const {
    float cur_dist = Distance(ep, query);

    while (true) {
      uint32_t best = ep;
      float best_dist = cur_dist;

      for (uint32_t neighbor : nodes[ep].neighbors[level]) {
        float d = Distance(neighbor, query);
        if (d < best_dist) {
          best_dist = d;
          best = neighbor;
        }
      }

      if (best == ep) {
        return ep;
      }
      ep = best;
      cur_dist = best_dist;
    }
  }

  std::vector<std::pair<float, uint32_t>> SearchLevel(
      const float* query, uint32_t ep, uint32_t ef, uint32_t level) const {
    // Min-heap for candidates (closest first)
    std::priority_queue<std::pair<float, uint32_t>,
                        std::vector<std::pair<float, uint32_t>>,
                        std::greater<>> candidates;
    // Max-heap for results (furthest first)
    std::priority_queue<std::pair<float, uint32_t>> results;

    std::vector<bool> visited(nodes.size(), false);

    float ep_dist = Distance(ep, query);
    candidates.emplace(ep_dist, ep);
    results.emplace(ep_dist, ep);
    visited[ep] = true;

    while (!candidates.empty()) {
      auto [dist, cur] = candidates.top();
      candidates.pop();

      if (dist > results.top().first) {
        break;
      }

      if (level < nodes[cur].neighbors.size()) {
        for (uint32_t neighbor : nodes[cur].neighbors[level]) {
          if (!visited[neighbor]) {
            visited[neighbor] = true;
            float d = Distance(neighbor, query);

            if (results.size() < ef || d < results.top().first) {
              candidates.emplace(d, neighbor);
              results.emplace(d, neighbor);

              if (results.size() > ef) {
                results.pop();
              }
            }
          }
        }
      }
    }

    std::vector<std::pair<float, uint32_t>> result_vec;
    result_vec.reserve(results.size());
    while (!results.empty()) {
      result_vec.push_back(results.top());
      results.pop();
    }
    std::reverse(result_vec.begin(), result_vec.end());
    return result_vec;
  }

  void PruneNeighbors(uint32_t id, uint32_t level, uint32_t max_m) {
    auto& neighbors = nodes[id].neighbors[level];
    if (neighbors.size() <= max_m) {
      return;
    }

    const float* node_data = data + static_cast<size_t>(id) * dim;
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(neighbors.size());

    for (uint32_t n : neighbors) {
      float d = simd::L2DistanceSquared(node_data,
                                         data + static_cast<size_t>(n) * dim, dim);
      scored.emplace_back(d, n);
    }

    std::partial_sort(scored.begin(),
                      scored.begin() + max_m,
                      scored.end());


    neighbors.clear();
    neighbors.reserve(max_m);
    for (uint32_t i = 0; i < max_m; ++i) {
      neighbors.push_back(scored[i].second);
    }
  }

  std::vector<std::pair<float, uint32_t>> Search(
      const float* query, uint32_t k, uint32_t ef) const {
    if (nodes.empty()) {
      return {};
    }

    uint32_t ep = entry_point;

    // Navigate from top level
    for (int level = static_cast<int>(max_level); level > 0; --level) {
      ep = GreedySearch(query, ep, level);
    }

    // Search at level 0
    auto results = SearchLevel(query, ep, std::max(ef, k), 0);

    if (results.size() > k) {
      results.resize(k);
    }

    return results;
  }
};

HNSWInitializer::HNSWInitializer(uint32_t m, uint32_t ef_construction)
    : m_(m), ef_construction_(ef_construction), ef_(std::max(100u, 2 * m)) {}

HNSWInitializer::~HNSWInitializer() = default;

bool HNSWInitializer::AddCentroids(const float* centroids, uint32_t k,
                                    uint32_t dim) {
  if (centroids == nullptr || k == 0 || dim == 0) {
    return false;
  }

  num_clusters_ = k;
  dim_ = dim;
  centroids_.assign(centroids, centroids + static_cast<size_t>(k) * dim);

  graph_ = std::make_unique<HNSWGraph>(m_, ef_construction_, 42);
  graph_->data = centroids_.data();
  graph_->dim = dim;

  for (uint32_t i = 0; i < k; ++i) {
    graph_->Insert(i);
  }

  return true;
}

const float* HNSWInitializer::GetCentroid(uint32_t id) const {
  if (id >= num_clusters_) {
    return nullptr;
  }
  return centroids_.data() + static_cast<size_t>(id) * dim_;
}

void HNSWInitializer::FindNearestClusters(
    const float* query, uint32_t dim, uint32_t nprobe,
    std::vector<ClusterCandidate>& candidates) const {
  if (query == nullptr || dim != dim_ || !graph_) {
    candidates.clear();
    return;
  }

  nprobe = std::min(nprobe, num_clusters_);
  uint32_t ef = std::max(ef_, 2 * nprobe);

  auto results = graph_->Search(query, nprobe, ef);

  candidates.clear();
  candidates.reserve(results.size());
  for (const auto& [dist, id] : results) {
    candidates.push_back({id, dist});
  }
}

std::vector<uint8_t> HNSWInitializer::Serialize() const {
  // For simplicity, serialize same as Flat and rebuild on load
  // (HNSW graphs are small for centroid search)
  size_t data_size = 4 * sizeof(uint32_t) +
                     centroids_.size() * sizeof(float);
  std::vector<uint8_t> buffer(data_size);
  uint8_t* ptr = buffer.data();

  std::memcpy(ptr, &num_clusters_, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(ptr, &dim_, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(ptr, &m_, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(ptr, &ef_construction_, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(ptr, centroids_.data(), centroids_.size() * sizeof(float));

  return buffer;
}

bool HNSWInitializer::Deserialize(const std::vector<uint8_t>& data) {
  if (data.size() < 4 * sizeof(uint32_t)) {
    return false;
  }

  const uint8_t* ptr = data.data();
  std::memcpy(&num_clusters_, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(&dim_, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(&m_, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  std::memcpy(&ef_construction_, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);

  size_t expected_size = 4 * sizeof(uint32_t) +
                         static_cast<size_t>(num_clusters_) * dim_ * sizeof(float);
  if (data.size() != expected_size) {
    return false;
  }

  centroids_.resize(static_cast<size_t>(num_clusters_) * dim_);
  std::memcpy(centroids_.data(), ptr, centroids_.size() * sizeof(float));

  // Rebuild HNSW graph
  return AddCentroids(centroids_.data(), num_clusters_, dim_);
}

// ============================================================================
// IVFIndex Implementation
// ============================================================================

std::string IVFIndex::Build(const float* data, uint32_t n_vectors,
                             uint32_t dim, const float* centroids,
                             const uint32_t* cluster_ids,
                             const IVFTrainConfig& config) {
  if (data == nullptr || n_vectors == 0 || dim == 0) {
    return "Invalid input data";
  }
  if (centroids == nullptr || cluster_ids == nullptr) {
    return "Centroids and cluster_ids are required";
  }

  built_ = false;
  num_vectors_ = n_vectors;
  dim_ = dim;
  default_nprobe_ = config.ivf.nprobe;

  const uint32_t num_clusters = config.ivf.num_clusters;

  // Step 1: Create centroid initializer
  bool use_hnsw = config.ivf.use_hnsw_initializer ||
                  (num_clusters >= 20000);

  if (use_hnsw) {
    auto hnsw = std::make_unique<HNSWInitializer>(
        config.ivf.hnsw_m, config.ivf.hnsw_ef_construction);
    if (!hnsw->AddCentroids(centroids, num_clusters, dim)) {
      return "Failed to add centroids to HNSW initializer";
    }
    initializer_ = std::move(hnsw);
  } else {
    auto flat = std::make_unique<FlatInitializer>();
    if (!flat->AddCentroids(centroids, num_clusters, dim)) {
      return "Failed to add centroids to Flat initializer";
    }
    initializer_ = std::move(flat);
  }

  // Step 2: Organize vectors by cluster
  std::vector<std::vector<uint32_t>> cluster_vectors(num_clusters);
  for (uint32_t i = 0; i < n_vectors; ++i) {
    uint32_t cid = cluster_ids[i];
    if (cid >= num_clusters) {
      return "Invalid cluster_id: " + std::to_string(cid);
    }
    cluster_vectors[cid].push_back(i);
  }

  // Step 3: Train SAQ on a sample of vectors (or all if small enough)
  std::vector<float> training_data;
  std::vector<uint32_t> sample_indices;
  const uint32_t max_training = 100000;

  if (n_vectors <= max_training) {
    // Use all vectors
    training_data.assign(data, data + static_cast<size_t>(n_vectors) * dim);
  } else {
    // Subsample
    sample_indices.reserve(max_training);
    std::mt19937 rng(config.seed);

    // Sample proportionally from each cluster
    for (uint32_t c = 0; c < num_clusters; ++c) {
      const auto& vecs = cluster_vectors[c];
      uint32_t n_sample = std::max(1u, static_cast<uint32_t>(
          vecs.size() * max_training / n_vectors));
      n_sample = std::min(n_sample, static_cast<uint32_t>(vecs.size()));

      std::vector<uint32_t> perm(vecs.size());
      std::iota(perm.begin(), perm.end(), 0);
      std::shuffle(perm.begin(), perm.end(), rng);

      for (uint32_t i = 0; i < n_sample; ++i) {
        sample_indices.push_back(vecs[perm[i]]);
      }
    }

    training_data.resize(sample_indices.size() * dim);
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      std::memcpy(training_data.data() + i * dim,
                  data + static_cast<size_t>(sample_indices[i]) * dim,
                  dim * sizeof(float));
    }
  }

  // Train SAQ quantizer
  SAQQuantizer quantizer;
  uint32_t training_size = n_vectors <= max_training
      ? n_vectors
      : static_cast<uint32_t>(sample_indices.size());

  std::string train_err = quantizer.Train(training_data.data(), training_size,
                                           dim, config.saq);
  if (!train_err.empty()) {
    return "SAQ training failed: " + train_err;
  }

  plan_ = quantizer.Plan();
  num_segments_ = plan_.segment_count;

  // Initialize distance estimator
  if (!distance_estimator_.Initialize(plan_.codebooks, plan_.segments,
                                       config.ivf.metric)) {
    return "Failed to initialize distance estimator";
  }

  // Step 4: Encode all vectors into clusters
  clusters_.clear();
  clusters_.resize(num_clusters);
  id_to_location_.resize(n_vectors);

  SAQEncodeConfig encode_config;
  encode_config.use_caq = true;

  for (uint32_t c = 0; c < num_clusters; ++c) {
    const auto& vec_ids = cluster_vectors[c];
    Cluster& cluster = clusters_[c];
    cluster.id = c;
    cluster.num_vectors = static_cast<uint32_t>(vec_ids.size());
    cluster.global_ids = vec_ids;
    cluster.codes.resize(vec_ids.size() * num_segments_);

    for (size_t i = 0; i < vec_ids.size(); ++i) {
      uint32_t global_id = vec_ids[i];
      const float* vec = data + static_cast<size_t>(global_id) * dim;

      if (!quantizer.Encode(vec, cluster.codes.data() + i * num_segments_,
                             encode_config)) {
        return "Failed to encode vector " + std::to_string(global_id);
      }

      id_to_location_[global_id] = {c, static_cast<uint32_t>(i)};
    }
  }

  // Step 5: Determine if FastScan can be used and pack codes
  use_fast_scan_ = config.ivf.use_fast_scan;
  if (use_fast_scan_) {
    // Check if all segments have uniform centroid counts suitable for FastScan
    std::vector<uint32_t> centroids_per_segment(num_segments_);
    uint32_t max_centroids = 0;
    for (uint32_t s = 0; s < num_segments_; ++s) {
      // Number of centroids = 2^bits or from codebook
      centroids_per_segment[s] = plan_.codebooks[s].centroids;
      max_centroids = std::max(max_centroids, centroids_per_segment[s]);
    }

    if (!CanUseFastScan(centroids_per_segment.data(), num_segments_)) {
      // Fall back to standard scanning
      use_fast_scan_ = false;
    } else {
      fast_scan_bits_ = RecommendedFastScanBits(max_centroids);
      if (fast_scan_bits_ == 0) {
        use_fast_scan_ = false;
      }
    }
  }

  if (use_fast_scan_) {
    // Pack codes for each cluster into FastScan layout
    for (uint32_t c = 0; c < num_clusters; ++c) {
      Cluster& cluster = clusters_[c];
      if (cluster.num_vectors == 0) continue;

      bool pack_ok = false;
      if (fast_scan_bits_ == 4) {
        pack_ok = PackCodes4bit(cluster.codes.data(), cluster.num_vectors,
                                 num_segments_, cluster.packed_codes);
      } else if (fast_scan_bits_ == 8) {
        pack_ok = PackCodes8bit(cluster.codes.data(), cluster.num_vectors,
                                 num_segments_, cluster.packed_codes);
      }

      if (!pack_ok) {
        // If packing fails, disable FastScan for this index
        use_fast_scan_ = false;
        break;
      }
    }
  }

  built_ = true;
  return "";
}

void IVFIndex::Search(const float* query, uint32_t k,
                       std::vector<IVFSearchResult>& results,
                       uint32_t nprobe) const {
  results.clear();

  if (!built_ || query == nullptr || k == 0) {
    return;
  }

  if (nprobe == 0) {
    nprobe = default_nprobe_;
  }
  nprobe = std::min(nprobe, static_cast<uint32_t>(clusters_.size()));

  // Find nearest clusters
  std::vector<ClusterCandidate> candidates;
  initializer_->FindNearestClusters(query, dim_, nprobe, candidates);

  // Precompute distance table
  DistanceTable table = distance_estimator_.ComputeDistanceTable(query, dim_);

  // Vector used as heap for k-NN results
  std::vector<IVFSearchResult> heap;
  heap.reserve(k);

  // Prepare FastScan LUT if enabled
  FastScanLUT fast_lut;
  if (use_fast_scan_) {
    PackLUTForFastScan(table, fast_lut);
  }

  // Scan each candidate cluster
  for (const auto& candidate : candidates) {
    const Cluster& cluster = clusters_[candidate.id];
    if (use_fast_scan_ && cluster.packed_codes.num_vectors > 0) {
      ScanClusterFastScan(cluster, fast_lut, k, heap);
    } else {
      ScanCluster(cluster, table, k, heap);
    }
  }

  // Sort final results
  std::sort(heap.begin(), heap.end());
  results = std::move(heap);
}

void IVFIndex::SearchBatch(const float* queries, uint32_t n_queries,
                            uint32_t k,
                            std::vector<std::vector<IVFSearchResult>>& results,
                            uint32_t nprobe) const {
  results.clear();
  results.resize(n_queries);

  if (!built_ || queries == nullptr || k == 0) {
    return;
  }

  // Process each query
  for (uint32_t q = 0; q < n_queries; ++q) {
    const float* query = queries + static_cast<size_t>(q) * dim_;
    Search(query, k, results[q], nprobe);
  }
}

void IVFIndex::ScanCluster(const Cluster& cluster, const DistanceTable& table,
                            uint32_t k,
                            std::vector<IVFSearchResult>& heap) const {
  // Using vector as pseudo-heap with reordering
  for (uint32_t i = 0; i < cluster.num_vectors; ++i) {
    const uint32_t* codes = cluster.codes.data() +
                            static_cast<size_t>(i) * num_segments_;

    float dist = distance_estimator_.EstimateDistance(table, codes);
    uint32_t global_id = cluster.global_ids[i];

    if (heap.size() < k) {
      heap.push_back({global_id, dist});
      std::push_heap(heap.begin(), heap.end());
    } else if (dist < heap[0].distance) {
      std::pop_heap(heap.begin(), heap.end());
      heap.back() = {global_id, dist};
      std::push_heap(heap.begin(), heap.end());
    }
  }
}

void IVFIndex::ScanClusterFastScan(const Cluster& cluster, 
                                    const FastScanLUT& lut,
                                    uint32_t k,
                                    std::vector<IVFSearchResult>& heap) const {
  // Compute all distances using FastScan
  std::vector<float> distances(cluster.num_vectors);
  
  if (fast_scan_bits_ == 4) {
    FastScanEstimate4bit(cluster.packed_codes, lut, distances.data());
  } else if (fast_scan_bits_ == 8) {
    FastScanEstimate8bit(cluster.packed_codes, lut, distances.data());
  }

  // Update heap with results
  for (uint32_t i = 0; i < cluster.num_vectors; ++i) {
    float dist = distances[i];
    uint32_t global_id = cluster.global_ids[i];

    if (heap.size() < k) {
      heap.push_back({global_id, dist});
      std::push_heap(heap.begin(), heap.end());
    } else if (dist < heap[0].distance) {
      std::pop_heap(heap.begin(), heap.end());
      heap.back() = {global_id, dist};
      std::push_heap(heap.begin(), heap.end());
    }
  }
}

void IVFIndex::PackLUTForFastScan(const DistanceTable& table, 
                                   FastScanLUT& lut) const {
  // Prepare table pointers and centroid counts
  std::vector<const float*> table_ptrs(table.segment_tables.size());
  std::vector<uint32_t> centroids_per_seg(table.segment_tables.size());
  
  for (size_t s = 0; s < table.segment_tables.size(); ++s) {
    table_ptrs[s] = table.segment_tables[s].distances.data();
    centroids_per_seg[s] = table.segment_tables[s].num_centroids;
  }

  if (fast_scan_bits_ == 4) {
    // Use variable centroid count version for SAQ compatibility
    PackLUT4bitVariable(table_ptrs.data(), 
                         static_cast<uint32_t>(table.segment_tables.size()),
                         centroids_per_seg.data(),
                         lut);
  } else if (fast_scan_bits_ == 8) {
    // Use variable centroid count version for SAQ compatibility
    PackLUT8bitVariable(table_ptrs.data(), 
                         static_cast<uint32_t>(table.segment_tables.size()),
                         centroids_per_seg.data(),
                         lut);
  }
}

bool IVFIndex::Reconstruct(uint32_t global_id, float* output) const {
  if (!built_ || global_id >= num_vectors_ || output == nullptr) {
    return false;
  }

  auto [cluster_id, local_idx] = id_to_location_[global_id];
  const Cluster& cluster = clusters_[cluster_id];
  const uint32_t* codes = cluster.codes.data() +
                          static_cast<size_t>(local_idx) * num_segments_;

  return distance_estimator_.Reconstruct(codes, output);
}

void IVFIndex::GetClusterStats(uint32_t& min_size, uint32_t& max_size,
                                float& avg_size) const {
  if (clusters_.empty()) {
    min_size = max_size = 0;
    avg_size = 0.0f;
    return;
  }

  min_size = std::numeric_limits<uint32_t>::max();
  max_size = 0;
  uint64_t total = 0;

  for (const auto& cluster : clusters_) {
    min_size = std::min(min_size, cluster.num_vectors);
    max_size = std::max(max_size, cluster.num_vectors);
    total += cluster.num_vectors;
  }

  avg_size = static_cast<float>(total) / static_cast<float>(clusters_.size());
}

bool IVFIndex::Save(const std::string& filename) const {
  if (!built_) {
    return false;
  }

  std::ofstream out(filename, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }

  // Magic number and version
  const uint32_t magic = 0x53415149;  // "SAQI"
  const uint32_t version = 1;
  out.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

  // Metadata
  out.write(reinterpret_cast<const char*>(&num_vectors_), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&dim_), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&num_segments_), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&default_nprobe_), sizeof(uint32_t));

  uint32_t num_clusters = static_cast<uint32_t>(clusters_.size());
  out.write(reinterpret_cast<const char*>(&num_clusters), sizeof(uint32_t));

  // Initializer type and data
  bool is_hnsw = (dynamic_cast<HNSWInitializer*>(initializer_.get()) != nullptr);
  out.write(reinterpret_cast<const char*>(&is_hnsw), sizeof(bool));

  auto init_data = initializer_->Serialize();
  uint32_t init_size = static_cast<uint32_t>(init_data.size());
  out.write(reinterpret_cast<const char*>(&init_size), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(init_data.data()), init_size);

  // Quantization plan
  auto plan_data = plan_.SerializeBinary();
  uint32_t plan_size = static_cast<uint32_t>(plan_data.size());
  out.write(reinterpret_cast<const char*>(&plan_size), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(plan_data.data()), plan_size);

  // Clusters
  for (const auto& cluster : clusters_) {
    out.write(reinterpret_cast<const char*>(&cluster.id), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&cluster.num_vectors), sizeof(uint32_t));

    out.write(reinterpret_cast<const char*>(cluster.global_ids.data()),
              cluster.global_ids.size() * sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(cluster.codes.data()),
              cluster.codes.size() * sizeof(uint32_t));
  }

  // ID to location map
  out.write(reinterpret_cast<const char*>(id_to_location_.data()),
            id_to_location_.size() * sizeof(std::pair<uint32_t, uint32_t>));

  return out.good();
}

std::string IVFIndex::Load(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    return "Cannot open file: " + filename;
  }

  built_ = false;

  // Magic number and version
  uint32_t magic, version;
  in.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

  if (magic != 0x53415149) {
    return "Invalid file format";
  }
  if (version != 1) {
    return "Unsupported version: " + std::to_string(version);
  }

  // Metadata
  uint32_t num_clusters;
  in.read(reinterpret_cast<char*>(&num_vectors_), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&dim_), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&num_segments_), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&default_nprobe_), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&num_clusters), sizeof(uint32_t));

  // Initializer
  bool is_hnsw;
  in.read(reinterpret_cast<char*>(&is_hnsw), sizeof(bool));

  uint32_t init_size;
  in.read(reinterpret_cast<char*>(&init_size), sizeof(uint32_t));
  std::vector<uint8_t> init_data(init_size);
  in.read(reinterpret_cast<char*>(init_data.data()), init_size);

  if (is_hnsw) {
    auto hnsw = std::make_unique<HNSWInitializer>();
    if (!hnsw->Deserialize(init_data)) {
      return "Failed to deserialize HNSW initializer";
    }
    initializer_ = std::move(hnsw);
  } else {
    auto flat = std::make_unique<FlatInitializer>();
    if (!flat->Deserialize(init_data)) {
      return "Failed to deserialize Flat initializer";
    }
    initializer_ = std::move(flat);
  }

  // Quantization plan
  uint32_t plan_size;
  in.read(reinterpret_cast<char*>(&plan_size), sizeof(uint32_t));
  std::vector<uint8_t> plan_data(plan_size);
  in.read(reinterpret_cast<char*>(plan_data.data()), plan_size);

  std::string plan_error;
  if (!plan_.DeserializeBinary(plan_data, &plan_error)) {
    return "Failed to deserialize quantization plan: " + plan_error;
  }

  // Initialize distance estimator
  if (!distance_estimator_.Initialize(plan_.codebooks, plan_.segments)) {
    return "Failed to initialize distance estimator";
  }

  // Clusters
  clusters_.clear();
  clusters_.resize(num_clusters);

  for (auto& cluster : clusters_) {
    in.read(reinterpret_cast<char*>(&cluster.id), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&cluster.num_vectors), sizeof(uint32_t));

    cluster.global_ids.resize(cluster.num_vectors);
    cluster.codes.resize(static_cast<size_t>(cluster.num_vectors) * num_segments_);

    in.read(reinterpret_cast<char*>(cluster.global_ids.data()),
            cluster.global_ids.size() * sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(cluster.codes.data()),
            cluster.codes.size() * sizeof(uint32_t));
  }

  // ID to location map
  id_to_location_.resize(num_vectors_);
  in.read(reinterpret_cast<char*>(id_to_location_.data()),
          id_to_location_.size() * sizeof(std::pair<uint32_t, uint32_t>));

  if (!in.good()) {
    return "Error reading file";
  }

  // Repack codes for FastScan if applicable
  use_fast_scan_ = false;
  fast_scan_bits_ = 0;
  
  // Check if FastScan can be used
  std::vector<uint32_t> centroids_per_segment(num_segments_);
  uint32_t max_centroids = 0;
  for (uint32_t s = 0; s < num_segments_; ++s) {
    // Number of centroids from codebook
    centroids_per_segment[s] = plan_.codebooks[s].centroids;
    max_centroids = std::max(max_centroids, centroids_per_segment[s]);
  }

  if (CanUseFastScan(centroids_per_segment.data(), num_segments_)) {
    fast_scan_bits_ = RecommendedFastScanBits(max_centroids);
    if (fast_scan_bits_ != 0) {
      use_fast_scan_ = true;

      // Pack codes for each cluster
      for (auto& cluster : clusters_) {
        if (cluster.num_vectors == 0) continue;

        bool pack_ok = false;
        if (fast_scan_bits_ == 4) {
          pack_ok = PackCodes4bit(cluster.codes.data(), cluster.num_vectors,
                                   num_segments_, cluster.packed_codes);
        } else if (fast_scan_bits_ == 8) {
          pack_ok = PackCodes8bit(cluster.codes.data(), cluster.num_vectors,
                                   num_segments_, cluster.packed_codes);
        }

        if (!pack_ok) {
          use_fast_scan_ = false;
          break;
        }
      }
    }
  }

  built_ = true;
  return "";
}

ClusterAssignment IVFIndex::AssignToCluster(const float* vector) const {
  std::vector<ClusterCandidate> candidates;
  initializer_->FindNearestClusters(vector, dim_, 1, candidates);

  if (candidates.empty()) {
    return {0, std::numeric_limits<float>::max()};
  }

  return {candidates[0].id, candidates[0].distance};
}

void IVFIndex::ComputeResidual(const float* vector, const float* centroid,
                                float* residual) const {
  for (uint32_t i = 0; i < dim_; ++i) {
    residual[i] = vector[i] - centroid[i];
  }
}

}  // namespace saq
