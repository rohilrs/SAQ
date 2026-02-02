/// @file saq_quantizer.cpp
/// @brief Implementation of the main SAQ quantizer.

#include "saq/saq_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

namespace saq {

namespace {

/// @brief Compute squared L2 distance.
float SquaredL2(const float* a, const float* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

/// @brief Simple k-means clustering for a single segment.
/// @param data Segment data, row-major (n_vectors × dim).
/// @param n_vectors Number of vectors.
/// @param dim Segment dimensionality.
/// @param k Number of centroids.
/// @param max_iter Maximum iterations.
/// @param seed Random seed.
/// @return Centroids, row-major (k × dim).
std::vector<float> KMeans(const float* data, uint32_t n_vectors, uint32_t dim,
                           uint32_t k, uint32_t max_iter, uint32_t seed) {
  if (n_vectors == 0 || dim == 0 || k == 0) {
    return {};
  }

  // Clamp k to available vectors
  k = std::min(k, n_vectors);

  std::vector<float> centroids(static_cast<size_t>(k) * dim);
  std::vector<uint32_t> assignments(n_vectors);
  std::vector<uint32_t> counts(k);

  // Initialize centroids with k-means++ style
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> uniform(0, n_vectors - 1);

  // First centroid: random
  uint32_t first = uniform(rng);
  std::memcpy(centroids.data(), data + static_cast<size_t>(first) * dim,
              dim * sizeof(float));

  // Remaining centroids: probability proportional to distance
  std::vector<float> min_dists(n_vectors, std::numeric_limits<float>::max());

  for (uint32_t c = 1; c < k; ++c) {
    // Update min distances
    const float* prev_centroid = centroids.data() + static_cast<size_t>(c - 1) * dim;
    float total_dist = 0.0f;

    for (uint32_t v = 0; v < n_vectors; ++v) {
      float d = SquaredL2(data + static_cast<size_t>(v) * dim, prev_centroid, dim);
      min_dists[v] = std::min(min_dists[v], d);
      total_dist += min_dists[v];
    }

    // Sample proportionally
    std::uniform_real_distribution<float> sample(0.0f, total_dist);
    float threshold = sample(rng);
    float cumsum = 0.0f;
    uint32_t selected = 0;

    for (uint32_t v = 0; v < n_vectors; ++v) {
      cumsum += min_dists[v];
      if (cumsum >= threshold) {
        selected = v;
        break;
      }
    }

    std::memcpy(centroids.data() + static_cast<size_t>(c) * dim,
                data + static_cast<size_t>(selected) * dim,
                dim * sizeof(float));
  }

  // K-means iterations
  for (uint32_t iter = 0; iter < max_iter; ++iter) {
    // Assignment step
    bool changed = false;
    for (uint32_t v = 0; v < n_vectors; ++v) {
      const float* vec = data + static_cast<size_t>(v) * dim;
      uint32_t best = 0;
      float best_dist = std::numeric_limits<float>::max();

      for (uint32_t c = 0; c < k; ++c) {
        float d = SquaredL2(vec, centroids.data() + static_cast<size_t>(c) * dim, dim);
        if (d < best_dist) {
          best_dist = d;
          best = c;
        }
      }

      if (assignments[v] != best) {
        assignments[v] = best;
        changed = true;
      }
    }

    if (!changed && iter > 0) {
      break;  // Converged
    }

    // Update step
    std::fill(centroids.begin(), centroids.end(), 0.0f);
    std::fill(counts.begin(), counts.end(), 0);

    for (uint32_t v = 0; v < n_vectors; ++v) {
      uint32_t c = assignments[v];
      counts[c]++;
      const float* vec = data + static_cast<size_t>(v) * dim;
      float* centroid = centroids.data() + static_cast<size_t>(c) * dim;
      for (uint32_t d = 0; d < dim; ++d) {
        centroid[d] += vec[d];
      }
    }

    for (uint32_t c = 0; c < k; ++c) {
      if (counts[c] > 0) {
        float* centroid = centroids.data() + static_cast<size_t>(c) * dim;
        float inv_count = 1.0f / static_cast<float>(counts[c]);
        for (uint32_t d = 0; d < dim; ++d) {
          centroid[d] *= inv_count;
        }
      } else {
        // Empty cluster: reinitialize randomly
        uint32_t random_vec = uniform(rng);
        std::memcpy(centroids.data() + static_cast<size_t>(c) * dim,
                    data + static_cast<size_t>(random_vec) * dim,
                    dim * sizeof(float));
      }
    }
  }

  return centroids;
}

/// @brief Extract segment data from full vectors.
std::vector<float> ExtractSegment(const float* data, uint32_t n_vectors,
                                   uint32_t full_dim, uint32_t start_dim,
                                   uint32_t seg_dim) {
  std::vector<float> segment(static_cast<size_t>(n_vectors) * seg_dim);

  for (uint32_t v = 0; v < n_vectors; ++v) {
    const float* src = data + static_cast<size_t>(v) * full_dim + start_dim;
    float* dst = segment.data() + static_cast<size_t>(v) * seg_dim;
    std::memcpy(dst, src, seg_dim * sizeof(float));
  }

  return segment;
}

}  // namespace

std::string SAQQuantizer::Train(const float* data, uint32_t n_vectors,
                                 uint32_t dim, const SAQTrainConfig& config) {
  if (data == nullptr || n_vectors == 0 || dim == 0) {
    return "Invalid training data";
  }

  if (config.total_bits == 0) {
    return "Total bits must be positive";
  }

  if (config.num_segments == 0 || config.num_segments > dim) {
    return "Invalid number of segments";
  }

  trained_ = false;
  plan_ = QuantizationPlan{};
  plan_.version = 1;
  plan_.dimension = dim;
  plan_.total_bits = config.total_bits;
  plan_.seed = config.seed;
  plan_.use_pca = config.use_pca;

  // Working data (potentially PCA-transformed)
  std::vector<float> work_data;
  const float* work_ptr = data;
  uint32_t work_dim = dim;

  // Step 1: Optional PCA
  if (config.use_pca && config.pca_dim > 0 && config.pca_dim < dim) {
    if (!pca_.Train(data, n_vectors, dim, config.pca_dim)) {
      return "PCA training failed";
    }

    work_dim = config.pca_dim;
    work_data.resize(static_cast<size_t>(n_vectors) * work_dim);

    pca_.ProjectBatch(data, work_data.data(), n_vectors);

    work_ptr = work_data.data();
    pca_.ExportParams(&plan_.pca);
  }

  // Step 2: Dimension segmentation
  DimensionSegmenter segmenter;
  if (!segmenter.ComputeStats(work_ptr, n_vectors, work_dim)) {
    return "Failed to compute dimension statistics";
  }

  SegmentationConfig seg_config;
  seg_config.strategy = config.segmentation_strategy;
  seg_config.num_segments = config.num_segments;
  seg_config.min_dims_per_segment = 1;

  auto seg_result = segmenter.Segment(seg_config);
  if (!seg_result.IsValid()) {
    return "Segmentation failed: " + seg_result.error;
  }

  // Step 3: Bit allocation
  BitAllocatorDP allocator;
  std::vector<uint32_t> segment_dims;
  segment_dims.reserve(seg_result.segments.size());
  for (const auto& seg : seg_result.segments) {
    segment_dims.push_back(seg.dim_count);
  }

  BitAllocationConfig bit_config;
  bit_config.total_bits = config.total_bits;
  bit_config.min_bits_per_segment = config.min_bits_per_segment;
  bit_config.max_bits_per_segment = config.max_bits_per_segment;

  auto bit_result = allocator.Allocate(seg_result.segment_variances,
                                        segment_dims, bit_config);
  if (!bit_result.IsValid()) {
    return "Bit allocation failed: " + bit_result.error;
  }

  // Apply bit allocation to segments
  if (!BitAllocatorDP::ApplyAllocation(bit_result, seg_result.segments)) {
    return "Failed to apply bit allocation";
  }

  plan_.segments = seg_result.segments;
  plan_.segment_count = static_cast<uint32_t>(plan_.segments.size());

  // Step 4: Train codebooks
  std::string codebook_error = TrainCodebooks(work_ptr, n_vectors, work_dim, config);
  if (!codebook_error.empty()) {
    return codebook_error;
  }

  // Step 5: Initialize components
  std::string init_error = InitializeComponents();
  if (!init_error.empty()) {
    return init_error;
  }

  trained_ = true;
  return "";
}

std::string SAQQuantizer::TrainCodebooks(const float* data, uint32_t n_vectors,
                                          uint32_t dim, const SAQTrainConfig& config) {
  plan_.codebooks.clear();
  plan_.codebooks.reserve(plan_.segments.size());

  for (size_t s = 0; s < plan_.segments.size(); ++s) {
    const Segment& seg = plan_.segments[s];

    if (seg.bits == 0) {
      // No bits = 1 centroid (mean)
      Codebook cb;
      cb.segment_id = seg.id;
      cb.bits = 0;
      cb.centroids = 1;
      cb.dim_count = seg.dim_count;
      cb.data.resize(seg.dim_count, 0.0f);

      // Compute mean
      for (uint32_t v = 0; v < n_vectors; ++v) {
        const float* vec = data + static_cast<size_t>(v) * dim + seg.start_dim;
        for (uint32_t d = 0; d < seg.dim_count; ++d) {
          cb.data[d] += vec[d];
        }
      }
      float inv_n = 1.0f / static_cast<float>(n_vectors);
      for (float& val : cb.data) {
        val *= inv_n;
      }

      plan_.codebooks.push_back(std::move(cb));
      continue;
    }

    uint32_t num_centroids = 1u << seg.bits;

    // Extract segment data
    auto segment_data = ExtractSegment(data, n_vectors, dim,
                                         seg.start_dim, seg.dim_count);

    // Clamp centroids to available training data
    uint32_t actual_centroids = std::min(num_centroids, n_vectors);

    // Run k-means
    auto centroids = KMeans(segment_data.data(), n_vectors, seg.dim_count,
                             actual_centroids, config.kmeans_iterations,
                             config.seed + static_cast<uint32_t>(s));

    if (centroids.empty()) {
      return "K-means failed for segment " + std::to_string(s);
    }

    Codebook cb;
    cb.segment_id = seg.id;
    cb.bits = seg.bits;
    cb.centroids = actual_centroids;
    cb.dim_count = seg.dim_count;
    cb.data = std::move(centroids);

    plan_.codebooks.push_back(std::move(cb));
  }

  plan_.codebook_count = static_cast<uint32_t>(plan_.codebooks.size());
  return "";
}

std::string SAQQuantizer::InitializeComponents() {
  // Initialize CAQ adjuster
  if (!caq_adjuster_.Initialize(plan_.codebooks, plan_.segments)) {
    return "Failed to initialize CAQ adjuster";
  }

  // Initialize distance estimator
  if (!distance_estimator_.Initialize(plan_.codebooks, plan_.segments)) {
    return "Failed to initialize distance estimator";
  }

  // Initialize PCA if enabled
  if (plan_.use_pca) {
    if (!pca_.ImportParams(plan_.pca)) {
      return "Failed to import PCA parameters";
    }
  }

  return "";
}

std::string SAQQuantizer::LoadPlan(const QuantizationPlan& plan) {
  plan_ = plan;
  trained_ = false;

  std::string error = InitializeComponents();
  if (!error.empty()) {
    return error;
  }

  trained_ = true;
  return "";
}

bool SAQQuantizer::Encode(const float* vector, uint32_t* codes,
                          const SAQEncodeConfig& config) const {
  if (!trained_ || vector == nullptr || codes == nullptr) {
    return false;
  }

  // Apply PCA if needed
  std::vector<float> projected;
  const float* work_vec = vector;

  if (plan_.use_pca) {
    projected.resize(plan_.pca.output_dim);
    pca_.Project(vector, projected.data());
    work_vec = projected.data();
  }

  // Greedy encoding
  auto encoded = caq_adjuster_.EncodeGreedy(work_vec, WorkingDim());
  if (encoded.codes.size() != plan_.segment_count) {
    return false;
  }

  // Optional CAQ refinement
  if (config.use_caq) {
    encoded = caq_adjuster_.RefineCAQ(work_vec, WorkingDim(), encoded,
                                       config.caq_config);
  }

  std::memcpy(codes, encoded.codes.data(), plan_.segment_count * sizeof(uint32_t));
  return true;
}

bool SAQQuantizer::EncodeBatch(const float* vectors, uint32_t n_vectors,
                                uint32_t* codes, const SAQEncodeConfig& config) const {
  if (!trained_ || vectors == nullptr || codes == nullptr) {
    return false;
  }

  for (uint32_t v = 0; v < n_vectors; ++v) {
    const float* vec = vectors + static_cast<size_t>(v) * plan_.dimension;
    uint32_t* vec_codes = codes + static_cast<size_t>(v) * plan_.segment_count;

    if (!Encode(vec, vec_codes, config)) {
      return false;
    }
  }

  return true;
}

bool SAQQuantizer::Decode(const uint32_t* codes, float* vector) const {
  if (!trained_ || codes == nullptr || vector == nullptr) {
    return false;
  }

  // Reconstruct in working space
  std::vector<float> work_vec(WorkingDim());
  if (!distance_estimator_.Reconstruct(codes, work_vec.data())) {
    return false;
  }

  // Inverse PCA if needed
  if (plan_.use_pca) {
    pca_.InverseProject(work_vec.data(), vector);
  } else {
    std::memcpy(vector, work_vec.data(), plan_.dimension * sizeof(float));
  }

  return true;
}

void SAQQuantizer::Search(const float* query, const uint32_t* codes,
                          uint32_t n_vectors, uint32_t k,
                          std::vector<SearchResult>& results) const {
  results.clear();

  if (!trained_ || query == nullptr || codes == nullptr || n_vectors == 0) {
    return;
  }

  k = std::min(k, n_vectors);

  // Apply PCA if needed
  std::vector<float> projected;
  const float* work_query = query;

  if (plan_.use_pca) {
    projected.resize(plan_.pca.output_dim);
    pca_.Project(query, projected.data());
    work_query = projected.data();
  }

  // Compute distance table
  auto table = distance_estimator_.ComputeDistanceTable(work_query, WorkingDim());
  if (table.segment_tables.empty()) {
    return;
  }

  // Use max-heap to track k smallest distances
  using Entry = std::pair<float, uint32_t>;  // (distance, index)
  std::priority_queue<Entry> top_k;

  const size_t num_segments = plan_.segment_count;

  for (uint32_t v = 0; v < n_vectors; ++v) {
    const uint32_t* vec_codes = codes + static_cast<size_t>(v) * num_segments;
    float dist = distance_estimator_.EstimateDistance(table, vec_codes);

    if (top_k.size() < k) {
      top_k.emplace(dist, v);
    } else if (dist < top_k.top().first) {
      top_k.pop();
      top_k.emplace(dist, v);
    }
  }

  // Extract results in sorted order
  results.resize(top_k.size());
  for (size_t i = results.size(); i > 0; --i) {
    results[i - 1].index = top_k.top().second;
    results[i - 1].distance = top_k.top().first;
    top_k.pop();
  }
}

}  // namespace saq
