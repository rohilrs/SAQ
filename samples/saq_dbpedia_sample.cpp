/// @file saq_dbpedia_sample.cpp
/// @brief End-to-end SAQ sample on DBpedia 100K dataset.
///
/// Demonstrates:
/// - Loading vectors from .fvecs format
/// - K-means clustering for IVF
/// - Building SAQ-IVF index with scalar quantization
/// - Search with varying nprobe
/// - Computing recall, QPS, and error metrics

#include "index/ivf_index.h"
#include "saq/saq_quantizer.h"
#include "saq/simd_kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_set>
#include <vector>

using namespace saq;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// File I/O Utilities
// ============================================================================

/// @brief Read vectors from .fvecs format.
bool ReadFvecs(const std::string& filename, std::vector<float>& data,
               uint32_t& n_vectors, uint32_t& dim) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Cannot open: " << filename << std::endl;
    return false;
  }
  
  // Read first dimension
  int32_t d;
  file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
  if (d <= 0) return false;
  dim = static_cast<uint32_t>(d);
  
  // Get file size to compute number of vectors
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  size_t bytes_per_vector = sizeof(int32_t) + dim * sizeof(float);
  n_vectors = static_cast<uint32_t>(file_size / bytes_per_vector);
  
  // Read all vectors
  data.resize(static_cast<size_t>(n_vectors) * dim);
  file.seekg(0, std::ios::beg);
  
  for (uint32_t i = 0; i < n_vectors; ++i) {
    int32_t vec_dim;
    file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
    if (static_cast<uint32_t>(vec_dim) != dim) {
      std::cerr << "Dimension mismatch at vector " << i << std::endl;
      return false;
    }
    file.read(reinterpret_cast<char*>(data.data() + i * dim), dim * sizeof(float));
  }
  
  return true;
}

/// @brief Read ground truth from .ivecs format.
bool ReadIvecs(const std::string& filename, std::vector<std::vector<uint32_t>>& gt) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Cannot open: " << filename << std::endl;
    return false;
  }
  
  gt.clear();
  while (file.peek() != EOF) {
    int32_t k;
    file.read(reinterpret_cast<char*>(&k), sizeof(int32_t));
    if (file.eof()) break;
    
    std::vector<uint32_t> neighbors(k);
    file.read(reinterpret_cast<char*>(neighbors.data()), k * sizeof(int32_t));
    gt.push_back(std::move(neighbors));
  }
  
  return !gt.empty();
}

// ============================================================================
// K-means Clustering
// ============================================================================

/// @brief Simple k-means clustering.
void KMeansClustering(const float* data, uint32_t n, uint32_t dim,
                       uint32_t k, std::vector<float>& centroids,
                       std::vector<uint32_t>& assignments,
                       uint32_t max_iter = 20, uint32_t seed = 42) {
  std::mt19937 rng(seed);
  
  // Initialize centroids with k-means++
  centroids.resize(static_cast<size_t>(k) * dim);
  assignments.resize(n);
  
  // First centroid: random
  std::uniform_int_distribution<uint32_t> uniform(0, n - 1);
  uint32_t first = uniform(rng);
  std::memcpy(centroids.data(), data + first * dim, dim * sizeof(float));
  
  // Remaining centroids: k-means++ initialization
  std::vector<float> min_dists(n, std::numeric_limits<float>::max());
  
  for (uint32_t c = 1; c < k; ++c) {
    // Update min distances
    const float* prev = centroids.data() + (c - 1) * dim;
    std::vector<float> dists(n);
    simd::L2DistancesBatch(prev, data, n, dim, dists.data());
    
    float total = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
      min_dists[i] = std::min(min_dists[i], dists[i]);
      total += min_dists[i];
    }
    
    // Sample proportionally
    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng);
    float cumsum = 0.0f;
    uint32_t selected = n - 1;
    for (uint32_t i = 0; i < n; ++i) {
      cumsum += min_dists[i];
      if (cumsum >= r) {
        selected = i;
        break;
      }
    }
    
    std::memcpy(centroids.data() + c * dim, data + selected * dim, dim * sizeof(float));
    
    if ((c + 1) % 100 == 0) {
      std::cout << "  Initialized " << (c + 1) << "/" << k << " centroids\r" << std::flush;
    }
  }
  std::cout << "  Initialized " << k << "/" << k << " centroids" << std::endl;
  
  // Lloyd's algorithm
  std::vector<uint32_t> counts(k);
  std::vector<float> new_centroids(k * dim);
  
  for (uint32_t iter = 0; iter < max_iter; ++iter) {
    // Assign points to nearest centroid
    std::fill(counts.begin(), counts.end(), 0);
    std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
    
    for (uint32_t i = 0; i < n; ++i) {
      const float* vec = data + i * dim;
      
      // Find nearest centroid
      float best_dist = std::numeric_limits<float>::max();
      uint32_t best_c = 0;
      
      for (uint32_t c = 0; c < k; ++c) {
        float dist = 0.0f;
        const float* cent = centroids.data() + c * dim;
        for (uint32_t d = 0; d < dim; ++d) {
          float diff = vec[d] - cent[d];
          dist += diff * diff;
        }
        if (dist < best_dist) {
          best_dist = dist;
          best_c = c;
        }
      }
      
      assignments[i] = best_c;
      counts[best_c]++;
      
      // Accumulate for new centroid
      for (uint32_t d = 0; d < dim; ++d) {
        new_centroids[best_c * dim + d] += vec[d];
      }
    }
    
    // Update centroids
    for (uint32_t c = 0; c < k; ++c) {
      if (counts[c] > 0) {
        float inv = 1.0f / counts[c];
        for (uint32_t d = 0; d < dim; ++d) {
          centroids[c * dim + d] = new_centroids[c * dim + d] * inv;
        }
      }
    }
    
    std::cout << "  K-means iteration " << (iter + 1) << "/" << max_iter << "\r" << std::flush;
  }
  std::cout << std::endl;
}

// ============================================================================
// Evaluation Metrics
// ============================================================================

/// @brief Compute recall@k.
float ComputeRecall(const std::vector<std::vector<IVFSearchResult>>& results,
                    const std::vector<std::vector<uint32_t>>& gt,
                    uint32_t k) {
  uint32_t hits = 0;
  uint32_t total = 0;
  
  for (size_t q = 0; q < results.size(); ++q) {
    std::unordered_set<uint32_t> gt_set;
    for (uint32_t i = 0; i < std::min(k, static_cast<uint32_t>(gt[q].size())); ++i) {
      gt_set.insert(gt[q][i]);
    }
    
    for (size_t i = 0; i < std::min(static_cast<size_t>(k), results[q].size()); ++i) {
      if (gt_set.count(results[q][i].index)) {
        hits++;
      }
    }
    total += k;
  }
  
  return static_cast<float>(hits) / static_cast<float>(total);
}

/// @brief Compute average relative error of distances.
float ComputeRelativeError(const std::vector<std::vector<IVFSearchResult>>& results,
                            const float* base_vectors, const float* queries,
                            uint32_t dim, uint32_t k) {
  float total_error = 0.0f;
  uint32_t count = 0;
  
  for (size_t q = 0; q < results.size(); ++q) {
    const float* query = queries + q * dim;
    
    for (size_t i = 0; i < std::min(static_cast<size_t>(k), results[q].size()); ++i) {
      uint32_t idx = results[q][i].index;
      float estimated_dist = results[q][i].distance;
      
      // Compute true distance
      const float* vec = base_vectors + idx * dim;
      float true_dist = 0.0f;
      for (uint32_t d = 0; d < dim; ++d) {
        float diff = query[d] - vec[d];
        true_dist += diff * diff;
      }
      
      // Relative error
      if (true_dist > 1e-10f) {
        total_error += std::abs(estimated_dist - true_dist) / true_dist;
        count++;
      }
    }
  }
  
  return count > 0 ? total_error / count : 0.0f;
}

/// @brief Compute compression ratio.
float ComputeRatio(uint32_t dim, uint32_t total_bits) {
  uint32_t original_bits = dim * 32;  // float32
  return static_cast<float>(original_bits) / static_cast<float>(total_bits);
}

// ============================================================================
// Result Output
// ============================================================================

struct BenchmarkResult {
  uint32_t nprobe;
  uint32_t k;
  float recall;
  float qps;
  float avg_relative_error;
  float ratio;
  double search_time_ms;
};

void WriteResults(const std::string& filename,
                  const std::vector<BenchmarkResult>& results,
                  uint32_t n_base, uint32_t n_queries, uint32_t dim,
                  uint32_t total_bits, uint32_t num_clusters,
                  double build_time_s) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    std::cerr << "Cannot write results to: " << filename << std::endl;
    return;
  }
  
  out << "================================================================================\n";
  out << "SAQ-IVF Benchmark Results\n";
  out << "================================================================================\n\n";
  
  out << "Dataset Configuration:\n";
  out << "  Base vectors:    " << n_base << "\n";
  out << "  Query vectors:   " << n_queries << "\n";
  out << "  Dimension:       " << dim << "\n\n";
  
  out << "Index Configuration:\n";
  out << "  Clusters (K):    " << num_clusters << "\n";
  out << "  Total bits:      " << total_bits << "\n";
  out << "  Compression:     " << std::fixed << std::setprecision(1) 
      << ComputeRatio(dim, total_bits) << "x\n";
  out << "  Build time:      " << std::fixed << std::setprecision(2) 
      << build_time_s << " seconds\n\n";
  
  out << "Search Results:\n";
  out << std::string(80, '-') << "\n";
  out << std::setw(8) << "nprobe" 
      << std::setw(8) << "k"
      << std::setw(12) << "Recall@k"
      << std::setw(12) << "QPS"
      << std::setw(16) << "Rel.Error"
      << std::setw(14) << "Search(ms)"
      << "\n";
  out << std::string(80, '-') << "\n";
  
  for (const auto& r : results) {
    out << std::setw(8) << r.nprobe
        << std::setw(8) << r.k
        << std::setw(11) << std::fixed << std::setprecision(2) << (r.recall * 100) << "%"
        << std::setw(12) << std::fixed << std::setprecision(1) << r.qps
        << std::setw(15) << std::fixed << std::setprecision(4) << r.avg_relative_error
        << std::setw(14) << std::fixed << std::setprecision(2) << r.search_time_ms
        << "\n";
  }
  out << std::string(80, '-') << "\n\n";
  
  out << "Legend:\n";
  out << "  nprobe:      Number of clusters searched\n";
  out << "  k:           Number of nearest neighbors returned\n";
  out << "  Recall@k:    Percentage of true k-NN found\n";
  out << "  QPS:         Queries per second\n";
  out << "  Rel.Error:   Average relative error of estimated distances\n";
  out << "  Search(ms):  Total search time for all queries\n";
  
  out.close();
  std::cout << "Results written to: " << filename << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  std::cout << "================================================================================\n";
  std::cout << "SAQ-IVF Sample: DBpedia 100K Dataset\n";
  std::cout << "================================================================================\n\n";
  
  // Paths
  std::string data_dir = "data/datasets/dbpedia_100k";
  std::string results_dir = "results/saq";
  
  if (argc > 1) data_dir = argv[1];
  if (argc > 2) results_dir = argv[2];
  
  // Load data
  std::cout << "[1/5] Loading data...\n";
  
  std::vector<float> base_vectors, query_vectors;
  uint32_t n_base, n_queries, dim, qdim;
  
  if (!ReadFvecs(data_dir + "/vectors.fvecs", base_vectors, n_base, dim)) {
    std::cerr << "Failed to load base vectors\n";
    return 1;
  }
  std::cout << "  Base vectors: " << n_base << " x " << dim << "\n";
  
  if (!ReadFvecs(data_dir + "/queries.fvecs", query_vectors, n_queries, qdim)) {
    std::cerr << "Failed to load queries\n";
    return 1;
  }
  std::cout << "  Queries: " << n_queries << " x " << qdim << "\n";
  
  if (dim != qdim) {
    std::cerr << "Dimension mismatch!\n";
    return 1;
  }
  
  std::vector<std::vector<uint32_t>> ground_truth;
  if (!ReadIvecs(data_dir + "/groundtruth.ivecs", ground_truth)) {
    std::cerr << "Failed to load ground truth\n";
    return 1;
  }
  std::cout << "  Ground truth: " << ground_truth.size() << " queries\n\n";
  
  // Configuration
  const uint32_t num_clusters = static_cast<uint32_t>(4 * std::sqrt(n_base));
  const uint32_t total_bits = 64;  // 64 bits per vector (48x compression for 1536d)
  
  std::cout << "[2/5] Clustering (" << num_clusters << " clusters)...\n";
  
  std::vector<float> centroids;
  std::vector<uint32_t> assignments;
  
  auto cluster_start = Clock::now();
  KMeansClustering(base_vectors.data(), n_base, dim, num_clusters,
                    centroids, assignments, 15, 42);
  auto cluster_end = Clock::now();
  double cluster_time = std::chrono::duration<double>(cluster_end - cluster_start).count();
  std::cout << "  Clustering time: " << std::fixed << std::setprecision(2) 
            << cluster_time << " seconds\n\n";
  
  // Build index
  std::cout << "[3/5] Building SAQ-IVF index...\n";
  
  IVFIndex index;
  IVFTrainConfig config;
  config.ivf.num_clusters = num_clusters;
  config.ivf.nprobe = 32;
  config.saq.total_bits = total_bits;
  config.seed = 42;
  
  auto build_start = Clock::now();
  std::string err = index.Build(base_vectors.data(), n_base, dim,
                                 centroids.data(), assignments.data(), config);
  auto build_end = Clock::now();
  
  if (!err.empty()) {
    std::cerr << "Build failed: " << err << std::endl;
    return 1;
  }
  
  double build_time = std::chrono::duration<double>(build_end - build_start).count();
  std::cout << "  Build time: " << std::fixed << std::setprecision(2) 
            << build_time << " seconds\n";
  std::cout << std::endl;
  
  // Benchmark with varying nprobe
  std::cout << "[4/5] Running search benchmarks...\n";
  
  std::vector<uint32_t> nprobe_values = {1, 2, 4, 8, 16, 32, 64, 128};
  std::vector<uint32_t> k_values = {1, 10, 100};
  std::vector<BenchmarkResult> results;
  
  for (uint32_t nprobe : nprobe_values) {
    // Warmup
    std::vector<std::vector<IVFSearchResult>> warmup_results;
    index.SearchBatch(query_vectors.data(), 10, 10, warmup_results, nprobe);
    
    // Actual search
    std::vector<std::vector<IVFSearchResult>> search_results;
    
    auto search_start = Clock::now();
    index.SearchBatch(query_vectors.data(), n_queries, 100, search_results, nprobe);
    auto search_end = Clock::now();
    
    double search_time_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();
    double qps = n_queries / (search_time_ms / 1000.0);
    
    // Compute metrics for each k
    for (uint32_t k : k_values) {
      float recall = ComputeRecall(search_results, ground_truth, k);
      float rel_error = ComputeRelativeError(search_results, base_vectors.data(),
                                              query_vectors.data(), dim, k);
      
      BenchmarkResult r;
      r.nprobe = nprobe;
      r.k = k;
      r.recall = recall;
      r.qps = static_cast<float>(qps);
      r.avg_relative_error = rel_error;
      r.ratio = ComputeRatio(dim, total_bits);
      r.search_time_ms = search_time_ms;
      results.push_back(r);
      
      std::cout << "  nprobe=" << std::setw(3) << nprobe 
                << " k=" << std::setw(3) << k
                << " recall=" << std::fixed << std::setprecision(2) << std::setw(6) << (recall * 100) << "%"
                << " QPS=" << std::fixed << std::setprecision(0) << std::setw(6) << qps
                << "\n";
    }
  }
  
  // Write results
  std::cout << "\n[5/5] Writing results...\n";
  
  // Create results directory if needed
  #ifdef _WIN32
    std::string mkdir_cmd = "mkdir \"" + results_dir + "\" 2>nul";
  #else
    std::string mkdir_cmd = "mkdir -p \"" + results_dir + "\"";
  #endif
  system(mkdir_cmd.c_str());
  
  std::string results_file = results_dir + "/dbpedia_100k_results.txt";
  WriteResults(results_file, results, n_base, n_queries, dim,
               total_bits, num_clusters, build_time + cluster_time);
  
  std::cout << "\n================================================================================\n";
  std::cout << "Sample completed successfully!\n";
  std::cout << "================================================================================\n";
  
  return 0;
}
