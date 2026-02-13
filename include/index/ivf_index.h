#pragma once

/// @file ivf_index.h
/// @brief Inverted File Index (IVF) for scalable SAQ search.
///
/// Implements a clustering-based index structure that partitions the
/// database into clusters. At query time, only a subset of clusters
/// (nprobe) are searched, enabling sublinear search time for large
/// databases (1M-10M+ vectors).
///
/// Architecture follows the RaBitQ-Library pattern:
/// - FlatInitializer: brute-force centroid search for K < 20,000
/// - HNSWInitializer: HNSW-based centroid search for K >= 20,000
///
/// Uses scalar quantization with per-dimension codes and per-vector
/// v_max scaling factors.

#include "saq/distance_estimator.h"
#include "saq/quantization_plan.h"
#include "saq/saq_quantizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace saq {

/// @brief Configuration for IVF index construction.
struct IVFConfig {
  /// Number of clusters (partitions). Recommend: 4 * sqrt(n_vectors).
  uint32_t num_clusters = 1024;

  /// Number of clusters to search at query time.
  uint32_t nprobe = 32;

  /// Maximum vectors per cluster for training subsampling.
  uint32_t max_vectors_per_cluster = 256;

  /// Whether to use HNSW for centroid search (auto-selected based on K).
  bool use_hnsw_initializer = false;

  /// HNSW M parameter (edges per node).
  uint32_t hnsw_m = 16;

  /// HNSW ef_construction parameter.
  uint32_t hnsw_ef_construction = 200;

  /// Distance metric.
  DistanceMetric metric = DistanceMetric::kL2;
};

/// @brief Training configuration for IVF + SAQ.
struct IVFTrainConfig {
  /// IVF configuration.
  IVFConfig ivf;

  /// SAQ training configuration.
  SAQTrainConfig saq;

  /// Random seed.
  uint32_t seed = 42;
};

/// @brief Cluster assignment for a vector.
struct ClusterAssignment {
  uint32_t cluster_id = 0;  ///< Assigned cluster.
  float distance = 0.0f;    ///< Distance to centroid.
};

/// @brief Candidate cluster for search.
struct ClusterCandidate {
  uint32_t id = 0;          ///< Cluster ID.
  float distance = 0.0f;    ///< Distance from query to centroid.

  bool operator<(const ClusterCandidate& other) const {
    return distance < other.distance;
  }
};

/// @brief Search result with index and distance.
struct IVFSearchResult {
  uint32_t index = 0;       ///< Global vector index.
  float distance = 0.0f;    ///< Estimated distance.

  bool operator<(const IVFSearchResult& other) const {
    return distance < other.distance;
  }
};

/// @brief A single cluster containing scalar-encoded vectors.
struct Cluster {
  uint32_t id = 0;                       ///< Cluster ID.
  uint32_t num_vectors = 0;              ///< Number of vectors in cluster.
  std::vector<uint32_t> global_ids;      ///< Original vector indices.
  std::vector<uint8_t> codes;            ///< Scalar codes (num_vectors × working_dim).
  std::vector<float> v_maxs;             ///< Per-vector v_max scaling factors.
  std::vector<float> norms_sq;           ///< Per-vector ||ō||² for L2 distance.
};

/// @brief Abstract base class for centroid initialization/search.
class CentroidInitializer {
 public:
  virtual ~CentroidInitializer() = default;

  /// @brief Add centroid vectors.
  /// @param centroids Centroid data (k × dim).
  /// @param k Number of centroids.
  /// @param dim Dimensionality.
  virtual bool AddCentroids(const float* centroids, uint32_t k, uint32_t dim) = 0;

  /// @brief Get centroid by ID.
  /// @param id Centroid ID.
  /// @return Pointer to centroid vector.
  virtual const float* GetCentroid(uint32_t id) const = 0;

  /// @brief Find nearest clusters to a query.
  /// @param query Query vector.
  /// @param dim Query dimensionality.
  /// @param nprobe Number of clusters to return.
  /// @param candidates Output candidates (sorted by distance).
  virtual void FindNearestClusters(const float* query, uint32_t dim,
                                   uint32_t nprobe,
                                   std::vector<ClusterCandidate>& candidates) const = 0;

  /// @brief Serialize to binary.
  virtual std::vector<uint8_t> Serialize() const = 0;

  /// @brief Deserialize from binary.
  virtual bool Deserialize(const std::vector<uint8_t>& data) = 0;
};

/// @brief Flat (brute-force) centroid initializer.
///
/// Suitable for small numbers of clusters (K < 20,000).
class FlatInitializer : public CentroidInitializer {
 public:
  FlatInitializer() = default;
  ~FlatInitializer() override = default;

  bool AddCentroids(const float* centroids, uint32_t k, uint32_t dim) override;
  const float* GetCentroid(uint32_t id) const override;
  void FindNearestClusters(const float* query, uint32_t dim, uint32_t nprobe,
                           std::vector<ClusterCandidate>& candidates) const override;
  std::vector<uint8_t> Serialize() const override;
  bool Deserialize(const std::vector<uint8_t>& data) override;

 private:
  std::vector<float> centroids_;  ///< Centroid data (k × dim).
  uint32_t num_clusters_ = 0;
  uint32_t dim_ = 0;
};

/// @brief HNSW-based centroid initializer.
///
/// Suitable for large numbers of clusters (K >= 20,000).
/// Uses a small HNSW graph for fast approximate nearest centroid search.
class HNSWInitializer : public CentroidInitializer {
 public:
  /// @brief Construct with HNSW parameters.
  /// @param m HNSW M parameter.
  /// @param ef_construction Construction-time ef.
  explicit HNSWInitializer(uint32_t m = 16, uint32_t ef_construction = 200);
  ~HNSWInitializer() override;

  // Non-copyable
  HNSWInitializer(const HNSWInitializer&) = delete;
  HNSWInitializer& operator=(const HNSWInitializer&) = delete;

  bool AddCentroids(const float* centroids, uint32_t k, uint32_t dim) override;
  const float* GetCentroid(uint32_t id) const override;
  void FindNearestClusters(const float* query, uint32_t dim, uint32_t nprobe,
                           std::vector<ClusterCandidate>& candidates) const override;
  std::vector<uint8_t> Serialize() const override;
  bool Deserialize(const std::vector<uint8_t>& data) override;

  /// @brief Set ef for search.
  void SetEf(uint32_t ef) { ef_ = ef; }

 private:
  struct HNSWGraph;  // Forward declaration for implementation hiding
  std::unique_ptr<HNSWGraph> graph_;
  std::vector<float> centroids_;  // Also store centroids for direct access
  uint32_t m_ = 16;
  uint32_t ef_construction_ = 200;
  uint32_t ef_ = 100;  // Search ef
  uint32_t num_clusters_ = 0;
  uint32_t dim_ = 0;
};

/// @brief Inverted File Index with SAQ compression.
///
/// Combines clustering-based partitioning with SAQ quantization for
/// scalable approximate nearest neighbor search on 1M-10M+ vectors.
///
/// Workflow:
/// 1. Train: Cluster vectors (external K-means), train SAQ per-cluster
/// 2. Index: Assign vectors to clusters and encode with SAQ
/// 3. Search: Find nearest clusters, scan encoded vectors
class IVFIndex {
 public:
  IVFIndex() = default;
  ~IVFIndex() = default;

  // Non-copyable, movable
  IVFIndex(const IVFIndex&) = delete;
  IVFIndex& operator=(const IVFIndex&) = delete;
  IVFIndex(IVFIndex&&) = default;
  IVFIndex& operator=(IVFIndex&&) = default;

  /// @brief Build IVF index from pre-computed clustering.
  ///
  /// @param data All vectors (n_vectors × dim).
  /// @param n_vectors Total number of vectors.
  /// @param dim Vector dimensionality.
  /// @param centroids Cluster centroids (num_clusters × dim).
  /// @param cluster_ids Cluster assignment for each vector (size: n_vectors).
  /// @param config Training configuration.
  /// @return Error message, empty on success.
  std::string Build(const float* data, uint32_t n_vectors, uint32_t dim,
                    const float* centroids, const uint32_t* cluster_ids,
                    const IVFTrainConfig& config);

  /// @brief Search for k nearest neighbors.
  ///
  /// @param query Query vector (dim).
  /// @param k Number of results.
  /// @param results Output results (sorted by distance).
  /// @param nprobe Number of clusters to search (0 = use default).
  void Search(const float* query, uint32_t k,
              std::vector<IVFSearchResult>& results,
              uint32_t nprobe = 0) const;

  /// @brief Batch search for multiple queries.
  ///
  /// @param queries Query vectors (n_queries × dim).
  /// @param n_queries Number of queries.
  /// @param k Number of results per query.
  /// @param results Output results (n_queries × k).
  /// @param nprobe Number of clusters to search.
  void SearchBatch(const float* queries, uint32_t n_queries, uint32_t k,
                   std::vector<std::vector<IVFSearchResult>>& results,
                   uint32_t nprobe = 0) const;

  /// @brief Reconstruct a vector by global ID.
  /// @param global_id Original vector index.
  /// @param output Output buffer (dim).
  /// @return True on success.
  bool Reconstruct(uint32_t global_id, float* output) const;

  /// @brief Save index to file.
  /// @param filename Output file path.
  /// @return True on success.
  bool Save(const std::string& filename) const;

  /// @brief Load index from file.
  /// @param filename Input file path.
  /// @return Error message, empty on success.
  std::string Load(const std::string& filename);

  /// @brief Get number of indexed vectors.
  uint32_t NumVectors() const { return num_vectors_; }

  /// @brief Get vector dimensionality.
  uint32_t Dimension() const { return dim_; }

  /// @brief Get number of clusters.
  uint32_t NumClusters() const { return static_cast<uint32_t>(clusters_.size()); }

  /// @brief Get default nprobe.
  uint32_t DefaultNprobe() const { return default_nprobe_; }

  /// @brief Set default nprobe.
  void SetDefaultNprobe(uint32_t nprobe) { default_nprobe_ = nprobe; }

  /// @brief Check if index is built.
  bool IsBuilt() const { return built_; }

  /// @brief Get the quantization plan.
  const QuantizationPlan& GetPlan() const { return quantizer_.Plan(); }

  /// @brief Get the quantizer.
  const SAQQuantizer& GetQuantizer() const { return quantizer_; }

  /// @brief Get cluster statistics.
  /// @param min_size Output minimum cluster size.
  /// @param max_size Output maximum cluster size.
  /// @param avg_size Output average cluster size.
  void GetClusterStats(uint32_t& min_size, uint32_t& max_size,
                       float& avg_size) const;

 private:
  /// @brief Scan a cluster using scalar distance estimation.
  void ScanClusterScalar(const Cluster& cluster,
                         const ScalarQueryTable& table,
                         float query_norm_sq, uint32_t k,
                         std::vector<IVFSearchResult>& heap) const;

  /// @brief Assign a vector to a cluster.
  ClusterAssignment AssignToCluster(const float* vector) const;

  // Index state
  bool built_ = false;
  uint32_t num_vectors_ = 0;
  uint32_t dim_ = 0;
  uint32_t working_dim_ = 0;
  uint32_t default_nprobe_ = 32;
  DistanceMetric metric_ = DistanceMetric::kL2;

  // Components
  std::unique_ptr<CentroidInitializer> initializer_;
  std::vector<Cluster> clusters_;
  SAQQuantizer quantizer_;
  DistanceEstimator distance_estimator_;

  // For reconstruction: map global_id -> (cluster_id, local_idx)
  std::vector<std::pair<uint32_t, uint32_t>> id_to_location_;
};

}  // namespace saq
