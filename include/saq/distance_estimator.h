#pragma once

/// @file distance_estimator.h
/// @brief Asymmetric distance estimation for SAQ-encoded vectors.
///
/// Supports both scalar quantization (paper formula) and legacy
/// codebook-based ADC. For scalar codes, uses the SAQ paper's formula:
///   <o_bar, q> = delta * <codes, q> + q_sum * (-v_max + delta/2)

#include "saq/quantization_plan.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace saq {

/// @brief Distance metric type.
enum class DistanceMetric : uint8_t {
  kL2 = 0,           ///< Squared Euclidean distance.
  kInnerProduct = 1  ///< Negative inner product (for max inner product search).
};

/// @brief Precomputed query data for scalar distance estimation.
///
/// For each segment, stores the per-segment sum of query components.
/// Used with the paper's formula:
///   IP = sum_seg [ delta_seg * code_dot_q_seg + q_sum_seg * (-v_max + delta_seg/2) ]
struct ScalarQueryTable {
  /// Per-segment sum of query values: q_sum_seg = sum(q[i] for i in segment).
  std::vector<float> segment_q_sums;

  /// Per-segment scale factor: 2.0 / 2^B_seg (constant, independent of v_max).
  std::vector<float> segment_scales;

  /// Query vector in rotated space (stored for code_dot_q computation).
  std::vector<float> rotated_query;
};

// --- Legacy types (for codebook-based ADC) ---

/// @brief Precomputed distance table for a single segment (legacy).
struct SegmentDistanceTable {
  uint32_t segment_id = 0;
  uint32_t num_centroids = 0;
  std::vector<float> distances;
};

/// @brief Complete distance table for all segments (legacy).
struct DistanceTable {
  DistanceMetric metric = DistanceMetric::kL2;
  std::vector<SegmentDistanceTable> segment_tables;

  float Lookup(uint32_t segment_idx, uint32_t code) const {
    return segment_tables[segment_idx].distances[code];
  }

  float ComputeDistance(const uint32_t* codes) const {
    float total = 0.0f;
    for (size_t s = 0; s < segment_tables.size(); ++s) {
      total += segment_tables[s].distances[codes[s]];
    }
    return total;
  }
};

/// @brief Asymmetric distance estimator for SAQ search.
///
/// Provides both scalar-code estimation (paper formula) and legacy
/// codebook-based ADC lookup.
class DistanceEstimator {
 public:
  DistanceEstimator() = default;
  ~DistanceEstimator() = default;

  // Non-copyable, movable
  DistanceEstimator(const DistanceEstimator&) = delete;
  DistanceEstimator& operator=(const DistanceEstimator&) = delete;
  DistanceEstimator(DistanceEstimator&&) = default;
  DistanceEstimator& operator=(DistanceEstimator&&) = default;

  // --- Scalar quantization API ---

  /// @brief Initialize for scalar code distance estimation.
  /// @param segments Segment definitions (with bits_per_dim set).
  /// @param metric Distance metric.
  /// @return True on success.
  bool InitializeScalar(const std::vector<Segment>& segments,
                        DistanceMetric metric = DistanceMetric::kInnerProduct);

  /// @brief Precompute query table for scalar distance estimation.
  /// @param rotated_query Query vector in rotated space.
  /// @return Query table for fast estimation.
  ScalarQueryTable PrecomputeScalarQuery(const float* rotated_query) const;

  /// @brief Estimate inner product using the SAQ paper formula.
  ///
  /// <o_bar, q> = sum_seg [ delta * <codes_seg, q_seg> +
  ///                        q_sum_seg * (-v_max + delta/2) ]
  ///
  /// @param table Precomputed query table.
  /// @param codes Per-dimension scalar codes.
  /// @param v_max Per-vector scaling factor.
  /// @return Estimated inner product.
  float EstimateScalarIP(const ScalarQueryTable& table, const uint8_t* codes,
                         float v_max) const;

  /// @brief Batch estimate inner products for scalar codes.
  /// @param table Precomputed query table.
  /// @param codes_batch Array of per-dim code arrays.
  /// @param v_maxs Per-vector scaling factors.
  /// @param n_vectors Number of vectors.
  /// @param ips Output inner products.
  void EstimateScalarIPBatch(const ScalarQueryTable& table,
                             const uint8_t* const* codes_batch,
                             const float* v_maxs, uint32_t n_vectors,
                             float* ips) const;

  // --- Legacy codebook-based API ---

  /// @brief Initialize with codebooks and segments (legacy).
  bool Initialize(const std::vector<Codebook>& codebooks,
                  const std::vector<Segment>& segments,
                  DistanceMetric metric = DistanceMetric::kL2);

  /// @brief Precompute distance table for a query vector (legacy).
  DistanceTable ComputeDistanceTable(const float* query, uint32_t dim) const;

  /// @brief Estimate distance from query to encoded vector (legacy).
  float EstimateDistance(const DistanceTable& table,
                         const uint32_t* codes) const;

  /// @brief Batch distance estimation (legacy).
  void EstimateDistancesBatch(const DistanceTable& table,
                              const uint32_t* codes, uint32_t n_vectors,
                              float* distances) const;

  /// @brief Compute exact distance (legacy).
  float ComputeExactDistance(const float* query, const uint32_t* codes,
                             uint32_t dim) const;

  /// @brief Reconstruct vector from codes (legacy).
  bool Reconstruct(const uint32_t* codes, float* output) const;

  // --- Common accessors ---

  DistanceMetric Metric() const { return metric_; }
  uint32_t TotalDim() const { return total_dim_; }
  uint32_t NumSegments() const {
    return static_cast<uint32_t>(segments_.size());
  }
  bool IsInitialized() const { return !segments_.empty(); }

 private:
  std::vector<Codebook> codebooks_;  ///< Legacy codebooks.
  std::vector<Segment> segments_;
  DistanceMetric metric_ = DistanceMetric::kL2;
  uint32_t total_dim_ = 0;
};

}  // namespace saq
