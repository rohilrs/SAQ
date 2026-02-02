#pragma once

/// @file distance_estimator.h
/// @brief Asymmetric distance estimation for SAQ-encoded vectors.
///
/// Implements fast distance computation between query vectors and
/// SAQ-encoded database vectors using precomputed lookup tables.
/// Supports both L2 (Euclidean) and inner product distance metrics.

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

/// @brief Precomputed distance table for a single segment.
///
/// For a query vector q and segment s with centroids C, stores:
/// - L2: ||q[s] - C[i]||² for each centroid i
/// - IP: -<q[s], C[i]> for each centroid i
struct SegmentDistanceTable {
  uint32_t segment_id = 0;    ///< Segment this table corresponds to.
  uint32_t num_centroids = 0; ///< Number of entries in the table.
  std::vector<float> distances; ///< Distance to each centroid.
};

/// @brief Complete distance table for all segments.
struct DistanceTable {
  DistanceMetric metric = DistanceMetric::kL2;
  std::vector<SegmentDistanceTable> segment_tables;

  /// @brief Look up distance for a segment and code.
  /// @param segment_idx Segment index.
  /// @param code Quantization code.
  /// @return Precomputed distance.
  float Lookup(uint32_t segment_idx, uint32_t code) const {
    return segment_tables[segment_idx].distances[code];
  }

  /// @brief Compute total distance from codes.
  /// @param codes Quantization codes, one per segment.
  /// @return Sum of segment distances.
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
/// Precomputes distance tables from a query vector to all centroids,
/// then uses table lookups to rapidly estimate distances to encoded
/// database vectors.
class DistanceEstimator {
 public:
  DistanceEstimator() = default;
  ~DistanceEstimator() = default;

  // Non-copyable, movable
  DistanceEstimator(const DistanceEstimator&) = delete;
  DistanceEstimator& operator=(const DistanceEstimator&) = delete;
  DistanceEstimator(DistanceEstimator&&) = default;
  DistanceEstimator& operator=(DistanceEstimator&&) = default;

  /// @brief Initialize with codebooks and segments.
  /// @param codebooks Codebooks for each segment.
  /// @param segments Segment definitions.
  /// @param metric Distance metric to use.
  /// @return True on success.
  bool Initialize(const std::vector<Codebook>& codebooks,
                  const std::vector<Segment>& segments,
                  DistanceMetric metric = DistanceMetric::kL2);

  /// @brief Precompute distance table for a query vector.
  /// @param query Query vector.
  /// @param dim Query dimensionality.
  /// @return Distance table for fast lookups.
  DistanceTable ComputeDistanceTable(const float* query, uint32_t dim) const;

  /// @brief Estimate distance from query to an encoded vector.
  /// @param table Precomputed distance table.
  /// @param codes Quantization codes of the database vector.
  /// @return Estimated distance.
  float EstimateDistance(const DistanceTable& table,
                         const uint32_t* codes) const;

  /// @brief Estimate distances to a batch of encoded vectors.
  /// @param table Precomputed distance table.
  /// @param codes Codes for all vectors, shape (n_vectors, num_segments).
  /// @param n_vectors Number of vectors.
  /// @param distances Output distances, size n_vectors.
  void EstimateDistancesBatch(const DistanceTable& table,
                              const uint32_t* codes, uint32_t n_vectors,
                              float* distances) const;

  /// @brief Compute exact distance (for verification).
  /// @param query Query vector.
  /// @param codes Quantization codes.
  /// @param dim Query dimensionality.
  /// @return Exact distance using reconstructed vector.
  float ComputeExactDistance(const float* query, const uint32_t* codes,
                             uint32_t dim) const;

  /// @brief Get the distance metric.
  DistanceMetric Metric() const { return metric_; }

  /// @brief Get total dimensionality.
  uint32_t TotalDim() const { return total_dim_; }

  /// @brief Get number of segments.
  uint32_t NumSegments() const { return static_cast<uint32_t>(segments_.size()); }

  /// @brief Check if initialized.
  bool IsInitialized() const { return !codebooks_.empty(); }

  /// @brief Reconstruct vector from codes.
  /// @param codes Quantization codes.
  /// @param output Output buffer.
  /// @return True on success.
  bool Reconstruct(const uint32_t* codes, float* output) const;

 private:
  std::vector<Codebook> codebooks_;
  std::vector<Segment> segments_;
  DistanceMetric metric_ = DistanceMetric::kL2;
  uint32_t total_dim_ = 0;
};

}  // namespace saq
