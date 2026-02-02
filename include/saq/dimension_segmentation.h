#pragma once

/// @file dimension_segmentation.h
/// @brief Dimension partitioning for non-uniform bit allocation in SAQ.
///
/// Partitions vector dimensions into contiguous segments based on variance,
/// enabling the bit allocation algorithm to assign more bits to high-variance
/// dimensions and fewer to low-variance ones.

#include "saq/quantization_plan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace saq {

/// @brief Strategy for partitioning dimensions into segments.
enum class SegmentationStrategy : uint8_t {
  kUniform = 0,       ///< Equal-sized segments.
  kVarianceBased = 1, ///< Segments grouped by similar variance.
  kFixed = 2          ///< User-provided fixed segment boundaries.
};

/// @brief Configuration for dimension segmentation.
struct SegmentationConfig {
  SegmentationStrategy strategy = SegmentationStrategy::kVarianceBased;

  /// Number of segments to create (for kUniform and kVarianceBased).
  uint32_t num_segments = 4;

  /// Minimum dimensions per segment (prevents degenerate segments).
  uint32_t min_dims_per_segment = 1;

  /// Fixed segment boundaries for kFixed strategy.
  /// Each value is the exclusive end dimension of a segment.
  /// E.g., {32, 64, 128} creates segments [0,32), [32,64), [64,128).
  std::vector<uint32_t> fixed_boundaries;
};

/// @brief Per-dimension statistics computed from training data.
struct DimensionStats {
  uint32_t dim_index = 0;  ///< Original dimension index.
  float mean = 0.0f;       ///< Mean value across samples.
  float variance = 0.0f;   ///< Variance across samples.
  float min_val = 0.0f;    ///< Minimum observed value.
  float max_val = 0.0f;    ///< Maximum observed value.
};

/// @brief Result of dimension segmentation.
struct SegmentationResult {
  std::vector<Segment> segments;         ///< Output segments.
  std::vector<DimensionStats> dim_stats; ///< Per-dimension statistics.
  std::vector<float> segment_variances;  ///< Total variance per segment.
  std::string error;                     ///< Error message if failed.

  bool IsValid() const { return error.empty() && !segments.empty(); }
};

/// @brief Partitions dimensions into segments for bit allocation.
///
/// Computes per-dimension variance from training data and groups dimensions
/// into contiguous segments. Dimensions are optionally reordered by variance
/// before segmentation to maximize bit allocation efficiency.
class DimensionSegmenter {
 public:
  DimensionSegmenter() = default;
  ~DimensionSegmenter() = default;

  // Non-copyable, movable
  DimensionSegmenter(const DimensionSegmenter&) = delete;
  DimensionSegmenter& operator=(const DimensionSegmenter&) = delete;
  DimensionSegmenter(DimensionSegmenter&&) = default;
  DimensionSegmenter& operator=(DimensionSegmenter&&) = default;

  /// @brief Compute dimension statistics from training data.
  /// @param data Row-major matrix of shape (n_samples, dim).
  /// @param n_samples Number of training vectors.
  /// @param dim Vector dimensionality.
  /// @return True on success.
  bool ComputeStats(const float* data, uint32_t n_samples, uint32_t dim);

  /// @brief Segment dimensions using the configured strategy.
  /// @param config Segmentation configuration.
  /// @return Segmentation result (check IsValid()).
  SegmentationResult Segment(const SegmentationConfig& config) const;

  /// @brief Get computed dimension statistics.
  /// @return Vector of per-dimension stats, empty if not computed.
  const std::vector<DimensionStats>& GetStats() const { return stats_; }

  /// @brief Get dimensionality.
  uint32_t Dim() const { return dim_; }

  /// @brief Check if statistics have been computed.
  bool HasStats() const { return !stats_.empty(); }

 private:
  uint32_t dim_ = 0;
  std::vector<DimensionStats> stats_;

  /// Create uniform segments of equal size.
  SegmentationResult SegmentUniform(uint32_t num_segments,
                                    uint32_t min_dims) const;

  /// Create segments grouping similar-variance dimensions.
  SegmentationResult SegmentByVariance(uint32_t num_segments,
                                       uint32_t min_dims) const;

  /// Create segments from fixed boundaries.
  SegmentationResult SegmentFixed(
      const std::vector<uint32_t>& boundaries) const;
};

}  // namespace saq
