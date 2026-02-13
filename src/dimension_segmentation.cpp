/// @file dimension_segmentation.cpp
/// @brief Implementation of DimensionSegmenter class.

#include "saq/dimension_segmentation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#ifdef SAQ_USE_OPENMP
#include <omp.h>
#endif

namespace saq {

bool DimensionSegmenter::ComputeStats(const float* data, uint32_t n_samples,
                                      uint32_t dim) {
  if (data == nullptr || n_samples < 2 || dim == 0) {
    return false;
  }

  dim_ = dim;
  stats_.resize(dim);

  // Initialize stats
  for (uint32_t j = 0; j < dim; ++j) {
    stats_[j].dim_index = j;
    stats_[j].mean = 0.0f;
    stats_[j].variance = 0.0f;
    stats_[j].min_val = std::numeric_limits<float>::max();
    stats_[j].max_val = std::numeric_limits<float>::lowest();
  }

  // First pass: compute mean, min, max
  // Parallelize over dimensions with thread-local accumulators
#ifdef SAQ_USE_OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (uint32_t j = 0; j < dim; ++j) {
    double sum = 0.0;
    float local_min = std::numeric_limits<float>::max();
    float local_max = std::numeric_limits<float>::lowest();
    for (uint32_t i = 0; i < n_samples; ++i) {
      float val = data[i * dim + j];
      sum += static_cast<double>(val);
      local_min = std::min(local_min, val);
      local_max = std::max(local_max, val);
    }
    stats_[j].mean = static_cast<float>(sum);
    stats_[j].min_val = local_min;
    stats_[j].max_val = local_max;
  }

  const float inv_n = 1.0f / static_cast<float>(n_samples);
  for (uint32_t j = 0; j < dim; ++j) {
    stats_[j].mean *= inv_n;
  }

  // Second pass: compute variance
#ifdef SAQ_USE_OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (uint32_t j = 0; j < dim; ++j) {
    double var_sum = 0.0;
    float mean_j = stats_[j].mean;
    for (uint32_t i = 0; i < n_samples; ++i) {
      float diff = data[i * dim + j] - mean_j;
      var_sum += static_cast<double>(diff) * static_cast<double>(diff);
    }
    stats_[j].variance = static_cast<float>(var_sum);
  }

  const float inv_n_minus_1 = 1.0f / static_cast<float>(n_samples - 1);
  for (uint32_t j = 0; j < dim; ++j) {
    stats_[j].variance *= inv_n_minus_1;
  }

  return true;
}

SegmentationResult DimensionSegmenter::Segment(
    const SegmentationConfig& config) const {
  SegmentationResult result;

  if (stats_.empty()) {
    result.error = "No statistics computed. Call ComputeStats first.";
    return result;
  }

  switch (config.strategy) {
    case SegmentationStrategy::kUniform:
      return SegmentUniform(config.num_segments, config.min_dims_per_segment);

    case SegmentationStrategy::kVarianceBased:
      return SegmentByVariance(config.num_segments,
                               config.min_dims_per_segment);

    case SegmentationStrategy::kFixed:
      return SegmentFixed(config.fixed_boundaries);

    default:
      result.error = "Unknown segmentation strategy.";
      return result;
  }
}

SegmentationResult DimensionSegmenter::SegmentUniform(uint32_t num_segments,
                                                      uint32_t min_dims) const {
  SegmentationResult result;
  result.dim_stats = stats_;

  if (num_segments == 0) {
    result.error = "num_segments must be > 0";
    return result;
  }

  // Clamp num_segments to valid range
  uint32_t max_segments = dim_ / std::max(min_dims, 1u);
  if (max_segments == 0) max_segments = 1;
  num_segments = std::min(num_segments, max_segments);

  // Compute base size and remainder
  uint32_t base_size = dim_ / num_segments;
  uint32_t remainder = dim_ % num_segments;

  result.segments.reserve(num_segments);
  result.segment_variances.reserve(num_segments);

  uint32_t start = 0;
  for (uint32_t seg_id = 0; seg_id < num_segments; ++seg_id) {
    // Distribute remainder across first segments
    uint32_t seg_size = base_size + (seg_id < remainder ? 1 : 0);

    ::saq::Segment seg;
    seg.id = seg_id;
    seg.start_dim = start;
    seg.dim_count = seg_size;
    seg.bits = 0;  // To be filled by bit allocation

    // Compute total variance for this segment
    float total_var = 0.0f;
    for (uint32_t d = start; d < start + seg_size; ++d) {
      total_var += stats_[d].variance;
    }

    result.segments.push_back(seg);
    result.segment_variances.push_back(total_var);

    start += seg_size;
  }

  return result;
}

SegmentationResult DimensionSegmenter::SegmentByVariance(
    uint32_t num_segments, uint32_t min_dims) const {
  SegmentationResult result;
  result.dim_stats = stats_;

  if (num_segments == 0) {
    result.error = "num_segments must be > 0";
    return result;
  }

  // Clamp num_segments
  uint32_t max_segments = dim_ / std::max(min_dims, 1u);
  if (max_segments == 0) max_segments = 1;
  num_segments = std::min(num_segments, max_segments);

  // Compute cumulative variance (dimensions are kept in original order)
  std::vector<double> cum_var(dim_ + 1, 0.0);
  for (uint32_t j = 0; j < dim_; ++j) {
    cum_var[j + 1] = cum_var[j] + static_cast<double>(stats_[j].variance);
  }
  double total_variance = cum_var[dim_];

  // Target variance per segment
  double target_var = total_variance / static_cast<double>(num_segments);

  result.segments.reserve(num_segments);
  result.segment_variances.reserve(num_segments);

  uint32_t start = 0;
  uint32_t seg_id = 0;

  for (uint32_t seg_idx = 0; seg_idx < num_segments - 1; ++seg_idx) {
    // Find dimension where cumulative variance reaches the target
    double target_cum = cum_var[start] + target_var;

    // Binary search for the split point
    uint32_t end = start + min_dims;
    while (end < dim_ && cum_var[end] < target_cum) {
      ++end;
    }

    // Ensure we leave enough dimensions for remaining segments
    uint32_t remaining_segments = num_segments - seg_idx - 1;
    uint32_t max_end = dim_ - remaining_segments * min_dims;
    end = std::min(end, max_end);
    end = std::max(end, start + min_dims);

    uint32_t seg_size = end - start;

    ::saq::Segment seg;
    seg.id = seg_id++;
    seg.start_dim = start;
    seg.dim_count = seg_size;
    seg.bits = 0;

    float seg_var = static_cast<float>(cum_var[end] - cum_var[start]);
    result.segments.push_back(seg);
    result.segment_variances.push_back(seg_var);

    start = end;
  }

  // Last segment gets all remaining dimensions
  if (start < dim_) {
    ::saq::Segment seg;
    seg.id = seg_id;
    seg.start_dim = start;
    seg.dim_count = dim_ - start;
    seg.bits = 0;

    float seg_var = static_cast<float>(cum_var[dim_] - cum_var[start]);
    result.segments.push_back(seg);
    result.segment_variances.push_back(seg_var);
  }

  return result;
}

SegmentationResult DimensionSegmenter::SegmentFixed(
    const std::vector<uint32_t>& boundaries) const {
  SegmentationResult result;
  result.dim_stats = stats_;

  if (boundaries.empty()) {
    result.error = "Fixed boundaries cannot be empty.";
    return result;
  }

  // Validate boundaries are sorted and within range
  uint32_t prev = 0;
  for (size_t i = 0; i < boundaries.size(); ++i) {
    if (boundaries[i] <= prev) {
      result.error = "Boundaries must be strictly increasing.";
      return result;
    }
    prev = boundaries[i];
  }

  if (boundaries.back() != dim_) {
    result.error = "Last boundary must equal dimension count.";
    return result;
  }

  result.segments.reserve(boundaries.size());
  result.segment_variances.reserve(boundaries.size());

  uint32_t start = 0;
  for (size_t seg_id = 0; seg_id < boundaries.size(); ++seg_id) {
    uint32_t end = boundaries[seg_id];
    uint32_t seg_size = end - start;

    ::saq::Segment seg;
    seg.id = static_cast<uint32_t>(seg_id);
    seg.start_dim = start;
    seg.dim_count = seg_size;
    seg.bits = 0;

    // Compute total variance
    float total_var = 0.0f;
    for (uint32_t d = start; d < end; ++d) {
      total_var += stats_[d].variance;
    }

    result.segments.push_back(seg);
    result.segment_variances.push_back(total_var);

    start = end;
  }

  return result;
}

}  // namespace saq
