/// @file distance_estimator.cpp
/// @brief Implementation of asymmetric distance estimation.
///
/// Scalar estimation uses the SAQ paper formula:
///   <o_bar, q> = delta * <codes, q> + q_sum * (-v_max + delta/2)
/// Legacy codebook-based ADC is preserved for backward compatibility.

#include "saq/distance_estimator.h"
#include "saq/simd_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#ifdef SAQ_USE_OPENMP
#include <omp.h>
#endif

namespace saq {

// ---------------------------------------------------------------------------
// Scalar quantization API
// ---------------------------------------------------------------------------

bool DistanceEstimator::InitializeScalar(const std::vector<Segment>& segments,
                                         DistanceMetric metric) {
  if (segments.empty()) {
    return false;
  }

  segments_ = segments;
  metric_ = metric;
  codebooks_.clear();

  total_dim_ = 0;
  for (const auto& seg : segments_) {
    total_dim_ += seg.dim_count;
  }

  return true;
}

ScalarQueryTable DistanceEstimator::PrecomputeScalarQuery(
    const float* rotated_query) const {
  ScalarQueryTable table;

  if (!IsInitialized() || rotated_query == nullptr) {
    return table;
  }

  table.segment_q_sums.resize(segments_.size());
  table.segment_scales.resize(segments_.size());
  table.rotated_query.assign(rotated_query, rotated_query + total_dim_);

  for (size_t s = 0; s < segments_.size(); ++s) {
    const auto& seg = segments_[s];

    // Compute q_sum for this segment
    float q_sum = 0.0f;
    for (uint32_t d = 0; d < seg.dim_count; ++d) {
      q_sum += rotated_query[seg.start_dim + d];
    }
    table.segment_q_sums[s] = q_sum;

    // Scale factor: 2 / 2^B (multiply by v_max at estimation time)
    if (seg.bits > 0) {
      table.segment_scales[s] =
          2.0f / static_cast<float>(1u << seg.bits);
    } else {
      table.segment_scales[s] = 0.0f;
    }
  }

  return table;
}

float DistanceEstimator::EstimateScalarIP(const ScalarQueryTable& table,
                                          const uint8_t* codes,
                                          float v_max) const {
  float total_ip = 0.0f;

  for (size_t s = 0; s < segments_.size(); ++s) {
    const auto& seg = segments_[s];

    if (seg.bits == 0) {
      continue;
    }

    // delta = 2 * v_max / 2^B = v_max * segment_scales[s]
    float delta = v_max * table.segment_scales[s];

    // Compute code_dot_q = sum(codes[i] * q[i]) for this segment
    float code_dot_q = 0.0f;
    for (uint32_t d = 0; d < seg.dim_count; ++d) {
      uint32_t idx = seg.start_dim + d;
      code_dot_q +=
          static_cast<float>(codes[idx]) * table.rotated_query[idx];
    }

    total_ip += delta * code_dot_q +
                table.segment_q_sums[s] * (-v_max + delta * 0.5f);
  }

  return total_ip;
}

void DistanceEstimator::EstimateScalarIPBatch(
    const ScalarQueryTable& table, const uint8_t* const* codes_batch,
    const float* v_maxs, uint32_t n_vectors, float* ips) const {
#ifdef SAQ_USE_OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (uint32_t v = 0; v < n_vectors; ++v) {
    ips[v] = EstimateScalarIP(table, codes_batch[v], v_maxs[v]);
  }
}

// ---------------------------------------------------------------------------
// Legacy codebook-based API
// ---------------------------------------------------------------------------

bool DistanceEstimator::Initialize(const std::vector<Codebook>& codebooks,
                                   const std::vector<Segment>& segments,
                                   DistanceMetric metric) {
  if (codebooks.empty() || segments.empty()) {
    return false;
  }

  if (codebooks.size() != segments.size()) {
    return false;
  }

  for (size_t i = 0; i < codebooks.size(); ++i) {
    if (codebooks[i].segment_id != segments[i].id) {
      return false;
    }
    if (codebooks[i].dim_count != segments[i].dim_count) {
      return false;
    }
    if (codebooks[i].centroids == 0) {
      return false;
    }
    size_t expected_size = static_cast<size_t>(codebooks[i].centroids) *
                           static_cast<size_t>(codebooks[i].dim_count);
    if (codebooks[i].data.size() != expected_size) {
      return false;
    }
  }

  codebooks_ = codebooks;
  segments_ = segments;
  metric_ = metric;

  total_dim_ = 0;
  for (const auto& seg : segments_) {
    total_dim_ += seg.dim_count;
  }

  return true;
}

DistanceTable DistanceEstimator::ComputeDistanceTable(const float* query,
                                                      uint32_t dim) const {
  DistanceTable table;
  table.metric = metric_;

  if (!IsInitialized() || dim != total_dim_ || codebooks_.empty()) {
    return table;
  }

  table.segment_tables.resize(segments_.size());

  for (size_t seg_idx = 0; seg_idx < segments_.size(); ++seg_idx) {
    const Segment& seg = segments_[seg_idx];
    const Codebook& cb = codebooks_[seg_idx];

    SegmentDistanceTable& seg_table = table.segment_tables[seg_idx];
    seg_table.segment_id = seg.id;
    seg_table.num_centroids = cb.centroids;
    seg_table.distances.resize(cb.centroids);

    const float* query_seg = query + seg.start_dim;

    if (metric_ == DistanceMetric::kL2) {
      simd::L2DistancesBatch(query_seg, cb.data.data(), cb.centroids,
                             cb.dim_count, seg_table.distances.data());
    } else {
      simd::InnerProductsBatch(query_seg, cb.data.data(), cb.centroids,
                               cb.dim_count, seg_table.distances.data());
      for (uint32_t c = 0; c < cb.centroids; ++c) {
        seg_table.distances[c] = -seg_table.distances[c];
      }
    }
  }

  return table;
}

float DistanceEstimator::EstimateDistance(const DistanceTable& table,
                                         const uint32_t* codes) const {
  if (table.segment_tables.size() != segments_.size()) {
    return std::numeric_limits<float>::max();
  }

  float total = 0.0f;
  for (size_t s = 0; s < segments_.size(); ++s) {
    uint32_t code = codes[s];
    if (code >= table.segment_tables[s].num_centroids) {
      return std::numeric_limits<float>::max();
    }
    total += table.segment_tables[s].distances[code];
  }

  return total;
}

void DistanceEstimator::EstimateDistancesBatch(const DistanceTable& table,
                                               const uint32_t* codes,
                                               uint32_t n_vectors,
                                               float* distances) const {
  const size_t num_segments = segments_.size();

#ifdef SAQ_USE_OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (uint32_t v = 0; v < n_vectors; ++v) {
    const uint32_t* vec_codes = codes + static_cast<size_t>(v) * num_segments;
    float total = 0.0f;

    for (size_t s = 0; s < num_segments; ++s) {
      total += table.segment_tables[s].distances[vec_codes[s]];
    }

    distances[v] = total;
  }
}

bool DistanceEstimator::Reconstruct(const uint32_t* codes,
                                    float* output) const {
  if (!IsInitialized() || codebooks_.empty()) {
    return false;
  }

  for (size_t seg_idx = 0; seg_idx < segments_.size(); ++seg_idx) {
    const Segment& seg = segments_[seg_idx];
    const Codebook& cb = codebooks_[seg_idx];
    uint32_t code = codes[seg_idx];

    if (code >= cb.centroids) {
      return false;
    }

    const float* centroid =
        cb.data.data() + static_cast<size_t>(code) * cb.dim_count;
    std::memcpy(output + seg.start_dim, centroid,
                cb.dim_count * sizeof(float));
  }

  return true;
}

float DistanceEstimator::ComputeExactDistance(const float* query,
                                             const uint32_t* codes,
                                             uint32_t dim) const {
  if (!IsInitialized() || dim != total_dim_ || codebooks_.empty()) {
    return std::numeric_limits<float>::max();
  }

  std::vector<float> reconstructed(total_dim_);
  if (!Reconstruct(codes, reconstructed.data())) {
    return std::numeric_limits<float>::max();
  }

  if (metric_ == DistanceMetric::kL2) {
    return simd::L2DistanceSquared(query, reconstructed.data(), total_dim_);
  } else {
    return -simd::InnerProduct(query, reconstructed.data(), total_dim_);
  }
}

}  // namespace saq
