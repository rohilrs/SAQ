/// @file distance_estimator.cpp
/// @brief Implementation of asymmetric distance estimation.

#include "saq/distance_estimator.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace saq {

namespace {

/// @brief Compute squared L2 distance between two vectors.
float SquaredL2(const float* a, const float* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

/// @brief Compute inner product between two vectors.
float InnerProduct(const float* a, const float* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

}  // namespace

bool DistanceEstimator::Initialize(const std::vector<Codebook>& codebooks,
                                    const std::vector<Segment>& segments,
                                    DistanceMetric metric) {
  if (codebooks.empty() || segments.empty()) {
    return false;
  }

  if (codebooks.size() != segments.size()) {
    return false;
  }

  // Validate codebook-segment correspondence
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

  // Compute total dimensionality
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

  if (!IsInitialized() || dim != total_dim_) {
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

    // Compute distance from query segment to each centroid
    for (uint32_t c = 0; c < cb.centroids; ++c) {
      const float* centroid = cb.data.data() +
                               static_cast<size_t>(c) * cb.dim_count;

      if (metric_ == DistanceMetric::kL2) {
        seg_table.distances[c] = SquaredL2(query_seg, centroid, cb.dim_count);
      } else {
        // For inner product, we want max IP, so store negative
        seg_table.distances[c] = -InnerProduct(query_seg, centroid, cb.dim_count);
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

  for (uint32_t v = 0; v < n_vectors; ++v) {
    const uint32_t* vec_codes = codes + static_cast<size_t>(v) * num_segments;
    float total = 0.0f;

    for (size_t s = 0; s < num_segments; ++s) {
      total += table.segment_tables[s].distances[vec_codes[s]];
    }

    distances[v] = total;
  }
}

bool DistanceEstimator::Reconstruct(const uint32_t* codes, float* output) const {
  if (!IsInitialized()) {
    return false;
  }

  for (size_t seg_idx = 0; seg_idx < segments_.size(); ++seg_idx) {
    const Segment& seg = segments_[seg_idx];
    const Codebook& cb = codebooks_[seg_idx];
    uint32_t code = codes[seg_idx];

    if (code >= cb.centroids) {
      return false;
    }

    const float* centroid = cb.data.data() +
                             static_cast<size_t>(code) * cb.dim_count;
    std::memcpy(output + seg.start_dim, centroid, cb.dim_count * sizeof(float));
  }

  return true;
}

float DistanceEstimator::ComputeExactDistance(const float* query,
                                               const uint32_t* codes,
                                               uint32_t dim) const {
  if (!IsInitialized() || dim != total_dim_) {
    return std::numeric_limits<float>::max();
  }

  // Reconstruct the encoded vector
  std::vector<float> reconstructed(total_dim_);
  if (!Reconstruct(codes, reconstructed.data())) {
    return std::numeric_limits<float>::max();
  }

  // Compute distance
  if (metric_ == DistanceMetric::kL2) {
    return SquaredL2(query, reconstructed.data(), total_dim_);
  } else {
    return -InnerProduct(query, reconstructed.data(), total_dim_);
  }
}

}  // namespace saq
