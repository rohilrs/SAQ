/// @file caq_code_adjustment.cpp
/// @brief Implementation of CAQ code adjustment.

#include "saq/caq_code_adjustment.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

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

/// @brief Compute squared L2 norm of a vector.
float SquaredNorm(const float* v, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    sum += v[i] * v[i];
  }
  return sum;
}

/// @brief Add vector b to vector a: a += b.
void VectorAdd(float* a, const float* b, uint32_t dim) {
  for (uint32_t i = 0; i < dim; ++i) {
    a[i] += b[i];
  }
}

/// @brief Subtract vector b from vector a: a -= b.
void VectorSub(float* a, const float* b, uint32_t dim) {
  for (uint32_t i = 0; i < dim; ++i) {
    a[i] -= b[i];
  }
}

}  // namespace

bool CAQAdjuster::Initialize(const std::vector<Codebook>& codebooks,
                              const std::vector<Segment>& segments) {
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
    // Validate data size
    size_t expected_size = static_cast<size_t>(codebooks[i].centroids) *
                           static_cast<size_t>(codebooks[i].dim_count);
    if (codebooks[i].data.size() != expected_size) {
      return false;
    }
  }

  codebooks_ = codebooks;
  segments_ = segments;

  // Compute total dimensionality
  total_dim_ = 0;
  for (const auto& seg : segments_) {
    total_dim_ += seg.dim_count;
  }

  return true;
}

EncodedVector CAQAdjuster::EncodeGreedy(const float* vector, uint32_t dim) const {
  EncodedVector result;

  if (!IsInitialized() || dim != total_dim_) {
    result.distortion = std::numeric_limits<float>::max();
    return result;
  }

  result.codes.resize(segments_.size());
  result.distortion = 0.0f;

  // For each segment, find the nearest centroid
  for (size_t seg_idx = 0; seg_idx < segments_.size(); ++seg_idx) {
    const Segment& seg = segments_[seg_idx];
    const Codebook& cb = codebooks_[seg_idx];

    const float* seg_vector = vector + seg.start_dim;
    uint32_t best_code = 0;
    float best_dist = std::numeric_limits<float>::max();

    // Linear search through centroids
    for (uint32_t c = 0; c < cb.centroids; ++c) {
      const float* centroid = cb.data.data() + static_cast<size_t>(c) * cb.dim_count;
      float dist = SquaredL2(seg_vector, centroid, cb.dim_count);

      if (dist < best_dist) {
        best_dist = dist;
        best_code = c;
      }
    }

    result.codes[seg_idx] = best_code;
    result.distortion += best_dist;
  }

  return result;
}

bool CAQAdjuster::Reconstruct(const std::vector<uint32_t>& codes,
                               float* output) const {
  if (!IsInitialized()) {
    return false;
  }

  if (codes.size() != segments_.size()) {
    return false;
  }

  // Validate codes and reconstruct
  for (size_t seg_idx = 0; seg_idx < segments_.size(); ++seg_idx) {
    const Segment& seg = segments_[seg_idx];
    const Codebook& cb = codebooks_[seg_idx];
    uint32_t code = codes[seg_idx];

    if (code >= cb.centroids) {
      return false;
    }

    const float* centroid = cb.data.data() + static_cast<size_t>(code) * cb.dim_count;
    std::memcpy(output + seg.start_dim, centroid, cb.dim_count * sizeof(float));
  }

  return true;
}

float CAQAdjuster::ComputeDistortion(const float* vector, uint32_t dim,
                                      const std::vector<uint32_t>& codes) const {
  if (!IsInitialized() || dim != total_dim_) {
    return std::numeric_limits<float>::max();
  }

  if (codes.size() != segments_.size()) {
    return std::numeric_limits<float>::max();
  }

  float total_dist = 0.0f;

  for (size_t seg_idx = 0; seg_idx < segments_.size(); ++seg_idx) {
    const Segment& seg = segments_[seg_idx];
    const Codebook& cb = codebooks_[seg_idx];
    uint32_t code = codes[seg_idx];

    if (code >= cb.centroids) {
      return std::numeric_limits<float>::max();
    }

    const float* seg_vector = vector + seg.start_dim;
    const float* centroid = cb.data.data() + static_cast<size_t>(code) * cb.dim_count;
    total_dist += SquaredL2(seg_vector, centroid, cb.dim_count);
  }

  return total_dist;
}

uint32_t CAQAdjuster::FindBestCode(const float* vector, uint32_t segment_idx,
                                    std::vector<uint32_t>& current_codes,
                                    std::vector<float>& residual) const {
  const Segment& seg = segments_[segment_idx];
  const Codebook& cb = codebooks_[segment_idx];

  // Get the current centroid contribution for this segment
  uint32_t old_code = current_codes[segment_idx];
  const float* old_centroid = cb.data.data() +
                               static_cast<size_t>(old_code) * cb.dim_count;

  // Compute target: what this segment should approximate
  // target = original[seg] = residual[seg] + old_centroid
  std::vector<float> target(seg.dim_count);
  for (uint32_t d = 0; d < seg.dim_count; ++d) {
    target[d] = residual[seg.start_dim + d] + old_centroid[d];
  }

  // Find the centroid that minimizes ||target - centroid||²
  uint32_t best_code = old_code;
  float best_dist = SquaredL2(target.data(), old_centroid, seg.dim_count);

  for (uint32_t c = 0; c < cb.centroids; ++c) {
    const float* centroid = cb.data.data() + static_cast<size_t>(c) * cb.dim_count;
    float dist = SquaredL2(target.data(), centroid, seg.dim_count);

    if (dist < best_dist) {
      best_dist = dist;
      best_code = c;
    }
  }

  // Only update residual if code changed
  if (best_code != old_code) {
    const float* new_centroid = cb.data.data() +
                                 static_cast<size_t>(best_code) * cb.dim_count;
    for (uint32_t d = 0; d < seg.dim_count; ++d) {
      // residual = original - centroid
      // new_residual = original - new_centroid = (residual + old_centroid) - new_centroid
      residual[seg.start_dim + d] = target[d] - new_centroid[d];
    }
  }

  return best_code;
}

EncodedVector CAQAdjuster::RefineCAQ(const float* vector, uint32_t dim,
                                      const EncodedVector& initial,
                                      const CAQConfig& config) const {
  EncodedVector result;

  if (!IsInitialized() || dim != total_dim_) {
    result.distortion = std::numeric_limits<float>::max();
    return result;
  }

  if (initial.codes.size() != segments_.size()) {
    result.distortion = std::numeric_limits<float>::max();
    return result;
  }

  // Start with initial codes
  result.codes = initial.codes;

  // Compute initial residual: original - reconstruction
  std::vector<float> residual(total_dim_);
  std::memcpy(residual.data(), vector, total_dim_ * sizeof(float));

  // Subtract all current centroids
  for (size_t seg_idx = 0; seg_idx < segments_.size(); ++seg_idx) {
    const Segment& seg = segments_[seg_idx];
    const Codebook& cb = codebooks_[seg_idx];
    uint32_t code = result.codes[seg_idx];
    const float* centroid = cb.data.data() + static_cast<size_t>(code) * cb.dim_count;

    for (uint32_t d = 0; d < seg.dim_count; ++d) {
      residual[seg.start_dim + d] -= centroid[d];
    }
  }

  float prev_distortion = SquaredNorm(residual.data(), total_dim_);

  // Iterative refinement
  for (uint32_t iter = 0; iter < config.max_iterations; ++iter) {
    bool any_changed = false;

    // Cycle through segments and optimize each
    for (size_t seg_idx = 0; seg_idx < segments_.size(); ++seg_idx) {
      uint32_t old_code = result.codes[seg_idx];
      uint32_t new_code = FindBestCode(vector, static_cast<uint32_t>(seg_idx),
                                        result.codes, residual);

      if (new_code != old_code) {
        result.codes[seg_idx] = new_code;
        any_changed = true;
      }
    }

    if (!any_changed) {
      break;  // Converged - no codes changed
    }

    // Check convergence by distortion
    float current_distortion = SquaredNorm(residual.data(), total_dim_);
    float improvement = (prev_distortion - current_distortion) / 
                        (prev_distortion + 1e-10f);

    if (improvement < config.convergence_threshold) {
      break;  // Converged - small improvement
    }

    prev_distortion = current_distortion;
  }

  // Compute final distortion using the same method as EncodeGreedy
  result.distortion = ComputeDistortion(vector, dim, result.codes);

  // Ensure we never return worse than initial (numerical safety)
  if (result.distortion > initial.distortion) {
    return initial;
  }

  return result;
}

std::vector<EncodedVector> CAQAdjuster::EncodeBatch(
    const float* vectors, uint32_t n_vectors, uint32_t dim,
    const CAQConfig& config, CAQStats* stats) const {
  std::vector<EncodedVector> results;
  results.reserve(n_vectors);

  if (!IsInitialized() || dim != total_dim_) {
    return results;
  }

  float initial_total = 0.0f;
  float final_total = 0.0f;
  uint64_t codes_changed = 0;

  for (uint32_t i = 0; i < n_vectors; ++i) {
    const float* vec = vectors + static_cast<size_t>(i) * dim;

    // Greedy initial encoding
    EncodedVector initial = EncodeGreedy(vec, dim);
    initial_total += initial.distortion;

    // Refine with CAQ
    EncodedVector refined = RefineCAQ(vec, dim, initial, config);
    final_total += refined.distortion;

    // Count changed codes
    for (size_t s = 0; s < initial.codes.size(); ++s) {
      if (initial.codes[s] != refined.codes[s]) {
        codes_changed++;
      }
    }

    results.push_back(std::move(refined));
  }

  if (stats != nullptr) {
    stats->iterations = config.max_iterations;  // Conservative estimate
    stats->initial_distortion = initial_total;
    stats->final_distortion = final_total;
    stats->codes_changed = codes_changed;
    stats->converged = (final_total < initial_total * 0.999f) || 
                       (codes_changed == 0);
  }

  return results;
}

}  // namespace saq
