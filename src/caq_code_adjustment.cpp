/// @file caq_code_adjustment.cpp
/// @brief Implementation of CAQ per-dimension code adjustment (Algorithm 1).
///
/// Refines scalar codes by trying +/-1 adjustments per dimension,
/// accepting changes that improve cosine similarity.  Uses incremental
/// dot-product and norm updates for O(R * D) per-vector complexity.

#include "saq/caq_code_adjustment.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#ifdef SAQ_USE_OPENMP
#include <omp.h>
#endif

namespace saq {

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

bool CAQAdjuster::Initialize(const std::vector<Segment>& segments) {
  if (segments.empty()) {
    return false;
  }

  segments_ = segments;

  total_dim_ = 0;
  for (const auto& seg : segments_) {
    total_dim_ += seg.dim_count;
  }

  return true;
}

bool CAQAdjuster::Initialize(const std::vector<Codebook>& /*codebooks*/,
                              const std::vector<Segment>& segments) {
  return Initialize(segments);
}

// ---------------------------------------------------------------------------
// Scalar reconstruction
// ---------------------------------------------------------------------------

void CAQAdjuster::ReconstructScalar(const uint8_t* codes, float v_max,
                                    float* output) const {
  for (const auto& seg : segments_) {
    if (seg.bits == 0) {
      for (uint32_t d = 0; d < seg.dim_count; ++d) {
        output[seg.start_dim + d] = 0.0f;
      }
      continue;
    }

    uint32_t levels = 1u << seg.bits;
    float delta = (2.0f * v_max) / static_cast<float>(levels);

    for (uint32_t d = 0; d < seg.dim_count; ++d) {
      uint32_t idx = seg.start_dim + d;
      output[idx] = delta * (static_cast<float>(codes[idx]) + 0.5f) - v_max;
    }
  }
}

// ---------------------------------------------------------------------------
// Per-dimension CAQ refinement (Algorithm 1)
// ---------------------------------------------------------------------------

uint64_t CAQAdjuster::RefineScalar(const float* original, uint8_t* codes,
                                   float v_max,
                                   const CAQConfig& config) const {
  if (!IsInitialized() || original == nullptr || codes == nullptr) {
    return 0;
  }

  if (v_max < 1e-30f) {
    return 0;  // Zero vector, nothing to adjust
  }

  // Compute initial reconstruction and running statistics
  // x_dot_o = <original, o_bar>
  // o_norm_sq = ||o_bar||^2
  // x_norm_sq = ||original||^2  (constant throughout)
  float x_norm_sq = 0.0f;
  float x_dot_o = 0.0f;
  float o_norm_sq = 0.0f;

  for (const auto& seg : segments_) {
    if (seg.bits == 0) {
      // Zero-bit dimensions contribute to x_norm_sq but not to reconstruction
      for (uint32_t d = 0; d < seg.dim_count; ++d) {
        uint32_t idx = seg.start_dim + d;
        x_norm_sq += original[idx] * original[idx];
      }
      continue;
    }

    uint32_t levels = 1u << seg.bits;
    float delta = (2.0f * v_max) / static_cast<float>(levels);

    for (uint32_t d = 0; d < seg.dim_count; ++d) {
      uint32_t idx = seg.start_dim + d;
      float x_i = original[idx];
      float o_i = delta * (static_cast<float>(codes[idx]) + 0.5f) - v_max;

      x_norm_sq += x_i * x_i;
      x_dot_o += x_i * o_i;
      o_norm_sq += o_i * o_i;
    }
  }

  float x_norm = std::sqrt(x_norm_sq);
  if (x_norm < 1e-30f) {
    return 0;  // Zero original vector
  }

  uint64_t total_changes = 0;

  for (uint32_t round = 0; round < config.num_rounds; ++round) {
    uint64_t round_changes = 0;

    for (const auto& seg : segments_) {
      if (seg.bits == 0) {
        continue;
      }

      uint32_t levels = 1u << seg.bits;
      float delta = (2.0f * v_max) / static_cast<float>(levels);
      auto max_code = static_cast<uint8_t>(levels - 1);

      for (uint32_t d = 0; d < seg.dim_count; ++d) {
        uint32_t idx = seg.start_dim + d;
        uint8_t cur_code = codes[idx];
        float x_i = original[idx];

        // Current reconstruction for this dimension
        float cur_o_i = delta * (static_cast<float>(cur_code) + 0.5f) - v_max;

        // Current cosine = x_dot_o / (x_norm * sqrt(o_norm_sq))
        // Since x_norm is constant, maximizing cosine is equivalent to
        // maximizing x_dot_o / sqrt(o_norm_sq).
        // We compare (x_dot_o_new)^2 * o_norm_sq_old vs
        //            (x_dot_o_old)^2 * o_norm_sq_new
        // to avoid sqrt.  But for simplicity and correctness,
        // use the full cosine comparison.

        float best_cosine = (o_norm_sq > 0.0f)
            ? x_dot_o / (x_norm * std::sqrt(o_norm_sq))
            : -1.0f;
        uint8_t best_code = cur_code;
        float best_x_dot_o = x_dot_o;
        float best_o_norm_sq = o_norm_sq;

        // Try c[i] + 1
        if (cur_code < max_code) {
          float new_o_i = cur_o_i + delta;
          float new_x_dot_o = x_dot_o + delta * x_i;
          float new_o_norm_sq =
              o_norm_sq + 2.0f * delta * cur_o_i + delta * delta;

          if (new_o_norm_sq > 0.0f) {
            float new_cosine =
                new_x_dot_o / (x_norm * std::sqrt(new_o_norm_sq));
            if (new_cosine > best_cosine) {
              best_cosine = new_cosine;
              best_code = cur_code + 1;
              best_x_dot_o = new_x_dot_o;
              best_o_norm_sq = new_o_norm_sq;
            }
          }
        }

        // Try c[i] - 1 (always from original state, not from +1)
        if (cur_code > 0) {
          float new_o_i = cur_o_i - delta;
          float new_x_dot_o = x_dot_o - delta * x_i;
          float new_o_norm_sq =
              o_norm_sq - 2.0f * delta * cur_o_i + delta * delta;

          if (new_o_norm_sq > 0.0f) {
            float new_cosine =
                new_x_dot_o / (x_norm * std::sqrt(new_o_norm_sq));
            if (new_cosine > best_cosine) {
              best_cosine = new_cosine;
              best_code = cur_code - 1;
              best_x_dot_o = new_x_dot_o;
              best_o_norm_sq = new_o_norm_sq;
            }
          }
        }

        // Apply best change
        if (best_code != cur_code) {
          codes[idx] = best_code;
          x_dot_o = best_x_dot_o;
          o_norm_sq = best_o_norm_sq;
          round_changes++;
        }
      }
    }

    total_changes += round_changes;

    if (round_changes == 0) {
      break;  // Converged
    }
  }

  return total_changes;
}

}  // namespace saq
