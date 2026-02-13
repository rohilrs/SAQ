#pragma once

/// @file caq_code_adjustment.h
/// @brief Context-Aware Quantization (CAQ) code adjustment for SAQ.
///
/// Implements Algorithm 1 from the SAQ paper (arXiv:2509.12086):
/// iterative per-dimension code refinement that adjusts scalar codes
/// by +/-1 to maximize cosine similarity between the original vector
/// and its reconstruction.

#include "saq/quantization_plan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace saq {

/// @brief Configuration for CAQ code adjustment.
struct CAQConfig {
  /// Number of refinement rounds (each round sweeps all dimensions).
  uint32_t num_rounds = 6;

  /// Convergence threshold: stop early if no codes change in a round.
  float convergence_threshold = 1e-4f;

  /// Number of parallel threads (0 = auto). Used in batch mode.
  uint32_t num_threads = 0;
};

/// @brief Statistics from CAQ adjustment.
struct CAQStats {
  /// Number of rounds performed.
  uint32_t rounds = 0;

  /// Cosine similarity before adjustment.
  float initial_cosine = 0.0f;

  /// Cosine similarity after adjustment.
  float final_cosine = 0.0f;

  /// Total number of code changes across all vectors.
  uint64_t codes_changed = 0;

  /// Whether convergence was reached (no changes in last round).
  bool converged = false;
};

/// @brief Per-dimension scalar code adjuster (Algorithm 1).
///
/// After initial scalar quantization, refines codes by iterating over
/// individual dimensions and trying c[i] +/- 1 adjustments. Accepts
/// changes that improve the cosine similarity between the original
/// rotated-space vector and its reconstruction.
///
/// Uses incremental updates for O(R * D) per-vector complexity,
/// maintaining running dot products and norms.
class CAQAdjuster {
 public:
  CAQAdjuster() = default;
  ~CAQAdjuster() = default;

  // Non-copyable, movable
  CAQAdjuster(const CAQAdjuster&) = delete;
  CAQAdjuster& operator=(const CAQAdjuster&) = delete;
  CAQAdjuster(CAQAdjuster&&) = default;
  CAQAdjuster& operator=(CAQAdjuster&&) = default;

  /// @brief Initialize with segment definitions.
  /// @param segments Segment definitions (with bits_per_dim set).
  /// @return True on success.
  bool Initialize(const std::vector<Segment>& segments);

  /// @brief Refine scalar codes to maximize cosine similarity.
  ///
  /// Implements Algorithm 1: for each round, sweep all dimensions,
  /// trying c[i]+1 and c[i]-1; accept if cosine similarity improves.
  ///
  /// @param original Original vector in rotated space.
  /// @param codes Per-dimension scalar codes (modified in-place).
  /// @param v_max Per-vector scaling factor.
  /// @param config CAQ configuration.
  /// @return Number of codes changed.
  uint64_t RefineScalar(const float* original, uint8_t* codes, float v_max,
                        const CAQConfig& config) const;

  /// @brief Reconstruct a vector from scalar codes.
  /// @param codes Per-dimension scalar codes.
  /// @param v_max Per-vector scaling factor.
  /// @param output Output buffer (size = total_dim_).
  void ReconstructScalar(const uint8_t* codes, float v_max,
                         float* output) const;

  /// @brief Get total dimensionality.
  uint32_t TotalDim() const { return total_dim_; }

  /// @brief Get number of segments.
  uint32_t NumSegments() const {
    return static_cast<uint32_t>(segments_.size());
  }

  /// @brief Check if initialized.
  bool IsInitialized() const { return !segments_.empty(); }

  // --- Legacy API (kept for backward compatibility) ---

  /// @brief Legacy: Initialize with codebooks (ignores codebooks,
  /// only uses segments).
  bool Initialize(const std::vector<Codebook>& codebooks,
                  const std::vector<Segment>& segments);

 private:
  std::vector<Segment> segments_;
  uint32_t total_dim_ = 0;
};

}  // namespace saq
