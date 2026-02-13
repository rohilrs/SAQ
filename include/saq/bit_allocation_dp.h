#pragma once

/// @file bit_allocation_dp.h
/// @brief Joint dimension segmentation and bit allocation via dynamic programming.
///
/// Implements Algorithm 2 from the SAQ paper (arXiv:2509.12086).
/// Given per-dimension variances and a bit budget, jointly determines
/// optimal segment boundaries and bits-per-dimension for each segment.
/// Uses the paper's distortion model: ERROR(Seg, B) = (1/(2^B * pi)) * sum(sigma_i^2).

#include "saq/quantization_plan.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace saq {

/// @brief Configuration for joint segmentation and bit allocation.
struct JointAllocationConfig {
  /// Total bit budget per vector (sum of B_seg * |Seg| across all segments).
  uint32_t total_bits = 64;

  /// Minimum bits per dimension within a segment (0 = segment can be skipped).
  uint32_t min_bits_per_dim = 0;

  /// Maximum bits per dimension within a segment.
  uint32_t max_bits_per_dim = 8;

  /// Minimum number of dimensions per segment.
  uint32_t min_dims_per_segment = 1;

  /// Maximum number of dimensions per segment (0 = no limit).
  uint32_t max_dims_per_segment = 0;
};

/// @brief Result of joint segmentation and bit allocation.
struct JointAllocationResult {
  /// Segments determined by the DP (with bits_per_dim set in Segment.bits).
  std::vector<Segment> segments;

  /// Expected distortion per segment.
  std::vector<float> distortion_per_segment;

  /// Total expected distortion across all segments.
  float total_distortion = 0.0f;

  /// Total bits used (should equal budget).
  uint32_t total_bits_used = 0;

  /// Error message if allocation failed.
  std::string error;

  bool IsValid() const { return error.empty() && !segments.empty(); }
};

// --- Legacy types (kept at namespace scope for backward compatibility) ---

/// @brief Configuration for segment-level bit allocation (legacy).
struct BitAllocationConfig {
  uint32_t total_bits = 64;
  uint32_t min_bits_per_segment = 0;
  uint32_t max_bits_per_segment = 16;
  float lambda = 0.0f;
};

/// @brief Result of segment-level bit allocation (legacy).
struct BitAllocationResult {
  std::vector<uint32_t> bits_per_segment;
  std::vector<float> distortion_per_segment;
  float total_distortion = 0.0f;
  uint32_t total_bits_used = 0;
  std::string error;
  bool IsValid() const { return error.empty() && !bits_per_segment.empty(); }
};

/// @brief Joint dimension segmentation and bit allocation optimizer.
///
/// Implements the SAQ paper's Algorithm 2: a DP over dimensions that
/// jointly determines segment boundaries and per-dimension bit allocation.
///
/// State: dp[d][Q] = min distortion covering dims [0..d-1] with Q total bits.
/// Transition: extend with segment [d..d'] using B bits/dim, cost = B*(d'-d).
///
/// Distortion model: ERROR(Seg, B) = (1 / (2^B * pi)) * sum_{i in Seg} sigma_i^2
class BitAllocatorDP {
 public:
  BitAllocatorDP() = default;
  ~BitAllocatorDP() = default;

  // Non-copyable, movable
  BitAllocatorDP(const BitAllocatorDP&) = delete;
  BitAllocatorDP& operator=(const BitAllocatorDP&) = delete;
  BitAllocatorDP(BitAllocatorDP&&) = default;
  BitAllocatorDP& operator=(BitAllocatorDP&&) = default;

  /// @brief Joint segmentation and bit allocation over dimensions.
  ///
  /// This is the primary API matching the SAQ paper's Algorithm 2.
  /// Given per-dimension variances (from PCA-projected data), determines
  /// both segment boundaries and bits-per-dimension for each segment.
  ///
  /// @param dim_variances Per-dimension variance, size D (working dimension).
  /// @param config Joint allocation configuration.
  /// @return Joint result with segments, bits, and distortion.
  JointAllocationResult AllocateJoint(const std::vector<float>& dim_variances,
                                      const JointAllocationConfig& config);

  /// @brief Compute distortion for a segment with given total variance and bits.
  ///
  /// Uses SAQ paper formula: ERROR = (1 / (2^B * pi)) * sum(sigma_i^2).
  ///
  /// @param segment_variance Sum of per-dimension variances in the segment.
  /// @param bits_per_dim Bits per dimension (B in the paper).
  /// @return Expected quantization distortion.
  static float ComputeDistortion(float segment_variance, uint32_t bits_per_dim);

  // --- Legacy API (for backward compatibility during transition) ---

  /// @brief Legacy: Allocate bits to pre-defined segments.
  BitAllocationResult Allocate(const std::vector<float>& segment_variances,
                               const std::vector<uint32_t>& segment_dims,
                               const BitAllocationConfig& config);

  /// @brief Legacy: Apply allocation result to segment definitions.
  static bool ApplyAllocation(const BitAllocationResult& result,
                              std::vector<Segment>& segments);

 private:
  /// @brief Joint DP implementation.
  JointAllocationResult AllocateJointDP(const std::vector<float>& dim_variances,
                                        const JointAllocationConfig& config);

  /// @brief Greedy fallback for joint allocation.
  JointAllocationResult AllocateJointGreedy(const std::vector<float>& dim_variances,
                                            const JointAllocationConfig& config);

  /// @brief Legacy: Greedy allocation fallback.
  BitAllocationResult AllocateGreedy(const std::vector<float>& segment_variances,
                                     const std::vector<uint32_t>& segment_dims,
                                     const BitAllocationConfig& config);

  /// @brief Legacy: DP allocation for pre-fixed segments.
  BitAllocationResult AllocateDP(const std::vector<float>& segment_variances,
                                 const std::vector<uint32_t>& segment_dims,
                                 const BitAllocationConfig& config);
};

}  // namespace saq
