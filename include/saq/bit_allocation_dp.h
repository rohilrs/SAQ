#pragma once

/// @file bit_allocation_dp.h
/// @brief Dynamic programming bit allocation for SAQ segments.
///
/// Implements the optimal bit allocation algorithm from the SAQ paper.
/// Given a bit budget and segment variances, finds the allocation that
/// minimizes total quantization distortion using rate-distortion theory.

#include "saq/quantization_plan.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace saq {

/// @brief Configuration for bit allocation.
struct BitAllocationConfig {
  /// Total bit budget per vector (across all segments).
  uint32_t total_bits = 64;

  /// Minimum bits per segment (0 means segment can be skipped).
  uint32_t min_bits_per_segment = 0;

  /// Maximum bits per segment (caps individual allocations).
  uint32_t max_bits_per_segment = 16;

  /// Regularization factor for distortion (higher = more uniform allocation).
  float lambda = 0.0f;
};

/// @brief Result of bit allocation optimization.
struct BitAllocationResult {
  /// Bit allocations per segment (indices match input segments).
  std::vector<uint32_t> bits_per_segment;

  /// Expected distortion per segment given allocation.
  std::vector<float> distortion_per_segment;

  /// Total expected distortion across all segments.
  float total_distortion = 0.0f;

  /// Total bits allocated (should equal budget if successful).
  uint32_t total_bits_used = 0;

  /// Error message if allocation failed.
  std::string error;

  bool IsValid() const { return error.empty() && !bits_per_segment.empty(); }
};

/// @brief Optimal bit allocation using dynamic programming.
///
/// Uses rate-distortion theory to minimize quantization distortion.
/// The distortion model assumes that for a segment with variance σ²
/// and b bits allocated, the expected distortion is approximately:
///   D(σ², b) ≈ σ² * 2^(-2b)
///
/// The DP finds the allocation that minimizes total distortion subject
/// to the bit budget constraint.
class BitAllocatorDP {
 public:
  BitAllocatorDP() = default;
  ~BitAllocatorDP() = default;

  // Non-copyable, movable
  BitAllocatorDP(const BitAllocatorDP&) = delete;
  BitAllocatorDP& operator=(const BitAllocatorDP&) = delete;
  BitAllocatorDP(BitAllocatorDP&&) = default;
  BitAllocatorDP& operator=(BitAllocatorDP&&) = default;

  /// @brief Compute optimal bit allocation for segments.
  /// @param segment_variances Total variance for each segment.
  /// @param segment_dims Number of dimensions in each segment.
  /// @param config Allocation configuration.
  /// @return Allocation result (check IsValid()).
  BitAllocationResult Allocate(const std::vector<float>& segment_variances,
                               const std::vector<uint32_t>& segment_dims,
                               const BitAllocationConfig& config);

  /// @brief Apply allocation result to segment definitions.
  /// @param result Allocation result from Allocate().
  /// @param segments Segments to update (bits field will be set).
  /// @return True if segments were updated successfully.
  static bool ApplyAllocation(const BitAllocationResult& result,
                              std::vector<Segment>& segments);

  /// @brief Compute distortion for a given variance and bit allocation.
  /// @param variance Segment variance (sum across dimensions).
  /// @param dim_count Number of dimensions in segment.
  /// @param bits Bits allocated to segment.
  /// @return Expected quantization distortion.
  static float ComputeDistortion(float variance, uint32_t dim_count,
                                 uint32_t bits);

 private:
  /// @brief Greedy allocation as fallback when DP is infeasible.
  BitAllocationResult AllocateGreedy(const std::vector<float>& segment_variances,
                                     const std::vector<uint32_t>& segment_dims,
                                     const BitAllocationConfig& config);

  /// @brief Full DP allocation for optimal solution.
  BitAllocationResult AllocateDP(const std::vector<float>& segment_variances,
                                 const std::vector<uint32_t>& segment_dims,
                                 const BitAllocationConfig& config);
};

}  // namespace saq
