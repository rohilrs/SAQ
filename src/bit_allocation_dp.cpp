/// @file bit_allocation_dp.cpp
/// @brief Implementation of dynamic programming bit allocation.

#include "saq/bit_allocation_dp.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace saq {

namespace {

/// @brief Infinity value for DP table.
constexpr float kInfinity = std::numeric_limits<float>::max();

/// @brief Maximum DP table size before falling back to greedy.
/// Prevents excessive memory usage for large problems.
constexpr size_t kMaxDPTableSize = 100'000'000;

}  // namespace

float BitAllocatorDP::ComputeDistortion(float variance, uint32_t dim_count,
                                        uint32_t bits) {
  if (bits == 0) {
    // No quantization: distortion equals total variance
    return variance;
  }

  // Rate-distortion theory: D ≈ σ² * 2^(-2R)
  // where R is bits per dimension
  // For a segment: D ≈ (total_variance / dim_count) * dim_count * 2^(-2*bits/dim_count)
  //              ≈ total_variance * 2^(-2*bits/dim_count)
  //
  // Simplified model for segment-level quantization:
  // With 2^bits centroids covering dim_count dimensions,
  // distortion scales as variance * 2^(-2*bits/dim_count)
  if (dim_count == 0) {
    return 0.0f;
  }

  float bits_per_dim = static_cast<float>(bits) / static_cast<float>(dim_count);
  float scale = std::exp2(-2.0f * bits_per_dim);
  return variance * scale;
}

BitAllocationResult BitAllocatorDP::Allocate(
    const std::vector<float>& segment_variances,
    const std::vector<uint32_t>& segment_dims,
    const BitAllocationConfig& config) {
  BitAllocationResult result;

  // Validate inputs
  if (segment_variances.empty()) {
    result.error = "No segments provided";
    return result;
  }

  if (segment_variances.size() != segment_dims.size()) {
    result.error = "Variance and dimension vectors have different sizes";
    return result;
  }

  const size_t num_segments = segment_variances.size();
  const uint32_t total_budget = config.total_bits;
  const uint32_t min_bits = config.min_bits_per_segment;
  const uint32_t max_bits = config.max_bits_per_segment;

  // Check if budget is feasible
  uint32_t min_required = static_cast<uint32_t>(num_segments) * min_bits;
  uint32_t max_possible = static_cast<uint32_t>(num_segments) * max_bits;

  if (total_budget < min_required) {
    result.error = "Bit budget too small: need at least " +
                   std::to_string(min_required) + " bits for " +
                   std::to_string(num_segments) + " segments";
    return result;
  }

  if (total_budget > max_possible) {
    result.error = "Bit budget exceeds maximum: can use at most " +
                   std::to_string(max_possible) + " bits";
    return result;
  }

  // Decide between DP and greedy based on problem size
  size_t dp_table_size = num_segments * (total_budget + 1);
  if (dp_table_size > kMaxDPTableSize) {
    return AllocateGreedy(segment_variances, segment_dims, config);
  }

  return AllocateDP(segment_variances, segment_dims, config);
}

BitAllocationResult BitAllocatorDP::AllocateDP(
    const std::vector<float>& segment_variances,
    const std::vector<uint32_t>& segment_dims,
    const BitAllocationConfig& config) {
  BitAllocationResult result;

  const size_t num_segments = segment_variances.size();
  const uint32_t total_budget = config.total_bits;
  const uint32_t min_bits = config.min_bits_per_segment;
  const uint32_t max_bits = config.max_bits_per_segment;

  // DP table: dp[s][b] = minimum distortion for segments [0..s-1] using b bits
  // We use 1-indexed segments for cleaner code
  std::vector<std::vector<float>> dp(num_segments + 1,
                                      std::vector<float>(total_budget + 1, kInfinity));

  // Backtrack table: choice[s][b] = bits allocated to segment s-1
  std::vector<std::vector<uint32_t>> choice(num_segments + 1,
                                             std::vector<uint32_t>(total_budget + 1, 0));

  // Base case: zero segments use zero bits with zero distortion
  dp[0][0] = 0.0f;

  // Fill DP table
  for (size_t s = 1; s <= num_segments; ++s) {
    size_t seg_idx = s - 1;
    float variance = segment_variances[seg_idx];
    uint32_t dim_count = segment_dims[seg_idx];

    for (uint32_t budget = 0; budget <= total_budget; ++budget) {
      // Try each possible allocation for this segment
      for (uint32_t alloc = min_bits; alloc <= max_bits && alloc <= budget; ++alloc) {
        uint32_t remaining = budget - alloc;
        if (dp[s - 1][remaining] < kInfinity) {
          float distortion = ComputeDistortion(variance, dim_count, alloc);
          distortion += config.lambda * static_cast<float>(alloc);  // Regularization
          float total = dp[s - 1][remaining] + distortion;

          if (total < dp[s][budget]) {
            dp[s][budget] = total;
            choice[s][budget] = alloc;
          }
        }
      }
    }
  }

  // Check if solution exists
  if (dp[num_segments][total_budget] >= kInfinity) {
    result.error = "No valid allocation found for given constraints";
    return result;
  }

  // Backtrack to find allocation
  result.bits_per_segment.resize(num_segments);
  result.distortion_per_segment.resize(num_segments);

  uint32_t remaining_budget = total_budget;
  for (size_t s = num_segments; s >= 1; --s) {
    size_t seg_idx = s - 1;
    uint32_t alloc = choice[s][remaining_budget];
    result.bits_per_segment[seg_idx] = alloc;
    result.distortion_per_segment[seg_idx] =
        ComputeDistortion(segment_variances[seg_idx], segment_dims[seg_idx], alloc);
    remaining_budget -= alloc;
  }

  // Compute totals
  result.total_bits_used = std::accumulate(result.bits_per_segment.begin(),
                                            result.bits_per_segment.end(), 0u);
  result.total_distortion = std::accumulate(result.distortion_per_segment.begin(),
                                             result.distortion_per_segment.end(), 0.0f);

  return result;
}

BitAllocationResult BitAllocatorDP::AllocateGreedy(
    const std::vector<float>& segment_variances,
    const std::vector<uint32_t>& segment_dims,
    const BitAllocationConfig& config) {
  BitAllocationResult result;

  const size_t num_segments = segment_variances.size();
  const uint32_t min_bits = config.min_bits_per_segment;
  const uint32_t max_bits = config.max_bits_per_segment;

  // Initialize with minimum bits
  result.bits_per_segment.assign(num_segments, min_bits);
  uint32_t used_bits = static_cast<uint32_t>(num_segments) * min_bits;
  uint32_t remaining = config.total_bits - used_bits;

  // Priority queue: (distortion reduction, segment index)
  // We want to greedily add bits to the segment that reduces distortion the most
  auto compute_reduction = [&](size_t seg_idx) -> float {
    uint32_t current = result.bits_per_segment[seg_idx];
    if (current >= max_bits) {
      return -kInfinity;  // Can't add more bits
    }
    float d_current = ComputeDistortion(segment_variances[seg_idx],
                                         segment_dims[seg_idx], current);
    float d_next = ComputeDistortion(segment_variances[seg_idx],
                                      segment_dims[seg_idx], current + 1);
    return d_current - d_next;  // Positive = improvement
  };

  // Max-heap by distortion reduction
  using Entry = std::pair<float, size_t>;
  std::priority_queue<Entry> pq;

  for (size_t i = 0; i < num_segments; ++i) {
    float reduction = compute_reduction(i);
    if (reduction > 0) {
      pq.emplace(reduction, i);
    }
  }

  // Greedily allocate remaining bits
  while (remaining > 0 && !pq.empty()) {
    auto [reduction, seg_idx] = pq.top();
    pq.pop();

    // Re-check in case allocation changed
    float current_reduction = compute_reduction(seg_idx);
    if (current_reduction <= 0) {
      continue;
    }

    // If reduction changed, re-insert with updated value
    if (std::abs(current_reduction - reduction) > 1e-6f) {
      pq.emplace(current_reduction, seg_idx);
      continue;
    }

    // Allocate one more bit to this segment
    result.bits_per_segment[seg_idx]++;
    remaining--;

    // Re-insert if can still accept more bits
    float next_reduction = compute_reduction(seg_idx);
    if (next_reduction > 0) {
      pq.emplace(next_reduction, seg_idx);
    }
  }

  // Compute distortions
  result.distortion_per_segment.resize(num_segments);
  for (size_t i = 0; i < num_segments; ++i) {
    result.distortion_per_segment[i] = ComputeDistortion(
        segment_variances[i], segment_dims[i], result.bits_per_segment[i]);
  }

  result.total_bits_used = std::accumulate(result.bits_per_segment.begin(),
                                            result.bits_per_segment.end(), 0u);
  result.total_distortion = std::accumulate(result.distortion_per_segment.begin(),
                                             result.distortion_per_segment.end(), 0.0f);

  return result;
}

bool BitAllocatorDP::ApplyAllocation(const BitAllocationResult& result,
                                      std::vector<Segment>& segments) {
  if (!result.IsValid()) {
    return false;
  }

  if (result.bits_per_segment.size() != segments.size()) {
    return false;
  }

  for (size_t i = 0; i < segments.size(); ++i) {
    segments[i].bits = result.bits_per_segment[i];
  }

  return true;
}

}  // namespace saq
