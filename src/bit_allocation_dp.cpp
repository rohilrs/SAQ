/// @file bit_allocation_dp.cpp
/// @brief Implementation of joint segmentation and bit allocation via DP.

#include "saq/bit_allocation_dp.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace saq {

namespace {

constexpr float kInfinity = std::numeric_limits<float>::max();
constexpr float kPi = 3.14159265358979323846f;

/// @brief Maximum DP table size before falling back to greedy.
constexpr size_t kMaxDPTableSize = 100'000'000;

/// @brief Maximum joint DP table size (D * Q) before greedy fallback.
constexpr size_t kMaxJointDPTableSize = 50'000'000;

}  // namespace

// =============================================================================
// New API: Joint segmentation + bit allocation (SAQ paper Algorithm 2)
// =============================================================================

float BitAllocatorDP::ComputeDistortion(float segment_variance,
                                        uint32_t bits_per_dim) {
  if (bits_per_dim == 0) {
    return segment_variance;
  }
  // SAQ paper: ERROR(Seg, B) = (1 / (2^B * pi)) * sum_{i in Seg} sigma_i^2
  float scale = 1.0f / (std::exp2(static_cast<float>(bits_per_dim)) * kPi);
  return segment_variance * scale;
}

JointAllocationResult BitAllocatorDP::AllocateJoint(
    const std::vector<float>& dim_variances,
    const JointAllocationConfig& config) {
  JointAllocationResult result;

  if (dim_variances.empty()) {
    result.error = "No dimension variances provided";
    return result;
  }

  if (config.total_bits == 0) {
    result.error = "Bit budget must be positive";
    return result;
  }

  const uint32_t D = static_cast<uint32_t>(dim_variances.size());
  const uint32_t Q = config.total_bits;

  // Check feasibility: even with max bits, can we cover all dimensions?
  // Minimum cost: each dim must be in some segment, min cost per dim = min_bits_per_dim
  // But dims with 0 bits are allowed (segment gets 0 bits = no quantization)

  // Choose DP vs greedy based on table size
  size_t table_size = static_cast<size_t>(D + 1) * static_cast<size_t>(Q + 1);
  if (table_size > kMaxJointDPTableSize) {
    return AllocateJointGreedy(dim_variances, config);
  }

  return AllocateJointDP(dim_variances, config);
}

JointAllocationResult BitAllocatorDP::AllocateJointDP(
    const std::vector<float>& dim_variances,
    const JointAllocationConfig& config) {
  JointAllocationResult result;

  const uint32_t D = static_cast<uint32_t>(dim_variances.size());
  const uint32_t Q = config.total_bits;
  const uint32_t min_b = config.min_bits_per_dim;
  const uint32_t max_b = config.max_bits_per_dim;
  const uint32_t min_seg = config.min_dims_per_segment;
  const uint32_t max_seg = config.max_dims_per_segment > 0
                               ? config.max_dims_per_segment
                               : D;

  // Prefix sum of variances for efficient range queries
  // cum_var[i] = sum of dim_variances[0..i-1]
  std::vector<double> cum_var(D + 1, 0.0);
  for (uint32_t i = 0; i < D; ++i) {
    cum_var[i + 1] = cum_var[i] + static_cast<double>(dim_variances[i]);
  }

  // DP table: dp[d][q] = min distortion covering dims [0..d-1] using exactly q bits
  // d ranges from 0 to D, q ranges from 0 to Q
  std::vector<std::vector<float>> dp(
      D + 1, std::vector<float>(Q + 1, kInfinity));

  // Backtrack: for each (d, q), store (segment_start, bits_per_dim)
  struct Choice {
    uint32_t seg_start = 0;
    uint32_t bits = 0;
  };
  std::vector<std::vector<Choice>> bt(D + 1, std::vector<Choice>(Q + 1));

  // Base case: 0 dimensions covered, 0 bits used
  dp[0][0] = 0.0f;

  // Fill DP table
  // For each ending dimension d' (1..D)
  for (uint32_t d_end = 1; d_end <= D; ++d_end) {
    // For each starting dimension of the last segment
    uint32_t seg_start_min = (d_end >= max_seg) ? d_end - max_seg : 0;
    uint32_t seg_start_max = (d_end >= min_seg) ? d_end - min_seg : 0;

    if (d_end < min_seg) continue;

    for (uint32_t d_start = seg_start_min; d_start <= seg_start_max; ++d_start) {
      uint32_t seg_len = d_end - d_start;

      // Variance of this segment [d_start..d_end-1]
      float seg_var = static_cast<float>(cum_var[d_end] - cum_var[d_start]);

      // Try each bits-per-dim allocation
      for (uint32_t b = min_b; b <= max_b; ++b) {
        uint32_t bit_cost = b * seg_len;
        float distortion = ComputeDistortion(seg_var, b);

        // For each prior budget level
        for (uint32_t q_prev = 0; q_prev + bit_cost <= Q; ++q_prev) {
          if (dp[d_start][q_prev] >= kInfinity) continue;

          uint32_t q_new = q_prev + bit_cost;
          float total = dp[d_start][q_prev] + distortion;

          if (total < dp[d_end][q_new]) {
            dp[d_end][q_new] = total;
            bt[d_end][q_new] = {d_start, b};
          }
        }
      }
    }
  }

  // Find best solution: dp[D][q] for any q <= Q
  // We want to use as much of the budget as possible for best quality,
  // but the DP already optimizes distortion, so take the minimum.
  float best_dist = kInfinity;
  uint32_t best_q = 0;
  for (uint32_t q = 0; q <= Q; ++q) {
    if (dp[D][q] < best_dist) {
      best_dist = dp[D][q];
      best_q = q;
    }
  }

  if (best_dist >= kInfinity) {
    result.error = "No valid segmentation found for given constraints";
    return result;
  }

  // Backtrack to reconstruct segments
  std::vector<Segment> segments;
  std::vector<float> distortions;
  uint32_t d = D;
  uint32_t q = best_q;

  while (d > 0) {
    const Choice& ch = bt[d][q];
    uint32_t seg_len = d - ch.seg_start;
    float seg_var = static_cast<float>(cum_var[d] - cum_var[ch.seg_start]);

    Segment seg;
    seg.id = 0;  // Will be assigned below
    seg.start_dim = ch.seg_start;
    seg.dim_count = seg_len;
    seg.bits = ch.bits;  // bits per dimension

    segments.push_back(seg);
    distortions.push_back(ComputeDistortion(seg_var, ch.bits));

    q -= ch.bits * seg_len;
    d = ch.seg_start;
  }

  // Reverse to get segments in dimension order
  std::reverse(segments.begin(), segments.end());
  std::reverse(distortions.begin(), distortions.end());

  // Assign segment IDs
  for (uint32_t i = 0; i < segments.size(); ++i) {
    segments[i].id = i;
  }

  result.segments = std::move(segments);
  result.distortion_per_segment = std::move(distortions);
  result.total_distortion = best_dist;
  result.total_bits_used = best_q;

  return result;
}

JointAllocationResult BitAllocatorDP::AllocateJointGreedy(
    const std::vector<float>& dim_variances,
    const JointAllocationConfig& config) {
  JointAllocationResult result;

  const uint32_t D = static_cast<uint32_t>(dim_variances.size());
  const uint32_t Q = config.total_bits;
  const uint32_t min_b = config.min_bits_per_dim;
  const uint32_t max_b = config.max_bits_per_dim;
  const uint32_t min_seg = config.min_dims_per_segment;

  // Greedy approach: start with uniform segments, then iteratively improve
  // Step 1: Create initial uniform segments
  uint32_t num_initial_segments = std::max(1u, Q / std::max(1u, min_seg * std::max(1u, min_b)));
  num_initial_segments = std::min(num_initial_segments, D / min_seg);
  if (num_initial_segments == 0) num_initial_segments = 1;

  uint32_t base_size = D / num_initial_segments;
  uint32_t remainder = D % num_initial_segments;

  std::vector<Segment> segments;
  uint32_t start = 0;
  for (uint32_t i = 0; i < num_initial_segments; ++i) {
    uint32_t seg_size = base_size + (i < remainder ? 1 : 0);
    Segment seg;
    seg.id = i;
    seg.start_dim = start;
    seg.dim_count = seg_size;
    seg.bits = min_b;
    segments.push_back(seg);
    start += seg_size;
  }

  // Step 2: Compute segment variances
  auto compute_seg_var = [&](const Segment& seg) -> float {
    float var = 0.0f;
    for (uint32_t d = seg.start_dim; d < seg.start_dim + seg.dim_count; ++d) {
      var += dim_variances[d];
    }
    return var;
  };

  // Step 3: Greedily allocate remaining bits
  uint32_t used_bits = 0;
  for (const auto& seg : segments) {
    used_bits += seg.bits * seg.dim_count;
  }

  // Priority queue: (distortion_reduction, segment_index)
  using Entry = std::pair<float, uint32_t>;
  std::priority_queue<Entry> pq;

  auto compute_reduction = [&](uint32_t seg_idx) -> float {
    const Segment& seg = segments[seg_idx];
    if (seg.bits >= max_b) return -kInfinity;
    float var = compute_seg_var(seg);
    float d_current = ComputeDistortion(var, seg.bits);
    float d_next = ComputeDistortion(var, seg.bits + 1);
    // Cost of adding 1 bit per dim = seg.dim_count total bits
    return (d_current - d_next);
  };

  for (uint32_t i = 0; i < segments.size(); ++i) {
    float red = compute_reduction(i);
    if (red > 0) pq.emplace(red, i);
  }

  while (used_bits < Q && !pq.empty()) {
    auto [reduction, seg_idx] = pq.top();
    pq.pop();

    Segment& seg = segments[seg_idx];
    uint32_t cost = seg.dim_count;  // 1 bit/dim * dim_count
    if (used_bits + cost > Q) continue;
    if (seg.bits >= max_b) continue;

    float current_red = compute_reduction(seg_idx);
    if (current_red <= 0) continue;

    if (std::abs(current_red - reduction) > 1e-6f) {
      pq.emplace(current_red, seg_idx);
      continue;
    }

    seg.bits++;
    used_bits += cost;

    float next_red = compute_reduction(seg_idx);
    if (next_red > 0) pq.emplace(next_red, seg_idx);
  }

  // Build result
  result.distortion_per_segment.resize(segments.size());
  result.total_distortion = 0.0f;
  for (size_t i = 0; i < segments.size(); ++i) {
    float var = compute_seg_var(segments[i]);
    result.distortion_per_segment[i] = ComputeDistortion(var, segments[i].bits);
    result.total_distortion += result.distortion_per_segment[i];
  }

  result.segments = std::move(segments);
  result.total_bits_used = used_bits;

  return result;
}

// =============================================================================
// Legacy API (kept for backward compatibility during transition)
// =============================================================================

BitAllocationResult BitAllocatorDP::Allocate(
    const std::vector<float>& segment_variances,
    const std::vector<uint32_t>& segment_dims,
    const BitAllocationConfig& config) {
  BitAllocationResult result;

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

  std::vector<std::vector<float>> dp(num_segments + 1,
                                      std::vector<float>(total_budget + 1, kInfinity));
  std::vector<std::vector<uint32_t>> choice(num_segments + 1,
                                             std::vector<uint32_t>(total_budget + 1, 0));

  dp[0][0] = 0.0f;

  for (size_t s = 1; s <= num_segments; ++s) {
    size_t seg_idx = s - 1;
    float variance = segment_variances[seg_idx];
    uint32_t dim_count = segment_dims[seg_idx];

    for (uint32_t budget = 0; budget <= total_budget; ++budget) {
      for (uint32_t alloc = min_bits; alloc <= max_bits && alloc <= budget; ++alloc) {
        uint32_t remaining = budget - alloc;
        if (dp[s - 1][remaining] < kInfinity) {
          // Legacy uses the same formula but with dim_count context
          float bits_per_dim_f = static_cast<float>(alloc) / static_cast<float>(dim_count);
          float scale = 1.0f / (std::exp2(bits_per_dim_f) * kPi);
          float distortion = variance * scale;
          distortion += config.lambda * static_cast<float>(alloc);
          float total = dp[s - 1][remaining] + distortion;

          if (total < dp[s][budget]) {
            dp[s][budget] = total;
            choice[s][budget] = alloc;
          }
        }
      }
    }
  }

  if (dp[num_segments][total_budget] >= kInfinity) {
    result.error = "No valid allocation found for given constraints";
    return result;
  }

  result.bits_per_segment.resize(num_segments);
  result.distortion_per_segment.resize(num_segments);

  uint32_t remaining_budget = total_budget;
  for (size_t s = num_segments; s >= 1; --s) {
    size_t seg_idx = s - 1;
    uint32_t alloc = choice[s][remaining_budget];
    result.bits_per_segment[seg_idx] = alloc;
    float bits_per_dim_f = static_cast<float>(alloc) / static_cast<float>(segment_dims[seg_idx]);
    float scale = 1.0f / (std::exp2(bits_per_dim_f) * kPi);
    result.distortion_per_segment[seg_idx] = segment_variances[seg_idx] * scale;
    remaining_budget -= alloc;
  }

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

  result.bits_per_segment.assign(num_segments, min_bits);
  uint32_t used_bits = static_cast<uint32_t>(num_segments) * min_bits;
  uint32_t remaining = config.total_bits - used_bits;

  auto compute_reduction = [&](size_t seg_idx) -> float {
    uint32_t current = result.bits_per_segment[seg_idx];
    if (current >= max_bits) return -kInfinity;
    uint32_t dim_count = segment_dims[seg_idx];
    float var = segment_variances[seg_idx];
    float bpd_cur = static_cast<float>(current) / static_cast<float>(dim_count);
    float bpd_next = static_cast<float>(current + 1) / static_cast<float>(dim_count);
    float d_current = var / (std::exp2(bpd_cur) * kPi);
    float d_next = var / (std::exp2(bpd_next) * kPi);
    return d_current - d_next;
  };

  using Entry = std::pair<float, size_t>;
  std::priority_queue<Entry> pq;

  for (size_t i = 0; i < num_segments; ++i) {
    float reduction = compute_reduction(i);
    if (reduction > 0) pq.emplace(reduction, i);
  }

  while (remaining > 0 && !pq.empty()) {
    auto [reduction, seg_idx] = pq.top();
    pq.pop();

    float current_reduction = compute_reduction(seg_idx);
    if (current_reduction <= 0) continue;

    if (std::abs(current_reduction - reduction) > 1e-6f) {
      pq.emplace(current_reduction, seg_idx);
      continue;
    }

    result.bits_per_segment[seg_idx]++;
    remaining--;

    float next_reduction = compute_reduction(seg_idx);
    if (next_reduction > 0) pq.emplace(next_reduction, seg_idx);
  }

  result.distortion_per_segment.resize(num_segments);
  for (size_t i = 0; i < num_segments; ++i) {
    uint32_t dim_count = segment_dims[i];
    float bpd = static_cast<float>(result.bits_per_segment[i]) /
                static_cast<float>(dim_count);
    result.distortion_per_segment[i] =
        segment_variances[i] / (std::exp2(bpd) * kPi);
  }

  result.total_bits_used = std::accumulate(result.bits_per_segment.begin(),
                                            result.bits_per_segment.end(), 0u);
  result.total_distortion = std::accumulate(result.distortion_per_segment.begin(),
                                             result.distortion_per_segment.end(), 0.0f);

  return result;
}

bool BitAllocatorDP::ApplyAllocation(const BitAllocationResult& result,
                                      std::vector<Segment>& segments) {
  if (!result.IsValid()) return false;
  if (result.bits_per_segment.size() != segments.size()) return false;

  for (size_t i = 0; i < segments.size(); ++i) {
    segments[i].bits = result.bits_per_segment[i];
  }

  return true;
}

}  // namespace saq
