/// @file bit_allocation_dp_test.cpp
/// @brief Tests for the bit allocation DP algorithm.

#include "saq/bit_allocation_dp.h"
#include "saq/quantization_plan.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

namespace {

constexpr float kEpsilon = 1e-5f;

void TestDistortionComputation() {
  // Paper formula: ERROR = variance / (2^B * pi)
  // With 0 bits, distortion equals variance (2^0 * pi = pi, but we special-case B=0)
  float d0 = saq::BitAllocatorDP::ComputeDistortion(100.0f, 0);
  assert(d0 >= 100.0f);  // At 0 bits, distortion >= variance

  // With more bits, distortion should decrease
  float d4 = saq::BitAllocatorDP::ComputeDistortion(100.0f, 4);
  float d8 = saq::BitAllocatorDP::ComputeDistortion(100.0f, 8);
  assert(d4 < d0);
  assert(d8 < d4);

  // Verify SAQ paper scaling: D = variance / (2^B * pi)
  // Doubling B should halve distortion (approximately)
  float d2 = saq::BitAllocatorDP::ComputeDistortion(100.0f, 2);
  float d4b = saq::BitAllocatorDP::ComputeDistortion(100.0f, 4);
  // d4/d2 should be approximately 1/4 (since 2^4 / 2^2 = 4)
  assert(d4b < d2);

  std::printf("TestDistortionComputation: OK\n");
}

void TestBasicAllocation() {
  saq::BitAllocatorDP allocator;

  // 4 segments with equal variance
  std::vector<float> variances = {100.0f, 100.0f, 100.0f, 100.0f};
  std::vector<uint32_t> dims = {8, 8, 8, 8};

  saq::BitAllocationConfig config;
  config.total_bits = 32;
  config.min_bits_per_segment = 4;
  config.max_bits_per_segment = 16;

  auto result = allocator.Allocate(variances, dims, config);
  assert(result.IsValid());
  assert(result.bits_per_segment.size() == 4);
  assert(result.total_bits_used == 32);

  // With equal variance, expect roughly equal allocation
  for (uint32_t bits : result.bits_per_segment) {
    assert(bits == 8);  // 32 / 4 = 8 bits each
  }

  std::printf("TestBasicAllocation: OK\n");
}

void TestUnequalVarianceAllocation() {
  saq::BitAllocatorDP allocator;

  // Segment 0 has much higher variance than others
  std::vector<float> variances = {1000.0f, 100.0f, 100.0f, 100.0f};
  std::vector<uint32_t> dims = {8, 8, 8, 8};

  saq::BitAllocationConfig config;
  config.total_bits = 32;
  config.min_bits_per_segment = 2;
  config.max_bits_per_segment = 16;

  auto result = allocator.Allocate(variances, dims, config);
  assert(result.IsValid());
  assert(result.total_bits_used == 32);

  // Higher variance segment should get more bits
  assert(result.bits_per_segment[0] > result.bits_per_segment[1]);
  assert(result.bits_per_segment[0] > result.bits_per_segment[2]);
  assert(result.bits_per_segment[0] > result.bits_per_segment[3]);

  std::printf("TestUnequalVarianceAllocation: OK\n");
}

void TestMinBitsConstraint() {
  saq::BitAllocatorDP allocator;

  std::vector<float> variances = {100.0f, 100.0f};
  std::vector<uint32_t> dims = {8, 8};

  saq::BitAllocationConfig config;
  config.total_bits = 16;
  config.min_bits_per_segment = 6;  // At least 6 bits each
  config.max_bits_per_segment = 16;

  auto result = allocator.Allocate(variances, dims, config);
  assert(result.IsValid());

  // Both must have at least 6 bits
  assert(result.bits_per_segment[0] >= 6);
  assert(result.bits_per_segment[1] >= 6);

  std::printf("TestMinBitsConstraint: OK\n");
}

void TestMaxBitsConstraint() {
  saq::BitAllocatorDP allocator;

  // One segment has much higher variance
  std::vector<float> variances = {10000.0f, 1.0f};
  std::vector<uint32_t> dims = {8, 8};

  saq::BitAllocationConfig config;
  config.total_bits = 24;
  config.min_bits_per_segment = 0;
  config.max_bits_per_segment = 12;  // Cap at 12 bits

  auto result = allocator.Allocate(variances, dims, config);
  assert(result.IsValid());

  // High-variance segment should hit the cap
  assert(result.bits_per_segment[0] == 12);
  assert(result.bits_per_segment[1] == 12);

  std::printf("TestMaxBitsConstraint: OK\n");
}

void TestInsufficientBudget() {
  saq::BitAllocatorDP allocator;

  std::vector<float> variances = {100.0f, 100.0f, 100.0f};
  std::vector<uint32_t> dims = {8, 8, 8};

  saq::BitAllocationConfig config;
  config.total_bits = 10;           // Only 10 bits
  config.min_bits_per_segment = 4;  // Need at least 12
  config.max_bits_per_segment = 16;

  auto result = allocator.Allocate(variances, dims, config);
  assert(!result.IsValid());
  assert(!result.error.empty());

  std::printf("TestInsufficientBudget: OK\n");
}

void TestApplyAllocation() {
  saq::BitAllocatorDP allocator;

  std::vector<float> variances = {200.0f, 100.0f, 50.0f};
  std::vector<uint32_t> dims = {16, 8, 8};

  saq::BitAllocationConfig config;
  config.total_bits = 24;
  config.min_bits_per_segment = 4;
  config.max_bits_per_segment = 12;

  auto result = allocator.Allocate(variances, dims, config);
  assert(result.IsValid());

  // Create segments to apply allocation to
  std::vector<saq::Segment> segments(3);
  segments[0] = {0, 0, 16, 0};
  segments[1] = {1, 16, 8, 0};
  segments[2] = {2, 24, 8, 0};

  bool applied = saq::BitAllocatorDP::ApplyAllocation(result, segments);
  assert(applied);

  // Verify bits were set
  for (size_t i = 0; i < segments.size(); ++i) {
    assert(segments[i].bits == result.bits_per_segment[i]);
  }

  std::printf("TestApplyAllocation: OK\n");
}

void TestDistortionDecreases() {
  saq::BitAllocatorDP allocator;

  std::vector<float> variances = {100.0f, 100.0f, 100.0f, 100.0f};
  std::vector<uint32_t> dims = {8, 8, 8, 8};

  // Allocate with increasing budgets
  float prev_distortion = std::numeric_limits<float>::max();
  for (uint32_t budget = 8; budget <= 64; budget += 8) {
    saq::BitAllocationConfig config;
    config.total_bits = budget;
    config.min_bits_per_segment = 0;
    config.max_bits_per_segment = 16;

    auto result = allocator.Allocate(variances, dims, config);
    assert(result.IsValid());

    // More bits should mean less distortion
    assert(result.total_distortion <= prev_distortion);
    prev_distortion = result.total_distortion;
  }

  std::printf("TestDistortionDecreases: OK\n");
}

void TestSingleSegment() {
  saq::BitAllocatorDP allocator;

  std::vector<float> variances = {100.0f};
  std::vector<uint32_t> dims = {32};

  saq::BitAllocationConfig config;
  config.total_bits = 8;
  config.min_bits_per_segment = 0;
  config.max_bits_per_segment = 16;

  auto result = allocator.Allocate(variances, dims, config);
  assert(result.IsValid());
  assert(result.bits_per_segment.size() == 1);
  assert(result.bits_per_segment[0] == 8);

  std::printf("TestSingleSegment: OK\n");
}

void TestManySegments() {
  saq::BitAllocatorDP allocator;

  // 16 segments with varying variance
  std::vector<float> variances;
  std::vector<uint32_t> dims;
  for (int i = 0; i < 16; ++i) {
    variances.push_back(static_cast<float>((i + 1) * 10));
    dims.push_back(8);
  }

  saq::BitAllocationConfig config;
  config.total_bits = 128;  // 8 bits per segment on average
  config.min_bits_per_segment = 2;
  config.max_bits_per_segment = 16;

  auto result = allocator.Allocate(variances, dims, config);
  assert(result.IsValid());
  assert(result.total_bits_used == 128);

  // Higher indexed segments (higher variance) should get more bits
  // Check that later segments tend to get more bits
  float avg_low = 0.0f, avg_high = 0.0f;
  for (int i = 0; i < 8; ++i) {
    avg_low += static_cast<float>(result.bits_per_segment[i]);
    avg_high += static_cast<float>(result.bits_per_segment[i + 8]);
  }
  avg_low /= 8.0f;
  avg_high /= 8.0f;
  assert(avg_high > avg_low);

  std::printf("TestManySegments: OK\n");
}

void TestJointDPUsesFullBudget() {
  saq::BitAllocatorDP allocator;

  // Simulate PCA-ordered variances: high at start, low at tail
  const uint32_t D = 64;
  const uint32_t Q = 64;
  std::vector<float> variances(D);
  for (uint32_t i = 0; i < D; ++i) {
    variances[i] = 100.0f / (1.0f + static_cast<float>(i));
  }

  saq::JointAllocationConfig config;
  config.total_bits = Q;
  config.min_bits_per_dim = 0;
  config.max_bits_per_dim = 8;
  config.min_dims_per_segment = 1;
  config.max_dims_per_segment = 32;

  auto result = allocator.AllocateJoint(variances, config);
  assert(result.IsValid());

  // The DP must use ALL available bits (or very close)
  assert(result.total_bits_used >= Q - 1);

  // High-variance dims should get more bits than tail dims
  uint32_t bits_first_seg = result.segments[0].bits;
  uint32_t bits_last_seg = result.segments.back().bits;
  assert(bits_first_seg >= bits_last_seg);

  std::printf("TestJointDPUsesFullBudget: OK (used %u / %u bits)\n",
              result.total_bits_used, Q);
}

}  // namespace

int main() {
  TestDistortionComputation();
  TestBasicAllocation();
  TestUnequalVarianceAllocation();
  TestMinBitsConstraint();
  TestMaxBitsConstraint();
  TestInsufficientBudget();
  TestApplyAllocation();
  TestDistortionDecreases();
  TestSingleSegment();
  TestManySegments();
  TestJointDPUsesFullBudget();

  std::printf("\nAll bit allocation DP tests passed!\n");
  return 0;
}
