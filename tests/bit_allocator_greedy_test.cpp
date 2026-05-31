/// @file bit_allocator_greedy_test.cpp
/// @brief Unit tests for BitAllocatorGreedy — 4 cases (trivial / unequal / capped / parity).

#include "saq/bit_allocator_greedy.h"

#include <Eigen/Dense>

#include <cassert>
#include <cstdio>
#include <vector>

namespace {
constexpr size_t kDimPad = 64;
constexpr size_t kMaxBits = 8;
constexpr size_t kNumBitFactors = 32;

saq::JointAllocationConfig make_config(size_t num_blocks, float avg_bits,
                                       size_t max_bits = kMaxBits) {
    saq::JointAllocationConfig c{};
    c.num_dim_padded     = num_blocks * kDimPad;
    c.dim_padding_size   = kDimPad;
    c.max_bits_per_dim   = max_bits;
    c.num_bit_factors    = kNumBitFactors;
    // Match the production dispatcher's total_bits formula (one factor of overhead,
    // not one-per-block) — see SaqDataMaker::analyze_plan.
    c.total_bits         = static_cast<size_t>(avg_bits * c.num_dim_padded) + c.num_bit_factors;
    return c;
}

Eigen::MatrixXf build_mse_table(const std::vector<float>& var_per_block, size_t max_bits = kMaxBits) {
    const size_t num_blocks = var_per_block.size();
    Eigen::MatrixXf m(static_cast<Eigen::Index>(num_blocks * kDimPad),
                      static_cast<Eigen::Index>(max_bits + 1));
    for (size_t i = 0; i < num_blocks; ++i) {
        for (size_t k = 0; k < kDimPad; ++k) {
            for (size_t b = 0; b <= max_bits; ++b) {
                // Synthetic cost matches the DP's closed-form: var / 2^b
                m(static_cast<Eigen::Index>(i * kDimPad + k),
                  static_cast<Eigen::Index>(b)) = var_per_block[i] / static_cast<float>(1u << b);
            }
        }
    }
    return m;
}

// Helper: extract per-block bit count from the quant_plan (de-merge segments).
std::vector<size_t> bits_per_block(const saq::BitAllocationResult& r, size_t num_blocks) {
    std::vector<size_t> out(num_blocks, 0);
    size_t cum_blocks = 0;
    for (const auto& seg : r.quant_plan) {
        const size_t seg_blocks = seg.first / kDimPad;
        for (size_t b = 0; b < seg_blocks; ++b) out[cum_blocks + b] = seg.second;
        cum_blocks += seg_blocks;
    }
    return out;
}

void TestTrivialOneBlock() {
    auto cfg = make_config(/*num_blocks=*/1, /*avg_bits=*/5.0f);
    auto mse = build_mse_table({100.0f});
    saq::BitAllocatorGreedy alloc;
    auto r = alloc.AllocateJoint(mse, cfg);
    assert(r.ok());
    assert(r.quant_plan.size() == 1);
    assert(r.quant_plan[0].first == kDimPad);
    assert(r.quant_plan[0].second == 5);
    std::printf("TestTrivialOneBlock: OK (1 seg of dim=%zu, bits=%zu)\n",
                r.quant_plan[0].first, r.quant_plan[0].second);
}

void TestTwoBlockUnequalVariance() {
    auto cfg = make_config(/*num_blocks=*/2, /*avg_bits=*/4.0f);
    auto mse = build_mse_table({1000.0f, 100.0f});  // 10x variance ratio
    saq::BitAllocatorGreedy alloc;
    auto r = alloc.AllocateJoint(mse, cfg);
    assert(r.ok());
    auto bits = bits_per_block(r, 2);
    assert(bits[0] > bits[1]);
    std::printf("TestTwoBlockUnequalVariance: OK (block0=%zu, block1=%zu)\n", bits[0], bits[1]);
}

void TestBitCap() {
    auto cfg = make_config(/*num_blocks=*/2, /*avg_bits=*/5.0f, /*max_bits=*/4);
    auto mse = build_mse_table({1e6f, 1.0f}, /*max_bits=*/4);
    saq::BitAllocatorGreedy alloc;
    auto r = alloc.AllocateJoint(mse, cfg);
    assert(r.ok());
    auto bits = bits_per_block(r, 2);
    // Block 0 wants huge bits; should hit the cap at 4. Surplus spills to block 1.
    assert(bits[0] == 4);
    assert(bits[1] > 0);
    std::printf("TestBitCap: OK (block0=%zu capped at max=4, block1=%zu)\n", bits[0], bits[1]);
}

void TestEqualVarianceParity() {
    auto cfg = make_config(/*num_blocks=*/4, /*avg_bits=*/4.0f);
    auto mse = build_mse_table({100.0f, 100.0f, 100.0f, 100.0f});
    saq::BitAllocatorGreedy alloc;
    auto r = alloc.AllocateJoint(mse, cfg);
    assert(r.ok());
    auto bits = bits_per_block(r, 4);
    // Uniform variance -> uniform allocation: all blocks should have the same bits.
    for (size_t i = 1; i < bits.size(); ++i) assert(bits[i] == bits[0]);
    // For avg_bits=4 with uniform variance, expect b=4 on every block.
    assert(bits[0] == 4);
    // After merging consecutive-equal blocks, expect 1 segment.
    assert(r.quant_plan.size() == 1);
    assert(r.quant_plan[0].first == 4 * kDimPad);
    assert(r.quant_plan[0].second == 4);
    std::printf("TestEqualVarianceParity: OK (uniform b=%zu, %zu seg merged)\n",
                bits[0], r.quant_plan.size());
}

// Regression test for the under-allocation bug — 24 blocks with a PCA-like
// descending variance gradient at avg_bits=2 (similar shape to dbpedia at b=2).
// Pre-fix greedy left ~43% of the budget unused; post-fix should use >=95%.
void TestBudgetUtilizationOnGradient() {
    const size_t num_blocks = 24;
    auto cfg = make_config(num_blocks, /*avg_bits=*/2.0f);

    // Build a synthetic 24-block PCA-rotated-style variance gradient:
    // var_per_block ranges from 1e-2 (head) to 1e-9 (tail), log-spaced.
    std::vector<float> vars(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        // 7 decades of decay across 24 blocks, ≈ 0.3 decades per block.
        vars[i] = static_cast<float>(1e-2 * std::pow(10.0, -7.0 * double(i) / double(num_blocks - 1)));
    }
    auto mse = build_mse_table(vars);
    saq::BitAllocatorGreedy alloc;
    auto r = alloc.AllocateJoint(mse, cfg);
    assert(r.ok());

    // Budget check: ≥ 95% of the budget should be used (was ~57% pre-fix).
    const double frac_used = double(r.total_bits_used) / double(cfg.total_bits);
    std::printf("TestBudgetUtilizationOnGradient: "
                "total_bits_used=%zu, tot_bits=%zu (%.1f%% used), n_segs=%zu\n",
                r.total_bits_used, cfg.total_bits, 100.0 * frac_used, r.quant_plan.size());
    assert(frac_used >= 0.95);
    // And we shouldn't overspend the budget.
    assert(r.total_bits_used <= cfg.total_bits);
}
}  // namespace

int main() {
    TestTrivialOneBlock();
    TestTwoBlockUnequalVariance();
    TestBitCap();
    TestEqualVarianceParity();
    TestBudgetUtilizationOnGradient();
    std::printf("\nAll bit_allocator_greedy tests passed!\n");
    return 0;
}
