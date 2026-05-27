/// @file codebook_builder_test.cpp
/// @brief Tests for the data-driven codebook builder (Lloyd + DP reference).

#include "saq/preprocessing/codebook_builder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
constexpr float kEps = 1e-4f;

// DP on a tiny, separated 2-cluster input: {0,0,10,10} at 1 bit -> centroids {0,10}, cost 0.
void TestDpTwoClusters() {
    std::vector<float> v = {0.f, 0.f, 10.f, 10.f};
    saq::CodebookResult r = saq::build_codebook_dp(v, /*max_bits=*/3, /*num_bins=*/500);
    assert(r.costs.size() == 4);
    assert(r.codebooks[1].num_entries == 2);
    assert(std::fabs(r.codebooks[1].centroids[0] - 0.f) < kEps);
    assert(std::fabs(r.codebooks[1].centroids[1] - 10.f) < kEps);
    assert(r.costs[1] < kEps);                 // perfect split -> ~0 MSE
    assert(r.costs[1] <= r.costs[0] + kEps);   // more bits never worse
    std::printf("TestDpTwoClusters: OK\n");
}

// Input with only 2 distinct values: the degenerate branch fires when 2^bits >= B
// (number of non-empty bins).  At bits=3 (8 clusters >= 2 bins) each distinct
// value gets its own centroid, so MSE is ~0.
void TestDpDegenerateBranch() {
    std::vector<float> v = {1.f, 1.f, 1.f, 5.f, 5.f};
    saq::CodebookResult r = saq::build_codebook_dp(v, /*max_bits=*/3, /*num_bins=*/500);
    // With only 2 distinct values, B == 2 non-empty bins.
    // For bits=3, 2^3=8 >= 2=B => degenerate branch: each bin its own centroid.
    assert(r.codebooks[3].num_entries == 2);
    assert(r.costs[3] < 1e-4f);
    // Centroids must be sorted ascending.
    assert(r.codebooks[3].centroids[0] <= r.codebooks[3].centroids[1]);
    std::printf("TestDpDegenerateBranch: OK\n");
}

// All values identical: every bit-rate should yield ~0 MSE and a valid centroid.
void TestDpAllEqual() {
    std::vector<float> v(100, 3.5f);
    saq::CodebookResult r = saq::build_codebook_dp(v, /*max_bits=*/4, /*num_bins=*/500);
    assert(r.costs.size() == 5);
    assert(r.codebooks.size() == 5);
    for (size_t bits = 0; bits <= 4; ++bits) {
        assert(r.codebooks[bits].num_entries >= 1);
        assert(std::isfinite(r.costs[bits]));
        assert(r.costs[bits] < 1e-4f);
    }
    assert(std::fabs(r.codebooks[0].centroids[0] - 3.5f) < 1e-4f);
    std::printf("TestDpAllEqual: OK\n");
}

// Gaussian samples: costs must be monotonically non-increasing with more bits,
// and each codebook's centroids must be sorted ascending.
void TestDpMonotonicMultiBit() {
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(2000);
    for (auto& x : v) x = dist(rng);

    saq::CodebookResult r = saq::build_codebook_dp(v, /*max_bits=*/6, /*num_bins=*/500);
    assert(r.costs.size() == 7);

    for (size_t bits = 1; bits <= 6; ++bits) {
        // Cost must be non-increasing (with a tiny tolerance for fp rounding).
        assert(r.costs[bits] <= r.costs[bits - 1] + 1e-6f);
        // Centroids must be sorted ascending.
        const auto& cen = r.codebooks[bits].centroids;
        assert(std::is_sorted(cen.begin(), cen.end()));
    }
    std::printf("TestDpMonotonicMultiBit: OK\n");
}

void TestLloydMonotonicAndDeterministic() {
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(5000);
    for (auto& x : v) x = nd(rng);

    saq::LloydOpts opts; opts.max_bits = 6;
    saq::CodebookResult a = saq::build_codebook_lloyd(v, opts);
    saq::CodebookResult b = saq::build_codebook_lloyd(v, opts);  // determinism

    for (size_t bits = 1; bits <= opts.max_bits; ++bits) {
        assert(a.codebooks[bits].num_entries == (size_t(1) << bits));
        assert(a.costs[bits] <= a.costs[bits - 1] + 1e-6f);          // monotone
        assert(std::fabs(a.costs[bits] - b.costs[bits]) < 1e-9f);    // deterministic
    }
    std::printf("TestLloydMonotonicAndDeterministic: OK\n");
}

void TestLloydDegenerate() {
    std::vector<float> v = {1.f, 1.f, 2.f, 2.f, 3.f, 3.f};  // 3 distinct values
    saq::LloydOpts opts; opts.max_bits = 5;
    saq::CodebookResult r = saq::build_codebook_lloyd(v, opts);
    assert(r.codebooks[5].num_entries == 3);  // k=32 >= 3 distinct -> 3 centroids
    assert(r.costs[5] < 1e-5f);               // exact representation
    std::printf("TestLloydDegenerate: OK\n");
}

}  // namespace

int main() {
    TestDpTwoClusters();
    TestDpDegenerateBranch();
    TestDpAllEqual();
    TestDpMonotonicMultiBit();
    TestLloydMonotonicAndDeterministic();
    TestLloydDegenerate();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
