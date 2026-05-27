/// @file codebook_builder_test.cpp
/// @brief Tests for the data-driven codebook builder (Lloyd + DP reference).

#include "saq/preprocessing/codebook_builder.h"

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

}  // namespace

int main() {
    TestDpTwoClusters();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
