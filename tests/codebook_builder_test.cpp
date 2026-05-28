/// @file codebook_builder_test.cpp
/// @brief Tests for the data-driven codebook builder (Lloyd + DP reference).

#include "saq/preprocessing/codebook_builder.h"
#include "index/ivf_index.h"

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

// Duplicate-heavy input forces empty-cell repair: 990 zeros + 10 distinct
// values means equal-mass init places several coincident centroids in the
// zero mass, so cells go empty and must be repaired.
void TestLloydRepairDuplicateHeavy() {
    std::vector<float> v(990, 0.f);
    for (int i = 1; i <= 10; ++i) v.push_back(static_cast<float>(i));  // 11 distinct values
    saq::LloydOpts opts; opts.max_bits = 3;  // k up to 8 < 11 distinct -> Lloyd path
    saq::CodebookResult r = saq::build_codebook_lloyd(v, opts);
    for (size_t bits = 1; bits <= 3; ++bits) {
        const auto& cb = r.codebooks[bits].centroids;
        assert(r.codebooks[bits].num_entries == cb.size());           // honest count
        assert(r.codebooks[bits].num_entries == (size_t(1) << bits)); // padded to 2^bits
        // After dedup+pad, centroids are non-decreasing with strict increase among
        // unique values; trailing duplicates are allowed when dedup removed entries.
        for (size_t i = 1; i < cb.size(); ++i) assert(cb[i] >= cb[i - 1]);
        assert(std::isfinite(r.costs[bits]));
        assert(r.costs[bits] <= r.costs[bits - 1] + 1e-6f);           // monotone
    }
    std::printf("TestLloydRepairDuplicateHeavy: OK\n");
}

void TestLloydVsDp() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(1200);
    for (auto& x : v) x = nd(rng);

    saq::LloydOpts opts; opts.max_bits = 6;
    // Match the experiment's recommendation: at low k (k<=1024) use a few
    // restarts to suppress kpp seed-variance. See
    // research-hub/experiments/2026-05-27-codebook-init-sizing.md.
    opts.restarts = 3;
    saq::CodebookResult lloyd = saq::build_codebook_lloyd(v, opts);
    // num_bins >= n makes the DP *exact* (each point its own bin -> arbitrary
    // boundaries). Binned DP (num_bins < n) is only optimal among bin-edge
    // partitions, so its cost can exceed the true optimum and `lloyd >= dp`
    // would not be a valid invariant. Keep n modest so O(k*B^2) DP stays fast.
    saq::CodebookResult dp = saq::build_codebook_dp(v, /*max_bits=*/6, /*num_bins=*/1200);

    for (size_t bits = 1; bits <= 6; ++bits) {
        // DP is the global optimum for contiguous 1-D clustering: Lloyd >= DP.
        assert(lloyd.costs[bits] >= dp.costs[bits] - 1e-6f);
        // ...and Lloyd should be close to optimal on smooth (Gaussian) data.
        double ratio = (dp.costs[bits] > 1e-9f) ? lloyd.costs[bits] / dp.costs[bits] : 1.0;
        assert(ratio <= 1.15);  // within 15% of optimal
    }
    std::printf("TestLloydVsDp: OK\n");
}

// Run a non-default init on 3000 N(0,1) samples and validate the resulting
// codebook: full size at the top bit-rate, strictly-increasing centroids, and
// MSE that beats the 0-bit codebook and is finite.
void TestInitVariant(saq::CodebookInit init, const char* name) {
    std::mt19937 rng(2024);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(3000);
    for (auto& x : v) x = nd(rng);

    saq::LloydOpts opts;
    opts.max_bits = 6;
    opts.init = init;
    saq::CodebookResult r = saq::build_codebook_lloyd(v, opts);

    for (size_t bits = 1; bits <= opts.max_bits; ++bits) {
        const auto& cb = r.codebooks[bits];
        assert(cb.num_entries == (size_t(1) << bits));        // always 2^bits after dedup+pad
        assert(cb.num_entries == cb.centroids.size());        // honest count
        // Non-decreasing; strictly increasing in the unique prefix (trailing
        // duplicate entries are allowed when dedup removed collisions).
        for (size_t i = 1; i < cb.centroids.size(); ++i)
            assert(cb.centroids[i] >= cb.centroids[i - 1]);
        float mse = saq::codebook_mse(v, cb);
        assert(std::isfinite(mse));
        assert(mse <= r.costs[0] + 1e-6f);                    // better than 0-bit
    }
    // At max_bits the full-resolution codebook should be exactly 2^bits here
    // (3000 distinct Gaussian samples >> 64).
    assert(r.codebooks[opts.max_bits].num_entries == (size_t(1) << opts.max_bits));
    std::printf("TestInitVariant[%s]: OK\n", name);
}

void TestRecommendedSampleSize() {
    using saq::recommended_sample_size;
    // Capped at n when 500*(1<<max_bits) > n.
    assert(recommended_sample_size(1000, 12) == 1000);
    // The 200k absolute floor binds for small k.
    assert(recommended_sample_size(10'000'000, 4) == 200'000);   // 500*16 = 8k < 200k -> floor
    assert(recommended_sample_size(10'000'000, 8) == 200'000);   // 500*256 = 128k < 200k -> floor
    // The 500*k term binds for larger k.
    assert(recommended_sample_size(10'000'000, 10) == 500 * (size_t{1} << 10));  // 512000
    assert(recommended_sample_size(100'000'000, 12) == 500 * (size_t{1} << 12)); // 2_048_000
    // Capped at n when 500*k > n.
    assert(recommended_sample_size(1'000'000, 12) == 1'000'000); // 500*4096=2M > 1M -> n
    std::printf("TestRecommendedSampleSize: OK\n");
}

void TestBuildAllDims() {
    const int N = 2000, D = 8;
    saq::FloatRowMat data(N, D);
    std::mt19937 rng(5);
    for (int d = 0; d < D; ++d) {
        std::normal_distribution<float> nd(0.f, 1.f + d);  // per-dim variance grows
        for (int i = 0; i < N; ++i) data(i, d) = nd(rng);
    }
    saq::LloydOpts opts; opts.max_bits = 5;
    std::vector<saq::CodebookResult> all = saq::build_all_dims(data, opts);

    assert(all.size() == static_cast<size_t>(D));
    // Per-dim result must match building that column directly.
    std::vector<float> col(N);
    for (int i = 0; i < N; ++i) col[i] = data(i, 3);
    saq::CodebookResult direct = saq::build_codebook_lloyd(col, opts);
    for (size_t bits = 0; bits <= opts.max_bits; ++bits)
        assert(std::fabs(all[3].costs[bits] - direct.costs[bits]) < 1e-6f);
    std::printf("TestBuildAllDims: OK\n");
}

void TestNativeDerivationEndToEnd() {
    // D must be a multiple of kDimPaddingSize (64). Use D=128 so the DP
    // allocator has at least 2 blocks (max_num_segs >= 1 at avg_bits=4) and
    // can produce non-zero-bit segments, which exercises the codebook
    // selection logic in construct().
    const int N = 1000, D = 128, K = 4;
    saq::FloatRowMat data(N, D);
    std::mt19937 rng(9);
    // Scale by 10 so the DP sees meaningful variance differences across dims.
    std::normal_distribution<float> nd(0.f, 10.f);
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < D; ++d) data(i, d) = nd(rng);

    // Minimal centroids + cluster ids for a tiny IVF.
    saq::FloatRowMat centroids(K, D); centroids.setZero();
    std::vector<saq::PID> cluster_ids(N);
    for (int i = 0; i < N; ++i) cluster_ids[i] = static_cast<saq::PID>(i % K);

    saq::QuantizeConfig cfg; cfg.avg_bits = 4;
    saq::IVF ivf(static_cast<size_t>(N), static_cast<size_t>(D),
                 static_cast<size_t>(K), cfg);
    saq::LloydOpts opts; opts.max_bits = 13;
    ivf.set_derive_codebooks(opts);
    ivf.construct(data, centroids, cluster_ids.data());  // must not crash; derives natively

    const auto& sd = *ivf.get_saq_data();

    // Print the quant_plan so failures are diagnosable.
    for (auto [dl, bb] : sd.quant_plan)
        std::printf("  seg dim_len=%zu bits=%zu\n", dl, bb);

    assert(sd.segment_codebooks.size() == sd.quant_plan.size());
    assert(!sd.codebook_costs.empty());

    // Per-dim costs vector populated for every dim.
    assert(sd.codebook_costs.size() == static_cast<size_t>(D));
    for (const auto& c : sd.codebook_costs) {
        assert(c.size() == opts.max_bits + 1);
        for (size_t b = 1; b < c.size(); ++b) assert(c[b] <= c[b - 1] + 1e-6f); // monotone
    }

    // For every non-zero-bit segment, codebooks_ entries match expected shape.
    bool saw_nonzero_segment = false;
    for (size_t s = 0; s < sd.quant_plan.size(); ++s) {
        const size_t dim_len = sd.quant_plan[s].first;
        const size_t bits    = sd.quant_plan[s].second;
        if (bits == 0) continue;
        saw_nonzero_segment = true;
        assert(sd.segment_codebooks[s].size() == dim_len);
        for (size_t j = 0; j < dim_len; ++j) {
            assert(sd.segment_codebooks[s][j].num_entries == (size_t(1) << bits));
            // sorted-ascending (non-decreasing) invariant
            const auto& cv = sd.segment_codebooks[s][j].centroids;
            for (size_t i = 1; i < cv.size(); ++i) assert(cv[i] >= cv[i - 1]);
        }
    }
    assert(saw_nonzero_segment && "test config produced no non-zero-bit segments; bump avg_bits or D");

    std::printf("TestNativeDerivationEndToEnd: OK\n");
}

}  // namespace

int main() {
    TestDpTwoClusters();
    TestDpDegenerateBranch();
    TestDpAllEqual();
    TestDpMonotonicMultiBit();
    TestLloydMonotonicAndDeterministic();
    TestLloydDegenerate();
    TestLloydRepairDuplicateHeavy();
    TestInitVariant(saq::CodebookInit::UniformSpaced, "UniformSpaced");
    TestInitVariant(saq::CodebookInit::KMeansPlusPlus, "KMeansPlusPlus");
    TestInitVariant(saq::CodebookInit::CubeRootDensity, "CubeRootDensity");
    TestLloydVsDp();
    TestRecommendedSampleSize();
    TestBuildAllDims();
    TestNativeDerivationEndToEnd();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
