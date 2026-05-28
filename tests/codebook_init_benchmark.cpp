/// @file codebook_init_benchmark.cpp
/// @brief Speed + quality comparison of codebook init strategies.
///
/// Two sweeps:
///   1. Quality sweep (n=5000 per distribution, DP-optimal reference):
///      ratios mse_init / mse_dp at bit-rates {1,2,4,6,8}.
///   2. Speed sweep (n in {100k, 1M}, b in {6, 12}): wall time + full-data MSE.
///
/// Manual-run benchmark — not part of the assertion suite.

#include "saq/preprocessing/codebook_builder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// ---- Distribution generators ----------------------------------------------

std::vector<float> gen_gaussian(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

// Laplace via inverse-CDF: X = -b * sign(U) * log(1 - 2|U|), U ~ Uniform(-0.5,0.5).
// b=1 -> variance 2, kurtosis 6 (heavy-tailed).
std::vector<float> gen_laplace(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ud(-0.5, 0.5);
    std::vector<float> v(n);
    for (auto& x : v) {
        double u = ud(rng);
        double sgn = (u >= 0.0) ? 1.0 : -1.0;
        double a = std::fabs(u);
        // Guard against log(0) at u = +/-0.5.
        double arg = 1.0 - 2.0 * a;
        if (arg <= 0.0) arg = 1e-12;
        x = static_cast<float>(-sgn * std::log(arg));
    }
    return v;
}

// fvecs reader: each row is int32 dim followed by dim float32 values.
// Returns dim 0 (first dimension) of the first `take` rows; empty on failure.
std::vector<float> read_fvecs_first_dim(const std::string& path, size_t take) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<float> out;
    out.reserve(take);
    while (out.size() < take) {
        int32_t d = 0;
        if (!f.read(reinterpret_cast<char*>(&d), sizeof(d))) break;
        if (d <= 0) break;
        std::vector<float> row(static_cast<size_t>(d));
        if (!f.read(reinterpret_cast<char*>(row.data()), sizeof(float) * d)) break;
        out.push_back(row[0]);
    }
    return out;
}

// ---- Helpers --------------------------------------------------------------

struct InitSpec {
    const char* name;
    saq::CodebookInit init;
    size_t sample_size;  // 0 = full
};

const std::vector<InitSpec>& quality_inits() {
    static const std::vector<InitSpec> v = {
        {"EqualMass",      saq::CodebookInit::EqualMassQuantile, 0},
        {"Uniform",        saq::CodebookInit::UniformSpaced,     0},
        {"CubeRoot",       saq::CodebookInit::CubeRootDensity,   0},
        {"KMeans++(full)", saq::CodebookInit::KMeansPlusPlus,    0},
        {"KMeans++(s=2k)", saq::CodebookInit::KMeansPlusPlus,    2000},
    };
    return v;
}

const std::vector<InitSpec>& speed_inits() {
    static const std::vector<InitSpec> v = {
        {"EqualMass",       saq::CodebookInit::EqualMassQuantile, 0},
        {"CubeRoot",        saq::CodebookInit::CubeRootDensity,   0},
        {"KMeans++(full)",  saq::CodebookInit::KMeansPlusPlus,    0},
        {"KMeans++(s=20k)", saq::CodebookInit::KMeansPlusPlus,    20000},
    };
    return v;
}

void print_header(const std::string& title) {
    std::printf("\n========== %s ==========\n", title.c_str());
}

// ---- Quality sweep --------------------------------------------------------

void run_quality(const std::string& dist_name, const std::vector<float>& v) {
    const std::vector<size_t> bits_list = {1, 2, 4, 6, 8};
    const size_t max_bits = 8;
    const size_t n = v.size();

    print_header(dist_name + " (n=" + std::to_string(n) + ")");

    // Exact DP reference (num_bins == n -> each point its own bin).
    auto dp = saq::build_codebook_dp(v, max_bits, /*num_bins=*/n);

    // Print bit-rate header.
    std::printf("%-18s", "init \\ bits");
    for (size_t b : bits_list) std::printf("  b=%-3zu", b);
    std::printf("\n");
    std::printf("--- quality ratio (mse_init / mse_dp), lower = better ---\n");

    // Cache per-init per-bit MSE + ratio + build time so we can also emit the time table.
    struct Row { std::vector<double> ratio, time_ms; };
    std::vector<Row> rows(quality_inits().size());

    for (size_t ii = 0; ii < quality_inits().size(); ++ii) {
        const InitSpec& spec = quality_inits()[ii];
        rows[ii].ratio.assign(bits_list.size(), 0.0);
        rows[ii].time_ms.assign(bits_list.size(), 0.0);

        for (size_t bi = 0; bi < bits_list.size(); ++bi) {
            size_t b = bits_list[bi];
            saq::LloydOpts opts;
            opts.max_bits    = b;
            opts.init        = spec.init;
            opts.restarts    = 1;
            opts.seed        = 42;
            opts.sample_size = spec.sample_size;

            auto t0 = clk::now();
            auto r  = saq::build_codebook_lloyd(v, opts);
            double t = ms_since(t0);

            float mse_full = saq::codebook_mse(v, r.codebooks[b]);
            double ratio = (dp.costs[b] > 1e-12f)
                ? double(mse_full) / double(dp.costs[b])
                : 1.0;
            rows[ii].ratio[bi]   = ratio;
            rows[ii].time_ms[bi] = t;
        }

        std::printf("%-18s", spec.name);
        for (double r : rows[ii].ratio) std::printf("  %6.3f", r);
        std::printf("\n");
    }

    std::printf("--- build wall-time (ms) ---\n");
    std::printf("%-18s", "init \\ bits");
    for (size_t b : bits_list) std::printf("  b=%-3zu", b);
    std::printf("\n");
    for (size_t ii = 0; ii < quality_inits().size(); ++ii) {
        std::printf("%-18s", quality_inits()[ii].name);
        for (double t : rows[ii].time_ms) std::printf("  %6.1f", t);
        std::printf("\n");
    }
}

// ---- Speed sweep ----------------------------------------------------------

void run_speed(size_t n, uint64_t seed) {
    const std::vector<size_t> bits_list = {6, 12};
    auto v = gen_gaussian(n, seed);

    print_header("Gaussian n=" + std::to_string(n));
    std::printf("%-18s | %-6s | %-12s | %-14s\n",
                "init", "bits", "build_ms", "mse_full");
    std::printf("---------------------------------------------------------------\n");

    for (const InitSpec& spec : speed_inits()) {
        for (size_t b : bits_list) {
            // KMeans++(full) at large n and high b is O(n*k) per iter; if we
            // expect >60s, skip and print a marker rather than hang.
            const bool kpp_full = (spec.init == saq::CodebookInit::KMeansPlusPlus
                                   && spec.sample_size == 0);
            if (kpp_full && n >= 1000000 && b >= 12) {
                std::printf("%-18s | b=%-4zu | %-12s | %-14s\n",
                            spec.name, b, "SKIP(>60s)", "n/a");
                continue;
            }

            saq::LloydOpts opts;
            opts.max_bits    = b;
            opts.init        = spec.init;
            opts.restarts    = 1;
            opts.seed        = 42;
            opts.sample_size = spec.sample_size;

            auto t0 = clk::now();
            auto r  = saq::build_codebook_lloyd(v, opts);
            double t_ms = ms_since(t0);

            float mse_full = saq::codebook_mse(v, r.codebooks[b]);
            std::printf("%-18s | b=%-4zu | %10.1f   | %12.6g\n",
                        spec.name, b, t_ms, mse_full);
        }
    }
}

}  // namespace

int main() {
    std::printf("=== Codebook init benchmark ===\n");
    std::printf("Comparing init strategies on quality (vs DP) and speed.\n");

    // -------------------- Quality sweep (n=5000) --------------------
    {
        const size_t n = 5000;
        run_quality("Gaussian N(0,1) seed=1", gen_gaussian(n, 1));
        run_quality("Laplace b=1 seed=2",      gen_laplace(n, 2));

        // Optional real-PCA dim if the dataset is present.
        const std::string fvecs = "data/datasets/dbpedia_100k/vectors_pca.fvecs";
        auto real = read_fvecs_first_dim(fvecs, n);
        if (!real.empty() && real.size() >= 1000) {
            run_quality("Real PCA dim0 (dbpedia_100k)", real);
        } else {
            std::printf("\n(skipping real PCA distribution: %s not found)\n",
                        fvecs.c_str());
        }
    }

    // -------------------- Speed sweep --------------------
    std::printf("\n\n========== SPEED SWEEP ==========\n");
    run_speed(100000, 11);
    run_speed(1000000, 12);

    std::printf("\n=== done ===\n");
    return 0;
}
