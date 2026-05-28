/// @file codebook_init_benchmark.cpp
/// @brief Speed + quality comparison of codebook init strategies.
///
/// Two sweeps:
///   1. Quality sweep (n=5000 per distribution, DP-optimal reference):
///      ratios mse_init / mse_dp at bit-rates {1,2,4,6,8}.
///   2. Speed sweep (n in {100k, 1M}, b in {6, 12}): wall time + full-data MSE.
///   3. Real-data sweep (dbpedia-100k PCA dims 0,100,500,1500, n=99000):
///      per-dim stats (mean, std, excess kurtosis) + quality ratio table vs DP
///      at bit-rates {1,2,4,6}. DP reference uses a sorted-stride sample of
///      DP_SAMPLE points (default 5000) so the O(k^2 * B^2) DP remains fast;
///      the same sample feeds both the DP reference and all Lloyd inits.
///      KMeans++(s=20k) uses a fresh random sample of the full 99000.
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
#include <numeric>
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

// fvecs reader for a specific column (0-indexed).
// Each row: int32_t d, then d float32 values. Row stride = 4 + d*4 bytes.
// Opens the file, reads d from row 0 to get stride, then seeks per row.
// Returns at most `take` values; empty on failure or if col >= d.
std::vector<float> read_fvecs_column(const std::string& path, size_t col, size_t take) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    int32_t d = 0;
    if (!f.read(reinterpret_cast<char*>(&d), sizeof(d))) return {};
    if (d <= 0 || col >= static_cast<size_t>(d)) return {};

    // Row stride in bytes: 4 (dim header) + d*4 (floats).
    const std::streamoff row_stride = 4 + static_cast<std::streamoff>(d) * 4;
    // Byte offset of col within row's float payload: col*4 after the int32 header.
    const std::streamoff col_offset_in_row = 4 + static_cast<std::streamoff>(col) * 4;

    std::vector<float> out;
    out.reserve(take);
    for (size_t row = 0; row < take; ++row) {
        std::streamoff pos = static_cast<std::streamoff>(row) * row_stride + col_offset_in_row;
        f.seekg(pos, std::ios::beg);
        float v = 0.f;
        if (!f.read(reinterpret_cast<char*>(&v), sizeof(v))) break;
        out.push_back(v);
    }
    return out;
}

// Count rows in an fvecs file (all rows assumed to have the same d).
// Returns 0 on failure.
size_t fvecs_row_count(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    std::streamoff total = f.tellg();
    f.seekg(0, std::ios::beg);
    int32_t d = 0;
    if (!f.read(reinterpret_cast<char*>(&d), sizeof(d)) || d <= 0) return 0;
    std::streamoff row_stride = 4 + static_cast<std::streamoff>(d) * 4;
    return static_cast<size_t>(total / row_stride);
}

// ---- Distribution statistics ----------------------------------------------

struct DistStats {
    double mean, std_dev, excess_kurtosis;
    size_t n;
};

DistStats compute_stats(const std::vector<float>& v) {
    DistStats s{};
    s.n = v.size();
    if (s.n < 2) return s;

    double sum = 0.0, sum2 = 0.0;
    for (float x : v) { sum += x; sum2 += double(x) * x; }
    s.mean = sum / s.n;
    double var = sum2 / s.n - s.mean * s.mean;
    s.std_dev = (var > 0.0) ? std::sqrt(var) : 0.0;

    if (s.std_dev < 1e-12) { s.excess_kurtosis = 0.0; return s; }
    double m4 = 0.0;
    for (float x : v) {
        double d = (double(x) - s.mean) / s.std_dev;
        double d2 = d * d;
        m4 += d2 * d2;
    }
    s.excess_kurtosis = m4 / s.n - 3.0;
    return s;
}

// ---- Helpers --------------------------------------------------------------

struct InitSpec {
    const char* name;
    saq::CodebookInit init;
    size_t sample_size;  // 0 = full
};

const std::vector<InitSpec>& quality_inits() {
    static const std::vector<InitSpec> v = {
        {"EqualMass",       saq::CodebookInit::EqualMassQuantile, 0},
        {"Uniform",         saq::CodebookInit::UniformSpaced,     0},
        {"CubeRoot",        saq::CodebookInit::CubeRootDensity,   0},
        {"KMeans++(full)",  saq::CodebookInit::KMeansPlusPlus,    0},
        {"KMeans++(s=2k)",  saq::CodebookInit::KMeansPlusPlus,    2000},
    };
    return v;
}

// Inits used in the real-data sweep (n=99000); KMeans++(s=20k) replaces s=2k.
const std::vector<InitSpec>& real_data_inits() {
    static const std::vector<InitSpec> v = {
        {"EqualMass",        saq::CodebookInit::EqualMassQuantile, 0},
        {"Uniform",          saq::CodebookInit::UniformSpaced,     0},
        {"CubeRoot",         saq::CodebookInit::CubeRootDensity,   0},
        {"KMeans++(full)",   saq::CodebookInit::KMeansPlusPlus,    0},
        {"KMeans++(s=20k)",  saq::CodebookInit::KMeansPlusPlus,    20000},
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

// ---- Real-data sweep (dbpedia-100k, selected PCA dims) --------------------
//
// DP reference: a sorted-stride sample of DP_SAMPLE points is drawn from the
// full 99000-row column.  The SAME sorted-stride sample feeds every Lloyd init
// so the comparison is apples-to-apples.  "Sorted-stride" means: sort the full
// column, then take every (n/DP_SAMPLE)-th element — this gives a representative
// spread across the distribution rather than a positional sample.
// KMeans++(s=20k) uses a separate random sample (seed 99) of 20000 points from
// the full 99000.
//
// max_bits is capped at REAL_MAX_BITS (6) so the DP remains feasible:
//   O(2^6 * DP_SAMPLE^2) = 64 * 5000^2 = 1.6e9 ops per dim (~2-5 s).

static constexpr size_t DP_SAMPLE     = 5000;  // sorted-stride sample for DP ref + Lloyd inits
static constexpr size_t REAL_MAX_BITS = 6;
static constexpr size_t N_REAL_ROWS   = 99000;

// Draw a sorted-stride sample of size `m` from sorted vector `sv`.
std::vector<float> sorted_stride_sample(const std::vector<float>& sv, size_t m) {
    const size_t n = sv.size();
    if (m >= n) return sv;
    std::vector<float> out;
    out.reserve(m);
    for (size_t i = 0; i < m; ++i) {
        size_t idx = (i * (n - 1)) / (m - 1);
        out.push_back(sv[idx]);
    }
    return out;
}

void run_real_dim(const std::string& fvecs_path, size_t col, size_t n_rows) {
    // 1. Read full column.
    auto t_read0 = clk::now();
    std::vector<float> full = read_fvecs_column(fvecs_path, col, n_rows);
    double t_read_ms = ms_since(t_read0);

    if (full.empty()) {
        std::printf("\nREAL-DATA DIM %zu SKIPPED: read returned empty (col OOB or IO error)\n", col);
        return;
    }
    const size_t n = full.size();

    // 2. Compute distribution stats on full column.
    DistStats st = compute_stats(full);
    std::printf("\n--- dim %zu  n=%zu  (read %.0f ms)\n", col, n, t_read_ms);
    std::printf("    mean=%.4f  std=%.4f  excess_kurtosis=%.3f\n",
                st.mean, st.std_dev, st.excess_kurtosis);
    std::printf("    (Gaussian=0, Laplace=3, >0 => heavier tails than Gaussian)\n");

    // 3. Build sorted-stride sample for DP reference (+ EqualMass/Uniform/CubeRoot Lloyd).
    std::vector<float> sorted_full = full;
    std::sort(sorted_full.begin(), sorted_full.end());
    std::vector<float> sample = sorted_stride_sample(sorted_full, std::min(DP_SAMPLE, n));
    // sample is already sorted (sorted_stride preserves order).

    std::printf("    DP reference on sorted-stride sample: n_sample=%zu, max_bits=%zu\n",
                sample.size(), REAL_MAX_BITS);

    // 4. DP reference on sample.
    auto t_dp0 = clk::now();
    auto dp = saq::build_codebook_dp(sample, REAL_MAX_BITS, /*num_bins=*/sample.size());
    double t_dp_ms = ms_since(t_dp0);
    std::printf("    DP build: %.0f ms\n", t_dp_ms);

    // 5. Quality + timing sweep.
    const std::vector<size_t> bits_list = {1, 2, 4, 6};

    print_header("Real PCA dim=" + std::to_string(col) +
                 " (dbpedia-100k, n=" + std::to_string(n) + ")");

    std::printf("%-20s", "init \\ bits");
    for (size_t b : bits_list) std::printf("  b=%-3zu", b);
    std::printf("\n");
    std::printf("--- quality ratio (mse_init / mse_dp on sample), lower = better ---\n");
    std::printf("    (EqualMass/Uniform/CubeRoot/KMeans++(full) train on sorted-stride sample n=%zu;\n"
                "     KMeans++(s=20k) trains on a random sample of 20000 from full n=%zu)\n",
                sample.size(), n);

    struct Row { std::vector<double> ratio, time_ms; };
    std::vector<Row> rows(real_data_inits().size());

    for (size_t ii = 0; ii < real_data_inits().size(); ++ii) {
        const InitSpec& spec = real_data_inits()[ii];
        rows[ii].ratio.assign(bits_list.size(), 0.0);
        rows[ii].time_ms.assign(bits_list.size(), 0.0);

        // Choose working set: KMeans++(s=20k) uses full column + sample_size opt;
        // all others use the sorted-stride sample directly.
        const bool use_full = (spec.init == saq::CodebookInit::KMeansPlusPlus
                               && spec.sample_size > 0);
        const std::vector<float>& working = use_full ? full : sample;

        for (size_t bi = 0; bi < bits_list.size(); ++bi) {
            size_t b = bits_list[bi];
            saq::LloydOpts opts;
            opts.max_bits    = b;
            opts.init        = spec.init;
            opts.restarts    = 1;
            opts.seed        = 99;
            opts.sample_size = use_full ? spec.sample_size : 0;

            auto t0 = clk::now();
            auto r  = saq::build_codebook_lloyd(working, opts);
            double t = ms_since(t0);

            // Evaluate MSE on the same sample that fed the DP reference.
            float mse = saq::codebook_mse(sample, r.codebooks[b]);
            double ratio = (dp.costs[b] > 1e-12f)
                ? double(mse) / double(dp.costs[b])
                : 1.0;
            rows[ii].ratio[bi]   = ratio;
            rows[ii].time_ms[bi] = t;
        }

        std::printf("%-20s", spec.name);
        for (double r : rows[ii].ratio) std::printf("  %6.3f", r);
        std::printf("\n");
    }

    std::printf("--- build wall-time (ms) ---\n");
    std::printf("%-20s", "init \\ bits");
    for (size_t b : bits_list) std::printf("  b=%-3zu", b);
    std::printf("\n");
    for (size_t ii = 0; ii < real_data_inits().size(); ++ii) {
        std::printf("%-20s", real_data_inits()[ii].name);
        for (double t : rows[ii].time_ms) std::printf("  %6.1f", t);
        std::printf("\n");
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

    // -------------------- Real-data sweep (dbpedia-100k PCA dims) ----------
    std::printf("\n\n========== REAL-DATA SWEEP (dbpedia-100k PCA dims) ==========\n");
    std::printf("Dims: 0 (highest variance), 100 (mid-high), 500 (mid-low), 1500 (tail-end).\n");
    std::printf("Full column n=%zu; DP reference on sorted-stride sample n=%zu, max_bits=%zu.\n",
                N_REAL_ROWS, DP_SAMPLE, REAL_MAX_BITS);

    const std::string fvecs_path = "data/datasets/dbpedia_100k/vectors_pca.fvecs";
    // Verify file is reachable before committing to all four dims.
    {
        std::ifstream probe(fvecs_path, std::ios::binary);
        if (!probe) {
            std::printf("\nREAL-DATA SWEEP SKIPPED: cannot open %s\n", fvecs_path.c_str());
            std::printf("  (symlink data/ -> ../SAQ/data may be missing; run:\n");
            std::printf("   ln -s ../SAQ/data/datasets data/datasets)\n");
            std::printf("\n=== done ===\n");
            return 0;
        }
    }

    size_t actual_rows = fvecs_row_count(fvecs_path);
    std::printf("File rows detected: %zu\n", actual_rows);
    const size_t rows_to_use = std::min(actual_rows, N_REAL_ROWS);

    for (size_t dim : {0ul, 100ul, 500ul, 1500ul}) {
        run_real_dim(fvecs_path, dim, rows_to_use);
    }

    std::printf("\n=== done ===\n");
    return 0;
}
