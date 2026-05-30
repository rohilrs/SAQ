/// @file codebook_sizing_benchmark.cpp
/// @brief E4 — Per-dim sample-size-vs-k validation across synthetic, SIFT, and
/// MSMARCO PCA-rotated data. Compares kpp(sample) MSE vs kpp(largest-feasible)
/// proxy "full" MSE.

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
#include <sys/resource.h>
#include <vector>

namespace {

long peak_rss_kb() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Generate one column of n N(0,1) samples, seeded for reproducibility.
std::vector<float> make_gaussian(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

// Read one column from a row-major fvecs file (each row: int32 dim, then dim floats).
// Returns the first `n` row values of dimension `dim_idx`. Returns empty on error.
std::vector<float> read_fvecs_column(const std::string& path, size_t n, size_t dim_idx) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::vector<float> col;
    col.reserve(n);
    int32_t d_header = 0;
    while (col.size() < n && in.read(reinterpret_cast<char*>(&d_header), 4)) {
        size_t d = static_cast<size_t>(d_header);
        if (dim_idx >= d) return {};
        in.seekg(static_cast<std::streamoff>(dim_idx) * 4, std::ios::cur);
        float v = 0.f;
        if (!in.read(reinterpret_cast<char*>(&v), 4)) break;
        col.push_back(v);
        in.seekg(static_cast<std::streamoff>(d - dim_idx - 1) * 4, std::ios::cur);
    }
    return col;
}

struct Cell {
    std::string source;     // "synthetic_10m" | "synthetic_100m" | "sift_10m" | "msmarco_2m"
    size_t      bits;
    size_t      sample_k_ratio;  // S/k value swept (0 = "proxy_full")
    size_t      sample_size;
    uint64_t    seed;
    size_t      dim_idx;    // which column was used (synthetic: random; real: dim_idx in fvecs)
    double      build_seconds;
    double      mse_on_eval;  // MSE evaluated on a held-out eval slice
};

void emit_cell_json(const Cell& c, bool first) {
    std::printf("%s\n  {\"source\":\"%s\",\"bits\":%zu,\"sample_k_ratio\":%zu,"
                "\"sample_size\":%zu,\"seed\":%lu,\"dim_idx\":%zu,"
                "\"build_seconds\":%.4f,\"mse_on_eval\":%.6e}",
                first ? "" : ",", c.source.c_str(), c.bits, c.sample_k_ratio,
                c.sample_size, c.seed, c.dim_idx, c.build_seconds, c.mse_on_eval);
}

// Build a Lloyd codebook on `col_train` with the given sample size + seed.
// Returns (build_time, MSE evaluated on `col_eval`).
std::pair<double, double> run_lloyd_cell(
    const std::vector<float>& col_train,
    const std::vector<float>& col_eval,
    size_t bits, size_t sample_size, uint64_t seed)
{
    saq::LloydOpts opts;
    opts.max_bits     = bits;
    opts.init         = saq::CodebookInit::KMeansPlusPlus;
    opts.restarts     = 1;
    opts.seed         = seed;
    opts.sample_size  = sample_size;
    double t0 = now_s();
    saq::CodebookResult r = saq::build_codebook_lloyd(col_train, opts);
    double t1 = now_s();
    double mse = static_cast<double>(saq::codebook_mse(col_eval, r.codebooks[bits]));
    return {t1 - t0, mse};
}

}  // namespace

int main() {
    // -- E4 Sweep 1: Synthetic Gaussian, n=10M, b ∈ {8, 10, 12} --
    // Stream one column per cell; never materialize the full N×D matrix.
    // Use a fixed dim index for naming (synthetic doesn't have meaningful dim semantics).
    const size_t N = 10'000'000;
    const std::vector<size_t> bits_list = {8, 10, 12};
    const std::vector<size_t> sk_ratios = {50, 100, 200, 500, 1000};
    const std::vector<uint64_t> seeds = {0, 1};
    const size_t proxy_full_S = 2'000'000;  // largest practical kpp(full); covers up to b=12 (k=4096; 2M/4096 ≈ 488 S/k, well-converged per init-sizing benchmark)

    std::fprintf(stderr, "Generating eval column (n=%zu N(0,1), seed=99)...\n", N);
    auto col_eval = make_gaussian(N, 99);
    std::fprintf(stderr, "Generating train column (n=%zu N(0,1), seed=42)...\n", N);
    auto col_train = make_gaussian(N, 42);
    std::fprintf(stderr, "Both columns ready; peak_rss=%ld KB\n", peak_rss_kb());

    std::vector<Cell> cells;
    for (size_t bits : bits_list) {
        // proxy "full" (S=2M, fixed) — used as the baseline for ratio comparisons.
        for (uint64_t seed : seeds) {
            std::fprintf(stderr, "[synthetic_10m] b=%zu proxy_full S=%zu seed=%lu...\n",
                         bits, proxy_full_S, seed);
            auto [dt, mse] = run_lloyd_cell(col_train, col_eval, bits, proxy_full_S, seed);
            cells.push_back(Cell{"synthetic_10m", bits, 0, proxy_full_S, seed, 0, dt, mse});
            std::fprintf(stderr, "  t=%.2fs mse=%.4e\n", dt, mse);
        }
        const size_t k = size_t{1} << bits;
        for (size_t r : sk_ratios) {
            size_t S = r * k;
            if (S >= N) {
                std::fprintf(stderr, "[synthetic_10m] b=%zu S/k=%zu S=%zu SKIPPED (S>=N)\n",
                             bits, r, S);
                continue;
            }
            for (uint64_t seed : seeds) {
                std::fprintf(stderr, "[synthetic_10m] b=%zu S/k=%zu S=%zu seed=%lu...\n",
                             bits, r, S, seed);
                auto [dt, mse] = run_lloyd_cell(col_train, col_eval, bits, S, seed);
                cells.push_back(Cell{"synthetic_10m", bits, r, S, seed, 0, dt, mse});
                std::fprintf(stderr, "  t=%.2fs mse=%.4e\n", dt, mse);
            }
        }
    }

    std::printf("{\n \"cells\": [");
    bool first = true;
    for (const auto& c : cells) { emit_cell_json(c, first); first = false; }
    std::printf("\n ]\n}\n");
    std::fprintf(stderr, "\nDone. Cells: %zu, peak_rss=%ld KB\n", cells.size(), peak_rss_kb());
    return 0;
}
