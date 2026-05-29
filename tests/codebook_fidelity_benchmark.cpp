/// @file codebook_fidelity_benchmark.cpp
/// @brief E2 + E3-strategy-2 — per-dim ratio-vs-DP at b ≤ 8 and ratio-vs-Bennett
/// at b ≥ 9, across the dbpedia PCA variance gradient.

#include "saq/preprocessing/codebook_builder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

// Read one column from a row-major fvecs file (each row: int32 dim, then dim floats).
// Returns the first `n` row values of dimension `dim_idx`. Returns empty on error.
std::vector<float> read_pca_column(const std::string& path, size_t n, size_t dim_idx) {
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

// Estimate ∫ f(x)^(1/3) dx from a uniform-width histogram of `values`.
//
// f(x_bin) ≈ count_bin / (n * width). Then g_bin = f(x_bin)^(1/3) * width,
// and the integral estimate is the sum of g_bin across bins. (Empty bins
// contribute 0.) Returns 0 if all values are equal (degenerate range).
double estimate_cube_root_integral(const std::vector<float>& values, size_t num_bins) {
    if (values.size() < 2) return 0.0;
    auto [lo_it, hi_it] = std::minmax_element(values.begin(), values.end());
    double lo = *lo_it, hi = *hi_it;
    if (hi - lo < 1e-12) return 0.0;
    double width = (hi - lo) / static_cast<double>(num_bins);
    std::vector<size_t> count(num_bins, 0);
    for (float v : values) {
        size_t bi = std::min<size_t>(num_bins - 1,
                                     static_cast<size_t>((static_cast<double>(v) - lo) / width));
        count[bi]++;
    }
    double sum = 0.0;
    double n_f = static_cast<double>(values.size());
    for (size_t i = 0; i < num_bins; ++i) {
        if (count[i] == 0) continue;
        double f_x = static_cast<double>(count[i]) / (n_f * width);
        sum += std::cbrt(f_x) * width;
    }
    return sum;
}

// Bennett / Panter–Dite high-resolution lower bound on scalar-quantizer MSE
// for `bits` bits per sample on the empirical distribution of `values`:
//
//   D >= (1/12) * 2^(-2 * bits) * (∫ f(x)^(1/3) dx)^3
//
// `cube_root_integral` is the precomputed estimate of ∫ f(x)^(1/3) dx
// (see estimate_cube_root_integral). Returns 0 if the integral is 0.
double bennett_lower_bound(double cube_root_integral, size_t bits) {
    if (cube_root_integral <= 0.0) return 0.0;
    double pow2 = std::pow(2.0, -2.0 * static_cast<double>(bits));
    return (1.0 / 12.0) * pow2 * std::pow(cube_root_integral, 3.0);
}

struct Cell {
    std::string source;       // e.g. "dbpedia_d0"
    size_t      bits;
    size_t      restart_seed; // 0,1,2 for the seed-variance sweep
    double      dp_mse;       // empty if b > 8
    double      lloyd_mse;
    double      bennett_lb;   // empty if b <= 8
    bool        has_dp;
    bool        has_bennett;
};

void emit_cell_json(const Cell& c, bool first) {
    std::printf("%s\n  {\"source\":\"%s\",\"bits\":%zu,\"seed\":%zu,"
                "\"lloyd_mse\":%.6e",
                first ? "" : ",", c.source.c_str(), c.bits, c.restart_seed,
                c.lloyd_mse);
    if (c.has_dp)      std::printf(",\"dp_mse\":%.6e", c.dp_mse);
    if (c.has_bennett) std::printf(",\"bennett_lb\":%.6e", c.bennett_lb);
    std::printf("}");
}

// Build a Lloyd codebook on `col_train`, evaluate on `col_eval` (held-out
// generalization MSE — what Bennett's asymptotic bound actually bounds).
double lloyd_mse_for(const std::vector<float>& col_train,
                     const std::vector<float>& col_eval,
                     size_t bits, uint64_t seed) {
    saq::LloydOpts opts;
    opts.max_bits = bits;
    opts.init     = saq::CodebookInit::KMeansPlusPlus;
    opts.restarts = 1;
    opts.seed     = seed;
    saq::CodebookResult r = saq::build_codebook_lloyd(col_train, opts);
    return static_cast<double>(saq::codebook_mse(col_eval, r.codebooks[bits]));
}

// Build the DP-optimal codebook on `col_train` (bits <= 8). Evaluate on `col_eval`.
double dp_mse_for(const std::vector<float>& col_train,
                  const std::vector<float>& col_eval,
                  size_t bits) {
    saq::CodebookResult r = saq::build_codebook_dp(col_train, bits, /*num_bins=*/500);
    return static_cast<double>(saq::codebook_mse(col_eval, r.codebooks[bits]));
}

long peak_rss_kb() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main() {
    const std::string pca_path = "data/datasets/dbpedia_100k/vectors_pca.fvecs";
    const size_t N_train = 10000;  // enough so Lloyd iterates at b=13 (k=8192 < N_train)
    const size_t N_eval  = 10000;  // held-out for generalization MSE (Bennett-comparable)
    const size_t N_total = N_train + N_eval;

    struct Source { std::string name; std::vector<float> col_train; std::vector<float> col_eval; double cube_root_I; };
    std::vector<Source> sources;
    for (size_t dim : {0, 100, 300, 500, 1000, 1500}) {
        std::string name = "dbpedia_d" + std::to_string(dim);
        std::vector<float> all = read_pca_column(pca_path, N_total, dim);
        if (all.size() != N_total) {
            std::fprintf(stderr, "source %s: read FAILED (got %zu, want %zu)\n",
                         name.c_str(), all.size(), N_total);
            return 1;
        }
        std::vector<float> col_train(all.begin(), all.begin() + N_train);
        std::vector<float> col_eval(all.begin() + N_train, all.end());
        // Bennett bound is a property of the density; estimate on the full data for stability.
        double I = estimate_cube_root_integral(all, /*num_bins=*/500);
        sources.push_back({name, std::move(col_train), std::move(col_eval), I});
        std::fprintf(stderr, "%-15s loaded N_train=%zu N_eval=%zu I(cube_root)=%.5f\n",
                     name.c_str(), N_train, N_eval, I);
    }

    const std::vector<size_t> e2_bits = {2, 4, 6, 8};
    const std::vector<size_t> e3_bits = {9, 10, 11, 12, 13};
    const std::vector<size_t> seeds   = {0, 1, 2};

    std::vector<Cell> cells;
    for (const auto& s : sources) {
        // E2 range: DP available, ratio vs DP.
        for (size_t b : e2_bits) {
            double dp = dp_mse_for(s.col_train, s.col_eval, b);
            for (uint64_t seed : seeds) {
                std::fprintf(stderr, "E2 %-15s b=%zu seed=%lu...\n", s.name.c_str(), b, seed);
                double ll = lloyd_mse_for(s.col_train, s.col_eval, b, seed);
                cells.push_back(Cell{s.name, b, seed, dp, ll, 0.0, true, false});
                std::fprintf(stderr, "  dp=%.4e lloyd=%.4e ratio=%.4f\n",
                             dp, ll, ll / std::max(dp, 1e-300));
            }
        }
        // E3 range: DP unavailable, ratio vs Bennett.
        for (size_t b : e3_bits) {
            double lb = bennett_lower_bound(s.cube_root_I, b);
            for (uint64_t seed : seeds) {
                std::fprintf(stderr, "E3 %-15s b=%zu seed=%lu...\n", s.name.c_str(), b, seed);
                double ll = lloyd_mse_for(s.col_train, s.col_eval, b, seed);
                cells.push_back(Cell{s.name, b, seed, 0.0, ll, lb, false, true});
                std::fprintf(stderr, "  lloyd=%.4e bennett_lb=%.4e ratio=%.4f\n",
                             ll, lb, ll / std::max(lb, 1e-300));
            }
        }
    }

    std::printf("{\n \"cells\": [");
    bool first = true;
    for (const auto& c : cells) { emit_cell_json(c, first); first = false; }
    std::printf("\n ]\n}\n");
    std::fprintf(stderr, "\nDone. Cells: %zu, peak_rss=%ld KB\n",
                 cells.size(), peak_rss_kb());
    return 0;
}
