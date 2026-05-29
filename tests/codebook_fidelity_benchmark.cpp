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
    const std::string pca_path =
        "data/datasets/dbpedia_100k/vectors_pca.fvecs";
    auto col = read_pca_column(pca_path, /*n=*/5000, /*dim_idx=*/0);
    if (col.size() != 5000) {
        std::fprintf(stderr, "dbpedia read FAILED (size=%zu). symlink ok?\n", col.size());
        return 1;
    }
    double I = estimate_cube_root_integral(col, /*num_bins=*/500);
    std::fprintf(stderr, "dbpedia dim0: I(cube_root) = %.6f\n", I);
    for (size_t b : {4, 6, 8, 10, 12}) {
        double lb = bennett_lower_bound(I, b);
        std::fprintf(stderr, "  Bennett lower bound at b=%zu: %.6e\n", b, lb);
    }
    std::printf("{\"status\":\"bennett_ok\"}\n");
    return 0;
}
