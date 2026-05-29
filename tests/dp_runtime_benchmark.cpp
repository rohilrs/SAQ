/// @file dp_runtime_benchmark.cpp
/// @brief E1/E1b — DP runtime + memory vs b on synthetic and real PCA dims,
/// plus Lloyd comparison. Emits JSON to stdout; stderr is a human summary.

#include "saq/preprocessing/codebook_builder.h"

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

// One column of n N(0,1) samples, seeded for reproducibility.
std::vector<float> make_gaussian(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

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
        // Skip d floats, but read the one at dim_idx.
        in.seekg(static_cast<std::streamoff>(dim_idx) * 4, std::ios::cur);
        float v = 0.f;
        if (!in.read(reinterpret_cast<char*>(&v), 4)) break;
        col.push_back(v);
        // Skip the rest of the row.
        in.seekg(static_cast<std::streamoff>(d - dim_idx - 1) * 4, std::ios::cur);
    }
    return col;
}

// Peak RSS in kilobytes (Linux: ru_maxrss is in KB).
long peak_rss_kb() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

// Wall-clock seconds since epoch, double precision.
double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main() {
    const std::string pca_path =
        "data/datasets/dbpedia_100k/vectors_pca.fvecs";
    auto col0 = read_pca_column(pca_path, /*n=*/5000, /*dim_idx=*/0);
    auto col1500 = read_pca_column(pca_path, /*n=*/5000, /*dim_idx=*/1500);
    if (col0.size() != 5000 || col1500.size() != 5000) {
        std::fprintf(stderr, "dbpedia read FAILED. Did you symlink data/?\n");
        return 1;
    }
    auto stats = [](const std::vector<float>& v) {
        double s = 0, sq = 0; for (float x : v) { s += x; sq += double(x) * x; }
        double m = s / v.size(); return std::pair<double,double>{m, sq / v.size() - m * m};
    };
    auto [m0, var0] = stats(col0);
    auto [m1500, var1500] = stats(col1500);
    std::fprintf(stderr,
                 "dbpedia dim0:    n=%zu mean=%.5f var=%.5f\n"
                 "dbpedia dim1500: n=%zu mean=%.5f var=%.5f\n"
                 "peak_rss=%ld KB\n",
                 col0.size(), m0, var0, col1500.size(), m1500, var1500, peak_rss_kb());
    std::printf("{\"status\":\"reader_ok\"}\n");
    return 0;
}
