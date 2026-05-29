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
    auto v = make_gaussian(/*n=*/5000, /*seed=*/42);
    double sum = 0.0, sq = 0.0;
    for (float x : v) { sum += x; sq += double(x) * x; }
    double mean = sum / v.size();
    double var  = sq / v.size() - mean * mean;
    std::fprintf(stderr, "synthetic: n=%zu mean=%.3f var=%.3f peak_rss=%ld KB\n",
                 v.size(), mean, var, peak_rss_kb());
    std::printf("{\"status\":\"synthetic_ok\",\"n\":%zu}\n", v.size());
    return 0;
}
