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

}  // namespace

int main() {
    std::fprintf(stderr, "codebook_sizing_benchmark: scaffold ok, peak_rss=%ld KB\n", peak_rss_kb());
    std::printf("{\"status\":\"scaffold\"}\n");
    return 0;
}
