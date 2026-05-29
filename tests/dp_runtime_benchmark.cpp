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

struct Cell {
    std::string source;       // "synthetic" | "dbpedia_d0" | "dbpedia_d500" | "dbpedia_d1500"
    std::string method;       // "dp" | "lloyd_kpp"
    size_t      bits;
    double      seconds;
    long        peak_rss_kb_after;
    long        delta_rss_kb;
    float       cost_at_bits; // costs[bits], for sanity
};

void emit_cell_json(const Cell& c, bool first) {
    std::printf("%s\n  {\"source\":\"%s\",\"method\":\"%s\",\"bits\":%zu,"
                "\"seconds\":%.6f,\"peak_rss_kb\":%ld,\"delta_rss_kb\":%ld,"
                "\"cost_at_bits\":%.6g}",
                first ? "" : ",", c.source.c_str(), c.method.c_str(),
                c.bits, c.seconds, c.peak_rss_kb_after, c.delta_rss_kb,
                c.cost_at_bits);
}

Cell run_dp(const std::string& source, const std::vector<float>& v, size_t bits) {
    long rss_before = peak_rss_kb();
    double t0 = now_s();
    saq::CodebookResult r = saq::build_codebook_dp(v, bits, /*num_bins=*/500);
    double t1 = now_s();
    long rss_after = peak_rss_kb();
    return Cell{source, "dp", bits, t1 - t0, rss_after, rss_after - rss_before,
                r.costs.empty() ? 0.f : r.costs[bits]};
}

Cell run_lloyd(const std::string& source, const std::vector<float>& v, size_t bits) {
    saq::LloydOpts opts;
    opts.max_bits = bits;
    opts.init     = saq::CodebookInit::KMeansPlusPlus;
    opts.restarts = 1;
    opts.seed     = 0;
    long rss_before = peak_rss_kb();
    double t0 = now_s();
    saq::CodebookResult r = saq::build_codebook_lloyd(v, opts);
    double t1 = now_s();
    long rss_after = peak_rss_kb();
    return Cell{source, "lloyd_kpp", bits, t1 - t0, rss_after, rss_after - rss_before,
                r.costs.empty() ? 0.f : r.costs[bits]};
}

void print_summary(const std::vector<Cell>& cells, double budget_s) {
    std::fprintf(stderr, "\n=== Summary (budget=%.1fs/dim) ===\n", budget_s);
    std::fprintf(stderr, "%-16s | %-9s | %-4s | %10s | %10s | %12s\n",
                 "source", "method", "b", "time(s)", "rss_kb", "cost");
    std::fprintf(stderr, "%s\n", std::string(80, '-').c_str());
    for (const auto& c : cells) {
        std::fprintf(stderr, "%-16s | %-9s | %4zu | %10.4f | %10ld | %12.4g %s\n",
                     c.source.c_str(), c.method.c_str(), c.bits,
                     c.seconds, c.peak_rss_kb_after, c.cost_at_bits,
                     (c.method == "dp" && c.seconds > budget_s) ? "*OVER*" : "");
    }
    // Declare b*: the largest b for which dp.seconds <= budget across ALL sources.
    size_t b_star = 0;
    for (size_t b = 4; b <= 13; ++b) {
        bool all_within = false; bool any_dp = false;
        for (const auto& c : cells) {
            if (c.method == "dp" && c.bits == b) {
                any_dp = true;
                if (c.seconds <= budget_s) all_within = true; else { all_within = false; break; }
            }
        }
        if (any_dp && all_within) b_star = b;
    }
    std::fprintf(stderr, "\nb* (largest b with DP <= %.1fs across all sources) = %zu\n",
                 budget_s, b_star);
}

}  // namespace

int main() {
    const std::string pca_path = "data/datasets/dbpedia_100k/vectors_pca.fvecs";
    const size_t N = 5000;

    struct Source { std::string name; std::vector<float> col; };
    std::vector<Source> sources;
    sources.push_back({"synthetic",     make_gaussian(N, 42)});
    sources.push_back({"dbpedia_d0",    read_pca_column(pca_path, N, 0)});
    sources.push_back({"dbpedia_d500",  read_pca_column(pca_path, N, 500)});
    sources.push_back({"dbpedia_d1500", read_pca_column(pca_path, N, 1500)});
    for (auto& s : sources) {
        if (s.col.size() != N) {
            std::fprintf(stderr, "source %s: read FAILED (got %zu, want %zu)\n",
                         s.name.c_str(), s.col.size(), N);
            return 1;
        }
    }

    std::vector<Cell> cells;
    std::vector<size_t> dp_bits = {4, 5, 6, 7, 8};
    for (const auto& s : sources) {
        for (size_t b : dp_bits) {
            std::fprintf(stderr, "DP  %-16s b=%zu...\n", s.name.c_str(), b);
            Cell c = run_dp(s.name, s.col, b);
            std::fprintf(stderr, "  t=%.4fs delta_rss=%ld KB cost=%.4g\n",
                         c.seconds, c.delta_rss_kb, c.cost_at_bits);
            cells.push_back(c);
        }
    }
    std::vector<size_t> lloyd_bits = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    for (const auto& s : sources) {
        for (size_t b : lloyd_bits) {
            std::fprintf(stderr, "LLD %-16s b=%zu...\n", s.name.c_str(), b);
            Cell c = run_lloyd(s.name, s.col, b);
            std::fprintf(stderr, "  t=%.4fs delta_rss=%ld KB cost=%.4g\n",
                         c.seconds, c.delta_rss_kb, c.cost_at_bits);
            cells.push_back(c);
        }
    }

    std::printf("{\n \"cells\": [");
    bool first = true;
    for (const auto& c : cells) { emit_cell_json(c, first); first = false; }
    std::printf("\n ]\n}\n");

    print_summary(cells, /*budget_s=*/60.0);
    return 0;
}
