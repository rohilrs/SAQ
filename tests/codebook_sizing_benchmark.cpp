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
// Uses sequential buffered reads (no seekg per row) for performance on slow filesystems.
std::vector<float> read_fvecs_column(const std::string& path, size_t n, size_t dim_idx) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    // Read first row to get dim
    int32_t d_header = 0;
    if (!in.read(reinterpret_cast<char*>(&d_header), 4)) return {};
    size_t d = static_cast<size_t>(d_header);
    if (dim_idx >= d) return {};
    // Rewind to start
    in.seekg(0, std::ios::beg);

    // Row buffer: 4 bytes header + d*4 bytes floats
    const size_t row_bytes = 4 + d * 4;
    std::vector<char> row_buf(row_bytes);

    std::vector<float> col;
    col.reserve(n);
    while (col.size() < n) {
        if (!in.read(row_buf.data(), static_cast<std::streamsize>(row_bytes))) break;
        float v;
        std::memcpy(&v, row_buf.data() + 4 + dim_idx * 4, 4);
        col.push_back(v);
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

// Run the S/k sweep on a single (train, eval) column pair for `source` name.
std::vector<Cell> sweep_source(
    const std::string& source,
    const std::vector<float>& col_train,
    const std::vector<float>& col_eval,
    const std::vector<size_t>& bits_list,
    const std::vector<size_t>& sk_ratios,
    const std::vector<uint64_t>& seeds,
    size_t proxy_full_S,
    size_t dim_idx)
{
    std::vector<Cell> cells;
    for (size_t bits : bits_list) {
        // proxy "full" baseline
        for (uint64_t seed : seeds) {
            std::fprintf(stderr, "[%s] b=%zu proxy_full S=%zu seed=%lu...\n",
                         source.c_str(), bits, proxy_full_S, seed);
            auto [dt, mse] = run_lloyd_cell(col_train, col_eval, bits, proxy_full_S, seed);
            cells.push_back(Cell{source, bits, 0, proxy_full_S, seed, dim_idx, dt, mse});
            std::fprintf(stderr, "  t=%.2fs mse=%.4e\n", dt, mse);
        }
        const size_t k = size_t{1} << bits;
        const size_t N = col_train.size();
        for (size_t r : sk_ratios) {
            size_t S = r * k;
            if (S >= N) {
                std::fprintf(stderr, "[%s] b=%zu S/k=%zu S=%zu SKIPPED (S>=N)\n",
                             source.c_str(), bits, r, S);
                continue;
            }
            for (uint64_t seed : seeds) {
                std::fprintf(stderr, "[%s] b=%zu S/k=%zu S=%zu seed=%lu...\n",
                             source.c_str(), bits, r, S, seed);
                auto [dt, mse] = run_lloyd_cell(col_train, col_eval, bits, S, seed);
                cells.push_back(Cell{source, bits, r, S, seed, dim_idx, dt, mse});
                std::fprintf(stderr, "  t=%.2fs mse=%.4e\n", dt, mse);
            }
        }
    }
    return cells;
}

int main(int argc, char** argv) {
    // Args: --source <name> [--sift-fvecs <path>] [--msmarco-fvecs <path>] [--bits B,B,...] [--dim N]
    // Defaults: --source synthetic_10m
    std::string source = "synthetic_10m";
    std::string sift_fvecs, msmarco_fvecs;
    std::vector<size_t> bits_list = {8, 10, 12};
    size_t dim_idx = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--source" && i + 1 < argc) source = argv[++i];
        else if (a == "--sift-fvecs" && i + 1 < argc) sift_fvecs = argv[++i];
        else if (a == "--msmarco-fvecs" && i + 1 < argc) msmarco_fvecs = argv[++i];
        else if (a == "--dim" && i + 1 < argc) dim_idx = std::stoul(argv[++i]);
        else if (a == "--bits" && i + 1 < argc) {
            bits_list.clear();
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                bits_list.push_back(std::stoul(s.substr(pos, comma - pos)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
    }

    const std::vector<size_t> sk_ratios = {50, 100, 200, 500, 1000};
    const std::vector<uint64_t> seeds = {0, 1};

    std::vector<float> col_train, col_eval;
    size_t proxy_full_S = 0;

    if (source == "synthetic_10m") {
        const size_t N = 10'000'000;
        std::fprintf(stderr, "Generating Gaussian columns at n=%zu...\n", N);
        col_eval  = make_gaussian(N, 99);
        col_train = make_gaussian(N, 42);
        proxy_full_S = 2'000'000;
    } else if (source == "synthetic_100m") {
        const size_t N = 100'000'000;
        std::fprintf(stderr, "Generating Gaussian columns at n=%zu (memory: ~%.1f GB total)...\n",
                     N, 2.0 * N * 4 / 1e9);
        col_eval  = make_gaussian(N, 99);
        col_train = make_gaussian(N, 42);
        proxy_full_S = 10'000'000;  // largest practical at this scale
    } else if (source == "sift_10m" || source == "sift_1m") {
        if (sift_fvecs.empty()) {
            std::fprintf(stderr, "ERROR: --sift-fvecs required for source=%s\n", source.c_str());
            return 1;
        }
        // Use the same dim_idx for train and eval; eval is a separate dim
        // to avoid train/eval correlation (different PCA dims are decorrelated by construction).
        std::fprintf(stderr, "Reading SIFT column dim=%zu (train) and dim=%zu+1 (eval) from %s...\n",
                     dim_idx, dim_idx, sift_fvecs.c_str());
        size_t N = (source == "sift_10m") ? 10'000'000 : 1'000'000;
        col_train = read_fvecs_column(sift_fvecs, N, dim_idx);
        col_eval  = read_fvecs_column(sift_fvecs, N, dim_idx + 1);
        if (col_train.size() != N || col_eval.size() != N) {
            std::fprintf(stderr, "ERROR: fvecs read failed (train=%zu eval=%zu want=%zu)\n",
                         col_train.size(), col_eval.size(), N);
            return 1;
        }
        proxy_full_S = (source == "sift_10m") ? 2'000'000 : 500'000;
    } else if (source == "msmarco_2m") {
        if (msmarco_fvecs.empty()) {
            std::fprintf(stderr, "ERROR: --msmarco-fvecs required for source=%s\n", source.c_str());
            return 1;
        }
        std::fprintf(stderr, "Reading MSMARCO column dim=%zu (train) and dim=%zu+1 (eval) from %s...\n",
                     dim_idx, dim_idx, msmarco_fvecs.c_str());
        const size_t N = 2'000'000;
        col_train = read_fvecs_column(msmarco_fvecs, N, dim_idx);
        col_eval  = read_fvecs_column(msmarco_fvecs, N, dim_idx + 1);
        if (col_train.size() != N || col_eval.size() != N) {
            std::fprintf(stderr, "ERROR: fvecs read failed (train=%zu eval=%zu want=%zu)\n",
                         col_train.size(), col_eval.size(), N);
            return 1;
        }
        proxy_full_S = 1'000'000;
        // Spec: workstation MSMARCO covers b ∈ {8, 10} only; force the bits list.
        bits_list = {8, 10};
    } else {
        std::fprintf(stderr, "ERROR: unknown source=%s\n", source.c_str());
        return 1;
    }

    std::fprintf(stderr, "Columns ready; peak_rss=%ld KB\n", peak_rss_kb());

    auto cells = sweep_source(source, col_train, col_eval, bits_list,
                              sk_ratios, seeds, proxy_full_S, dim_idx);

    std::printf("{\n \"cells\": [");
    bool first = true;
    for (const auto& c : cells) { emit_cell_json(c, first); first = false; }
    std::printf("\n ]\n}\n");
    std::fprintf(stderr, "\nDone. Cells: %zu, peak_rss=%ld KB\n", cells.size(), peak_rss_kb());
    return 0;
}
