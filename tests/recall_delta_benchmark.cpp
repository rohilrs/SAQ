/// @file recall_delta_benchmark.cpp
/// @brief E2b — Recall@10 delta between IVFs built with kpp+Lloyd-derived
/// codebooks vs DP-derived codebooks, on dbpedia-100K.
///
/// Strategy:
///   1. Build IVF #1 with set_derive_codebooks(KMeansPlusPlus). Capture
///      saq_data_->quant_plan from get_saq_data(). Search the queries.
///      Compute Recall@10.
///   2. For each global dim, run build_codebook_dp at the segment's bits,
///      assemble codebooks_[seg][dim], call set_codebooks.
///      Build IVF #2 with those injected codebooks. Search. Recall@10.
///   3. Emit JSON: {recall_lloyd, recall_dp, delta_pp}.

#include "index/ivf_index.h"
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
#include <unordered_set>
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

// Read a full row-major fvecs file into an Eigen FloatRowMat of (N, D).
saq::FloatRowMat read_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::fprintf(stderr, "open failed: %s\n", path.c_str()); std::exit(1); }
    in.seekg(0, std::ios::end);
    std::streamoff bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    int32_t d0 = 0;
    in.read(reinterpret_cast<char*>(&d0), 4);
    in.seekg(0, std::ios::beg);
    size_t row_bytes = 4 + static_cast<size_t>(d0) * 4;
    size_t n = static_cast<size_t>(bytes) / row_bytes;
    saq::FloatRowMat M(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(d0));
    for (size_t i = 0; i < n; ++i) {
        int32_t d = 0;
        in.read(reinterpret_cast<char*>(&d), 4);
        in.read(reinterpret_cast<char*>(M.row(static_cast<Eigen::Index>(i)).data()),
                static_cast<std::streamsize>(d) * 4);
    }
    return M;
}

// Read an ivecs file as (N, K) int matrix. Used for cluster_ids (K=1 per row) and
// groundtruth (K = number of true neighbors per query, typically 100).
std::vector<std::vector<int32_t>> read_ivecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::fprintf(stderr, "open failed: %s\n", path.c_str()); std::exit(1); }
    std::vector<std::vector<int32_t>> out;
    int32_t k = 0;
    while (in.read(reinterpret_cast<char*>(&k), 4)) {
        std::vector<int32_t> row(static_cast<size_t>(k));
        in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(k) * 4);
        out.push_back(std::move(row));
    }
    return out;
}

// Average Recall@k across queries: |found_topk ∩ true_topk| / k.
double recall_at_k(
    const std::vector<std::vector<int32_t>>& gt,
    const std::vector<std::vector<int32_t>>& found,
    size_t k)
{
    if (gt.size() != found.size() || gt.empty()) return 0.0;
    double sum = 0.0;
    for (size_t q = 0; q < gt.size(); ++q) {
        std::unordered_set<int32_t> truth(gt[q].begin(),
                                          gt[q].begin() + std::min(k, gt[q].size()));
        size_t hits = 0;
        for (size_t i = 0; i < std::min(k, found[q].size()); ++i)
            if (truth.count(found[q][i])) ++hits;
        sum += static_cast<double>(hits) / static_cast<double>(k);
    }
    return sum / static_cast<double>(gt.size());
}

}  // namespace

int main() {
    const std::string dir = "data/datasets/dbpedia_100k";
    std::fprintf(stderr, "loading inputs from %s/...\n", dir.c_str());
    auto data         = read_fvecs(dir + "/vectors_pca.fvecs");
    auto queries      = read_fvecs(dir + "/queries_pca.fvecs");
    auto centroids    = read_fvecs(dir + "/centroids_4096_pca.fvecs");
    auto cluster_rows = read_ivecs(dir + "/cluster_ids_4096.ivecs");
    auto gt_rows      = read_ivecs(dir + "/groundtruth.ivecs");

    // Flatten cluster_rows (each row has 1 element) into a contiguous PID array.
    std::vector<saq::PID> cluster_ids;
    cluster_ids.reserve(cluster_rows.size());
    for (const auto& row : cluster_rows) cluster_ids.push_back(static_cast<saq::PID>(row[0]));

    const size_t N = static_cast<size_t>(data.rows());
    const size_t D = static_cast<size_t>(data.cols());
    const size_t K = static_cast<size_t>(centroids.rows());
    const size_t k_top = 10;
    const size_t nprobe = 200;

    std::fprintf(stderr,
                 "data=%lldx%lld  queries=%lldx%lld  centroids=%lldx%lld  "
                 "cluster_ids=%zu  gt_rows=%zu  peak_rss=%ld KB\n",
                 (long long)data.rows(), (long long)data.cols(),
                 (long long)queries.rows(), (long long)queries.cols(),
                 (long long)centroids.rows(), (long long)centroids.cols(),
                 cluster_ids.size(), gt_rows.size(), peak_rss_kb());

    // --- IVF #1: kpp+Lloyd codebooks ---
    saq::QuantizeConfig cfg_lloyd;
    cfg_lloyd.avg_bits = 4;  // headline regime; segment bits stay <= 8 so the DP run will work too.

    std::fprintf(stderr, "[lloyd] constructing IVF with set_derive_codebooks(KMeansPlusPlus)...\n");
    saq::IVF ivf_lloyd(N, D, K, cfg_lloyd);
    saq::LloydOpts opts;
    opts.init = saq::CodebookInit::KMeansPlusPlus;
    opts.restarts = 1;
    opts.seed = 0;
    opts.max_bits = 13;
    ivf_lloyd.set_derive_codebooks(opts);
    double t0 = now_s();
    ivf_lloyd.construct(data, centroids, cluster_ids.data());
    double t1 = now_s();
    std::fprintf(stderr, "[lloyd] construct: %.2fs, peak_rss=%ld KB\n", t1 - t0, peak_rss_kb());

    // search() writes into a PID output array (no return value); nprobe is a
    // separate parameter (not part of SearcherConfig).
    saq::SearcherConfig scfg;
    scfg.dist_type = saq::DistType::L2Sqr;
    std::vector<std::vector<int32_t>> found_lloyd(static_cast<size_t>(queries.rows()));
    t0 = now_s();
    {
        std::vector<saq::PID> result_buf(k_top);
        for (Eigen::Index q = 0; q < queries.rows(); ++q) {
            ivf_lloyd.search(queries.row(q), k_top, nprobe, scfg, result_buf.data());
            auto& row = found_lloyd[static_cast<size_t>(q)];
            row.resize(k_top);
            for (size_t i = 0; i < k_top; ++i)
                row[i] = static_cast<int32_t>(result_buf[i]);
        }
    }
    t1 = now_s();
    double recall_lloyd = recall_at_k(gt_rows, found_lloyd, k_top);
    std::fprintf(stderr, "[lloyd] search %zu queries: %.2fs total, recall@%zu=%.4f\n",
                 static_cast<size_t>(queries.rows()), t1 - t0, k_top, recall_lloyd);

    // Capture the quant_plan from IVF #1 — we'll use the same bit allocation for the DP run
    // so the comparison is apples-to-apples (same segment structure, different codebook).
    auto quant_plan = ivf_lloyd.get_saq_data()->quant_plan;
    std::fprintf(stderr, "[lloyd] quant_plan: %zu segments\n", quant_plan.size());
    for (size_t s = 0; s < quant_plan.size(); ++s) {
        std::fprintf(stderr, "  seg %zu: dim_len=%zu bits=%zu\n",
                     s, quant_plan[s].first, quant_plan[s].second);
    }
    // Sanity: every segment must have bits <= 8 so we can run build_codebook_dp later.
    for (size_t s = 0; s < quant_plan.size(); ++s) {
        if (quant_plan[s].second > 8) {
            std::fprintf(stderr, "ERROR seg %zu has bits=%zu > 8; build_codebook_dp won't run.\n"
                                 "Pick a smaller cfg.avg_bits and rerun.\n",
                         s, quant_plan[s].second);
            return 2;
        }
    }

    // --- IVF #2: DP-derived codebooks, injected via set_codebooks ---
    // Build per-segment, per-dim DimensionCodebooks by running build_codebook_dp
    // on each global dim at its segment's bit count. quant_plan is what IVF #1
    // computed; we honor it to keep segment structure identical.

    std::fprintf(stderr, "[dp] deriving DP codebooks per dim (parallel over dims; not part of recall timing)...\n");
    std::vector<std::vector<saq::DimensionCodebook>> dp_codebooks(quant_plan.size());
    for (size_t s = 0; s < quant_plan.size(); ++s) {
        dp_codebooks[s].resize(quant_plan[s].first);
    }

    // Precompute the per-segment global-dim start indices, so each thread
    // can independently figure out which segment its dim belongs to.
    std::vector<size_t> seg_start(quant_plan.size() + 1, 0);
    for (size_t s = 0; s < quant_plan.size(); ++s) {
        seg_start[s + 1] = seg_start[s] + quant_plan[s].first;
    }
    const size_t total_dims = seg_start.back();

    double t_dp_start = now_s();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (long long gdim_ll = 0; gdim_ll < static_cast<long long>(total_dims); ++gdim_ll) {
        size_t gdim = static_cast<size_t>(gdim_ll);
        // Find which segment this gdim belongs to.
        size_t seg = 0;
        while (seg + 1 < seg_start.size() && seg_start[seg + 1] <= gdim) ++seg;
        const size_t bits = quant_plan[seg].second;
        const size_t j    = gdim - seg_start[seg];
        std::vector<float> col(static_cast<size_t>(data.rows()));
        for (Eigen::Index i = 0; i < data.rows(); ++i)
            col[static_cast<size_t>(i)] = data(i, static_cast<Eigen::Index>(gdim));
        auto r = saq::build_codebook_dp(col, bits, /*num_bins=*/500);
        dp_codebooks[seg][j] = r.codebooks[bits];
    }
    std::fprintf(stderr, "[dp] per-dim DP build done in %.1fs (total_dims=%zu)\n",
                 now_s() - t_dp_start, total_dims);

    std::fprintf(stderr, "[dp] constructing IVF with set_codebooks(DP-derived)...\n");
    saq::QuantizeConfig cfg_dp = cfg_lloyd;
    saq::IVF ivf_dp(N, D, K, cfg_dp);
    ivf_dp.set_codebooks(std::move(dp_codebooks));
    t0 = now_s();
    ivf_dp.construct(data, centroids, cluster_ids.data());
    t1 = now_s();
    std::fprintf(stderr, "[dp] construct: %.2fs\n", t1 - t0);

    std::vector<std::vector<int32_t>> found_dp(queries.rows());
    t0 = now_s();
    {
        std::vector<saq::PID> result_buf(k_top);
        for (Eigen::Index q = 0; q < queries.rows(); ++q) {
            ivf_dp.search(queries.row(q), k_top, nprobe, scfg, result_buf.data());
            auto& row = found_dp[static_cast<size_t>(q)];
            row.resize(k_top);
            for (size_t i = 0; i < k_top; ++i)
                row[i] = static_cast<int32_t>(result_buf[i]);
        }
    }
    t1 = now_s();
    double recall_dp = recall_at_k(gt_rows, found_dp, k_top);
    std::fprintf(stderr, "[dp] search %zu queries: %.2fs total, recall@%zu=%.4f\n",
                 static_cast<size_t>(queries.rows()), t1 - t0, k_top, recall_dp);

    double delta_pp = (recall_dp - recall_lloyd) * 100.0;  // in percentage points
    std::printf("{\n"
                "  \"k_top\": %zu,\n"
                "  \"nprobe\": %zu,\n"
                "  \"avg_bits\": %.2f,\n"
                "  \"recall_lloyd\": %.6f,\n"
                "  \"recall_dp\":    %.6f,\n"
                "  \"delta_pp\":     %.4f\n"
                "}\n", k_top, nprobe, cfg_lloyd.avg_bits, recall_lloyd, recall_dp, delta_pp);
    return 0;
}
