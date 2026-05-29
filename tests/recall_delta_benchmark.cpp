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

    std::fprintf(stderr,
                 "data=%lldx%lld  queries=%lldx%lld  centroids=%lldx%lld  "
                 "cluster_rows=%zu  gt_rows=%zu  peak_rss=%ld KB\n",
                 (long long)data.rows(), (long long)data.cols(),
                 (long long)queries.rows(), (long long)queries.cols(),
                 (long long)centroids.rows(), (long long)centroids.cols(),
                 cluster_rows.size(), gt_rows.size(), peak_rss_kb());

    std::printf("{\"status\":\"loaded\",\"n_data\":%lld,\"n_queries\":%lld,"
                "\"d\":%lld,\"k_clusters\":%lld}\n",
                (long long)data.rows(), (long long)queries.rows(),
                (long long)data.cols(), (long long)centroids.rows());
    return 0;
}
