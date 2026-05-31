/// @file allocator_comparison_benchmark.cpp
/// @brief P4-T7/T8 — compare DP vs Greedy allocators on dbpedia-100K.
///
/// T7 scaffold: loads dbpedia inputs and confirms dimensions.
/// T8 fills in the 4-cell loop (DP/Greedy × {2 bpd, 4 bpd}).

#include "index/ivf_index.h"
#include "saq/bit_allocator.h"
#include "saq/bit_allocator_greedy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
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

    std::vector<saq::PID> cluster_ids;
    cluster_ids.reserve(cluster_rows.size());
    for (const auto& row : cluster_rows) cluster_ids.push_back(static_cast<saq::PID>(row[0]));

    std::fprintf(stderr,
                 "data=%lldx%lld  queries=%lldx%lld  centroids=%lldx%lld  "
                 "cluster_ids=%zu  gt_rows=%zu  peak_rss=%ld KB\n",
                 (long long)data.rows(), (long long)data.cols(),
                 (long long)queries.rows(), (long long)queries.cols(),
                 (long long)centroids.rows(), (long long)centroids.cols(),
                 cluster_ids.size(), gt_rows.size(), peak_rss_kb());

    std::printf("{\"status\":\"loaded\",\"n_data\":%lld,\"n_queries\":%lld,"
                "\"d\":%lld,\"k_clusters\":%lld}\n",
                (long long)data.rows(), (long long)queries.rows(),
                (long long)data.cols(), (long long)centroids.rows());
    return 0;
}
