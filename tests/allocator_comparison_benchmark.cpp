/// @file allocator_comparison_benchmark.cpp
/// @brief P4-T7/T8 — compare DP vs Greedy allocators on dbpedia-100K.
///
/// 4-cell experiment: {DP, Greedy} × {2.0, 4.0} avg_bits.
/// MSE table is built once per avg_bits level and shared between DP and Greedy
/// cells so that the measurement is on a consistent empirical basis.

#include "index/ivf_index.h"
#include "saq/bit_allocator.h"
#include "saq/bit_allocator_greedy.h"
#include "saq/preprocessing/codebook_builder.h"
#include "saq/quantization_plan.h"

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

    const size_t N      = static_cast<size_t>(data.rows());
    const size_t D      = static_cast<size_t>(data.cols());
    const size_t K      = static_cast<size_t>(centroids.rows());
    const size_t k_top  = 10;
    const size_t nprobe = 200;

    std::fprintf(stderr,
                 "data=%lldx%lld  queries=%lldx%lld  centroids=%lldx%lld  "
                 "cluster_ids=%zu  gt_rows=%zu  peak_rss=%ld KB\n",
                 (long long)data.rows(), (long long)data.cols(),
                 (long long)queries.rows(), (long long)queries.cols(),
                 (long long)centroids.rows(), (long long)centroids.cols(),
                 cluster_ids.size(), gt_rows.size(), peak_rss_kb());

    // eval_cell: construct one IVF with the given allocator+avg_bits, search,
    // and emit a JSON object to stdout.
    // mse_table: precomputed empirical MSE matrix (D_padded x max_bits+1) shared
    // between DP and Greedy cells at the same avg_bits for consistent measurement.
    auto eval_cell = [&](const std::string& label,
                         saq::AllocatorKind allocator, float avg_bits,
                         const Eigen::MatrixXf& mse_table) {
        const char* alloc_str = (allocator == saq::AllocatorKind::DP) ? "DP" : "Greedy";
        std::fprintf(stderr, "\n[%s] allocator=%s avg_bits=%.1f -> constructing IVF...\n",
                     label.c_str(), alloc_str, avg_bits);

        saq::QuantizeConfig cfg;
        cfg.avg_bits  = avg_bits;
        cfg.allocator = allocator;

        saq::IVF ivf(N, D, K, cfg);
        saq::LloydOpts opts;
        opts.init     = saq::CodebookInit::KMeansPlusPlus;
        opts.restarts = 1;
        opts.seed     = 0;
        opts.max_bits = 13;
        ivf.set_derive_codebooks(opts);

        double t0 = now_s();
        ivf.construct(data, centroids, cluster_ids.data());
        double t_construct = now_s() - t0;
        std::fprintf(stderr, "  [%s] construct done in %.1fs  peak_rss=%ld KB\n",
                     label.c_str(), t_construct, peak_rss_kb());

        // Capture quant_plan and compute total MSE on the shared empirical table.
        auto quant_plan = ivf.get_saq_data()->quant_plan;
        double total_mse = 0.0;
        {
            Eigen::Index cum_dim = 0;
            for (const auto& seg : quant_plan) {
                const Eigen::Index seg_len  = static_cast<Eigen::Index>(seg.first);
                const Eigen::Index seg_bits = static_cast<Eigen::Index>(seg.second);
                for (Eigen::Index k = 0; k < seg_len; ++k) {
                    total_mse += static_cast<double>(mse_table(cum_dim + k, seg_bits));
                }
                cum_dim += seg_len;
            }
        }

        // Search the queries.
        saq::SearcherConfig scfg;
        scfg.dist_type = saq::DistType::L2Sqr;
        std::vector<std::vector<int32_t>> found(static_cast<size_t>(queries.rows()));
        {
            std::vector<saq::PID> result_buf(k_top);
            t0 = now_s();
            for (Eigen::Index q = 0; q < queries.rows(); ++q) {
                ivf.search(queries.row(q), k_top, nprobe, scfg, result_buf.data());
                auto& row = found[static_cast<size_t>(q)];
                row.reserve(k_top);
                for (size_t i = 0; i < k_top; ++i)
                    row.push_back(static_cast<int32_t>(result_buf[i]));
            }
        }
        double t_search = now_s() - t0;
        double recall   = recall_at_k(gt_rows, found, k_top);

        std::fprintf(stderr,
                     "  [%s] construct=%.1fs search=%.1fs total_mse=%.4e recall@%zu=%.4f\n",
                     label.c_str(), t_construct, t_search, total_mse, k_top, recall);

        // Emit JSON cell to stdout.
        std::printf("  {\"label\":\"%s\",\"allocator\":\"%s\",\"avg_bits\":%.2f,"
                    "\"construct_seconds\":%.2f,\"search_seconds\":%.2f,"
                    "\"total_codebook_mse\":%.6e,\"recall_at_10\":%.6f,"
                    "\"quant_plan\":[",
                    label.c_str(), alloc_str,
                    avg_bits, t_construct, t_search, total_mse, recall);
        for (size_t i = 0; i < quant_plan.size(); ++i) {
            std::printf("%s[%zu,%zu]",
                        i ? "," : "",
                        quant_plan[i].first, quant_plan[i].second);
        }
        std::printf("]}");
        std::fflush(stdout);
    };

    std::printf("{\n \"cells\": [");
    bool first = true;

    for (float avg_bits : {2.0f, 4.0f}) {
        // Build the MSE table once per avg_bits — both DP and Greedy cells use it
        // for measurement on a consistent empirical basis.
        std::fprintf(stderr,
                     "\n=== avg_bits=%.1f: building MSE table (max_bits=%zu)... ===\n",
                     avg_bits, saq::KMaxQuantizeBits);
        double t_mse0 = now_s();
        auto mse_table = saq::build_mse_table_for_allocation(data, saq::KMaxQuantizeBits);
        std::fprintf(stderr, "  MSE table built in %.1fs  shape=%lldx%lld\n",
                     now_s() - t_mse0,
                     (long long)mse_table.rows(), (long long)mse_table.cols());

        for (saq::AllocatorKind allocator : {saq::AllocatorKind::DP,
                                              saq::AllocatorKind::Greedy}) {
            const std::string label =
                std::string(allocator == saq::AllocatorKind::DP ? "dp" : "gd") + "_" +
                (avg_bits < 3.0f ? "2bpd" : "4bpd");
            if (!first) std::printf(",");
            std::printf("\n");
            eval_cell(label, allocator, avg_bits, mse_table);
            first = false;
        }
    }

    std::printf("\n ]\n}\n");
    std::fflush(stdout);
    std::fprintf(stderr, "\nDone. peak_rss=%ld KB\n", peak_rss_kb());
    return 0;
}
