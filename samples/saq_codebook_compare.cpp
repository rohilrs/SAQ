/// @file saq_codebook_compare.cpp
/// @brief A/B comparison: baseline (variance cost) vs optimal cost model.
///
/// Builds two IVF indexes with different DP cost models and compares
/// recall, quantization plans, and cosine distortion.
///
/// Usage: saq_codebook_compare [data_dir] [bpd] [num_clusters] [nprobe] [num_threads]

#include "index/ivf_index.h"
#include "saq/config.h"
#include "saq/defines.h"
#include "saq/io_utils.h"
#include "saq/stopw.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef SAQ_USE_OPENMP
#include <omp.h>
#endif

using namespace saq;

// ============================================================================
// Metrics
// ============================================================================

static float ComputeRecallAtK(const std::vector<std::vector<PID>>& results,
                               const UintRowMat& gt, size_t k) {
    size_t nq = results.size();
    size_t total_correct = 0;
    size_t total_count = 0;

    for (size_t q = 0; q < nq; ++q) {
        size_t gt_k = std::min(k, static_cast<size_t>(gt.cols()));
        size_t res_k = std::min(k, results[q].size());
        std::unordered_set<PID> gt_set;
        for (size_t i = 0; i < gt_k; ++i) gt_set.insert(gt(q, i));
        for (size_t i = 0; i < res_k; ++i) {
            if (gt_set.count(results[q][i])) total_correct++;
        }
        total_count += gt_k;
    }
    return total_count > 0
        ? static_cast<float>(total_correct) / static_cast<float>(total_count)
        : 0.0f;
}

/// Compute mean cosine similarity between queries and their search results.
static float MeanCosineSim(const FloatRowMat& queries,
                            const FloatRowMat& data,
                            const std::vector<std::vector<PID>>& results,
                            size_t k) {
    double total_sim = 0.0;
    size_t count = 0;
    size_t nq = results.size();

    for (size_t q = 0; q < nq; ++q) {
        size_t res_k = std::min(k, results[q].size());
        for (size_t i = 0; i < res_k; ++i) {
            PID id = results[q][i];
            if (id >= static_cast<PID>(data.rows())) continue;
            float dot = queries.row(q).dot(data.row(id));
            float nq_ = queries.row(q).norm();
            float nd = data.row(id).norm();
            if (nq_ > 1e-12f && nd > 1e-12f) {
                total_sim += dot / (nq_ * nd);
                count++;
            }
        }
    }
    return count > 0 ? static_cast<float>(total_sim / count) : 0.0f;
}

static void PrintPlan(const SaqData* saq_data, const char* label) {
    if (!saq_data) return;
    std::cout << "  " << label << ": ";
    size_t dims_sum = 0;
    for (auto& [dim_len, bits] : saq_data->quant_plan) {
        std::cout << "[" << dims_sum << ".." << (dims_sum + dim_len) << ")@" << bits << "b ";
        dims_sum += dim_len;
    }
    std::cout << "\n";
}

struct RunResult {
    std::vector<std::vector<PID>> results;
    float build_time_s;
    const SaqData* plan;
};

static RunResult BuildAndSearch(
    const FloatRowMat& data, const FloatRowMat& queries,
    const FloatRowMat& centroids, const UintRowMat& cluster_ids,
    const FloatVec& variances, float bpd, size_t K, size_t nprobe,
    int num_threads, SearcherConfig searcher_cfg,
    const FloatRowMat* optimal_costs)
{
    QuantizeConfig cfg;
    cfg.avg_bits = bpd;
    cfg.single.quant_type = BaseQuantType::CAQ;
    cfg.single.random_rotation = true;
    cfg.single.use_fastscan = true;
    cfg.single.caq_adj_rd_lmt = 6;
    cfg.enable_segmentation = true;

    size_t num_vecs = static_cast<size_t>(data.rows());
    size_t num_dim = static_cast<size_t>(data.cols());
    size_t nq = static_cast<size_t>(queries.rows());

    IVF ivf(num_vecs, num_dim, K, cfg);
    ivf.set_variance(FloatVec(variances));

    if (optimal_costs) {
        ivf.set_optimal_costs(FloatRowMat(*optimal_costs));
    }

    StopW timer;
    ivf.construct(data, centroids, cluster_ids.data(), num_threads);
    float build_time = timer.getElapsedTimeMili() / 1000.0f;

    const SaqData* plan = ivf.get_saq_data();

    constexpr size_t TOPK = 100;
    std::vector<std::vector<PID>> results(nq, std::vector<PID>(TOPK));

    for (size_t q = 0; q < nq; ++q) {
        ivf.search<DistType::L2Sqr>(
            queries.row(q), TOPK, nprobe, searcher_cfg, results[q].data());
    }

    return {std::move(results), build_time, plan};
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    std::string data_dir = "data/datasets/dbpedia_100k";
    float bpd = 2.0f;
    size_t K = 4096;
    size_t nprobe = 200;
    int num_threads = 8;

    if (argc > 1) data_dir = argv[1];
    if (argc > 2) bpd = std::stof(argv[2]);
    if (argc > 3) K = std::stoul(argv[3]);
    if (argc > 4) nprobe = std::stoul(argv[4]);
    if (argc > 5) num_threads = std::stoi(argv[5]);

    std::cout << "================================================================\n";
    std::cout << "SAQ Cost Model A/B Comparison\n";
    std::cout << "================================================================\n";
    std::cout << "  data_dir:    " << data_dir << "\n";
    std::cout << "  bpd:         " << std::fixed << std::setprecision(2) << bpd << "\n";
    std::cout << "  K:           " << K << "\n";
    std::cout << "  nprobe:      " << nprobe << "\n";
    std::cout << "  threads:     " << num_threads << "\n\n";

    // Load data
    std::string k_str = std::to_string(K);
    FloatRowMat data, queries, centroids;
    UintRowMat cluster_ids, gt;

    load_something<float, FloatRowMat>((data_dir + "/vectors_pca.fvecs").c_str(), data);
    load_something<float, FloatRowMat>((data_dir + "/queries_pca.fvecs").c_str(), queries);
    load_something<float, FloatRowMat>((data_dir + "/centroids_" + k_str + "_pca.fvecs").c_str(), centroids);
    load_something<uint32_t, UintRowMat>((data_dir + "/cluster_ids_" + k_str + ".ivecs").c_str(), cluster_ids);
    load_something<uint32_t, UintRowMat>((data_dir + "/groundtruth.ivecs").c_str(), gt);

    FloatRowMat var_mat;
    load_something<float, FloatRowMat>((data_dir + "/variances_pca.fvecs").c_str(), var_mat);
    FloatVec variances = var_mat.row(0);

    // Load optimal costs (optional)
    std::string costs_file = data_dir + "/optimal_costs.fvecs";
    FloatRowMat optimal_costs;
    bool have_costs = file_exists(costs_file.c_str());
    if (have_costs) {
        load_something<float, FloatRowMat>(costs_file.c_str(), optimal_costs);
        std::cout << "Loaded optimal_costs: " << optimal_costs.rows()
                  << " x " << optimal_costs.cols() << "\n\n";
    } else {
        std::cerr << "ERROR: " << costs_file << " not found.\n"
                  << "Run: python -m preprocessing.compute_costs --data-dir " << data_dir << "\n";
        return 1;
    }

    size_t nq = static_cast<size_t>(queries.rows());

    std::cout << "Data: " << data.rows() << " x " << data.cols()
              << ", Queries: " << nq << "\n\n";

    SearcherConfig searcher_cfg;
    searcher_cfg.dist_type = DistType::L2Sqr;

    // ===== Run A: Baseline =====
    std::cout << "[A] Building baseline (variance cost model)...\n";
    auto runA = BuildAndSearch(data, queries, centroids, cluster_ids,
                               variances, bpd, K, nprobe, num_threads,
                               searcher_cfg, nullptr);
    std::cout << "  Build time: " << std::fixed << std::setprecision(2)
              << runA.build_time_s << "s\n";
    PrintPlan(runA.plan, "Plan A");

    // ===== Run B: Optimal costs =====
    std::cout << "\n[B] Building with optimal cost model...\n";
    auto runB = BuildAndSearch(data, queries, centroids, cluster_ids,
                               variances, bpd, K, nprobe, num_threads,
                               searcher_cfg, &optimal_costs);
    std::cout << "  Build time: " << std::fixed << std::setprecision(2)
              << runB.build_time_s << "s\n";
    PrintPlan(runB.plan, "Plan B");

    // ===== Recall comparison =====
    std::cout << "\n================================================================\n";
    std::cout << "RECALL COMPARISON (nprobe=" << nprobe << ")\n";
    std::cout << "================================================================\n";

    std::cout << std::setw(10) << "Metric"
              << std::setw(14) << "A (baseline)"
              << std::setw(14) << "B (optimal)"
              << std::setw(10) << "Delta"
              << "\n";
    std::cout << std::string(48, '-') << "\n";

    auto print_row = [](const char* label, float a, float b) {
        float delta = (b - a) * 100;
        std::cout << std::setw(10) << label
                  << std::setw(13) << std::fixed << std::setprecision(2) << (a * 100) << "%"
                  << std::setw(13) << std::fixed << std::setprecision(2) << (b * 100) << "%"
                  << std::setw(9) << std::showpos << std::fixed << std::setprecision(2)
                  << delta << "%" << std::noshowpos
                  << "\n";
    };

    float r1_a  = ComputeRecallAtK(runA.results, gt, 1);
    float r1_b  = ComputeRecallAtK(runB.results, gt, 1);
    float r10_a = ComputeRecallAtK(runA.results, gt, 10);
    float r10_b = ComputeRecallAtK(runB.results, gt, 10);
    float r100_a = ComputeRecallAtK(runA.results, gt, 100);
    float r100_b = ComputeRecallAtK(runB.results, gt, 100);

    print_row("Recall@1", r1_a, r1_b);
    print_row("Recall@10", r10_a, r10_b);
    print_row("Recall@100", r100_a, r100_b);

    // ===== Cosine distortion =====
    std::cout << "\n================================================================\n";
    std::cout << "COSINE SIMILARITY OF RESULTS (higher = better)\n";
    std::cout << "================================================================\n";

    // Ground truth cosine: mean cosine sim of query to its true top-K
    for (size_t topk : {1ul, 10ul, 100ul}) {
        // Build GT results for cosine comparison
        std::vector<std::vector<PID>> gt_results(nq);
        for (size_t q = 0; q < nq; ++q) {
            size_t gt_k = std::min(topk, static_cast<size_t>(gt.cols()));
            gt_results[q].resize(gt_k);
            for (size_t i = 0; i < gt_k; ++i) gt_results[q][i] = gt(q, i);
        }

        float cos_gt = MeanCosineSim(queries, data, gt_results, topk);
        float cos_a  = MeanCosineSim(queries, data, runA.results, topk);
        float cos_b  = MeanCosineSim(queries, data, runB.results, topk);

        std::cout << "  Top-" << std::setw(3) << topk << ": "
                  << "GT=" << std::fixed << std::setprecision(6) << cos_gt
                  << "  A=" << cos_a
                  << "  B=" << cos_b
                  << "  (B-A=" << std::showpos << (cos_b - cos_a) << std::noshowpos << ")"
                  << "\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "Done.\n";
    std::cout << "================================================================\n";

    return 0;
}
