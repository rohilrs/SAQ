/// @file saq_full_codebook_compare.cpp
/// @brief A/B comparison: uniform (baseline) vs optimal codebook encoding.
///
/// Runs at BPD 1-8 (or specified range) with timing, recall, and cosine sim.
///
/// Usage: saq_full_codebook_compare [data_dir] [K] [nprobe] [threads] [min_bpd] [max_bpd]

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

static float ComputeRecallAtK(const std::vector<std::vector<PID>>& results,
                               const UintRowMat& gt, size_t k) {
    size_t nq = results.size();
    size_t total_correct = 0, total_count = 0;
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
    return total_count > 0 ? static_cast<float>(total_correct) / total_count : 0.0f;
}

static float MeanCosineSim(const FloatRowMat& queries, const FloatRowMat& data,
                            const std::vector<std::vector<PID>>& results, size_t k) {
    double total_sim = 0.0;
    size_t count = 0;
    for (size_t q = 0; q < results.size(); ++q) {
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

struct BenchResult {
    float bpd;
    std::string label;
    float build_time_s;
    float search_time_ms;
    float recall_1, recall_10, recall_100;
    float cosine_10;
    std::string plan_str;
};

static BenchResult RunBenchmark(
    const FloatRowMat& data, const FloatRowMat& queries, const FloatRowMat& centroids,
    const UintRowMat& cluster_ids, const UintRowMat& gt,
    const FloatVec& variances, float bpd, size_t K, size_t nprobe, int threads,
    const char* label, const FloatRowMat* codebooks)
{
    QuantizeConfig cfg;
    cfg.avg_bits = bpd;
    cfg.single.quant_type = BaseQuantType::CAQ;
    // Disable rotation for both runs — codebooks are in PCA space
    cfg.single.random_rotation = false;
    cfg.single.use_fastscan = true;
    cfg.single.caq_adj_rd_lmt = (codebooks == nullptr) ? 6 : 0;  // No CAQ adjustment for codebook
    cfg.enable_segmentation = true;

    size_t nv = static_cast<size_t>(data.rows());
    size_t nd = static_cast<size_t>(data.cols());
    size_t nq = static_cast<size_t>(queries.rows());
    constexpr size_t TOPK = 100;

    IVF ivf(nv, nd, K, cfg);
    ivf.set_variance(FloatVec(variances));
    if (codebooks) ivf.set_codebooks(FloatRowMat(*codebooks));

    StopW build_timer;
    ivf.construct(data, centroids, cluster_ids.data(), threads);
    float build_s = build_timer.getElapsedTimeMili() / 1000.0f;

    // Get plan string
    std::string plan;
    if (auto* sd = ivf.get_saq_data()) {
        size_t d = 0;
        for (auto& [dl, b] : sd->quant_plan) {
            plan += "[" + std::to_string(d) + ".." + std::to_string(d+dl) + ")@" + std::to_string(b) + "b ";
            d += dl;
        }
    }

    SearcherConfig scfg;
    scfg.dist_type = DistType::L2Sqr;

    std::vector<std::vector<PID>> results(nq, std::vector<PID>(TOPK));
    StopW search_timer;
    for (size_t q = 0; q < nq; ++q) {
        ivf.search<DistType::L2Sqr>(queries.row(q), TOPK, nprobe, scfg, results[q].data());
    }
    float search_ms = search_timer.getElapsedTimeMili();

    BenchResult r;
    r.bpd = bpd;
    r.label = label;
    r.build_time_s = build_s;
    r.search_time_ms = search_ms;
    r.recall_1 = ComputeRecallAtK(results, gt, 1);
    r.recall_10 = ComputeRecallAtK(results, gt, 10);
    r.recall_100 = ComputeRecallAtK(results, gt, 100);
    r.cosine_10 = MeanCosineSim(queries, data, results, 10);
    r.plan_str = plan;
    return r;
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    std::string data_dir = "data/datasets/dbpedia_100k";
    size_t K = 4096, nprobe = 200;
    int threads = 8;
    int min_bpd = 1, max_bpd = 6;

    if (argc > 1) data_dir = argv[1];
    if (argc > 2) K = std::stoul(argv[2]);
    if (argc > 3) nprobe = std::stoul(argv[3]);
    if (argc > 4) threads = std::stoi(argv[4]);
    if (argc > 5) min_bpd = std::stoi(argv[5]);
    if (argc > 6) max_bpd = std::stoi(argv[6]);

    std::cout << "================================================================\n";
    std::cout << "SAQ Full Codebook A/B Comparison (BPD " << min_bpd << "-" << max_bpd << ")\n";
    std::cout << "================================================================\n";
    std::cout << "  K=" << K << " nprobe=" << nprobe << " threads=" << threads << "\n\n";

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

    // Load codebooks
    std::string cb_file = data_dir + "/optimal_codebooks.fvecs";
    FloatRowMat codebooks;
    if (file_exists(cb_file.c_str())) {
        load_something<float, FloatRowMat>(cb_file.c_str(), codebooks);
        std::cout << "Codebooks: " << codebooks.rows() << " x " << codebooks.cols() << "\n\n";
    } else {
        std::cerr << "ERROR: " << cb_file << " not found.\n"
                  << "Run: python -m preprocessing.compute_codebooks --data-dir " << data_dir << "\n";
        return 1;
    }

    std::cout << "Data: " << data.rows() << "x" << data.cols()
              << ", Queries: " << queries.rows() << "\n\n";

    // Run benchmarks
    std::vector<BenchResult> results_a, results_b;

    for (int bpd = min_bpd; bpd <= max_bpd; bpd++) {
        float bpd_f = static_cast<float>(bpd);
        std::cout << "--- BPD " << bpd << " ---\n";

        std::cout << "  [A] Baseline (uniform)...\n";
        auto ra = RunBenchmark(data, queries, centroids, cluster_ids, gt,
                                variances, bpd_f, K, nprobe, threads, "uniform", nullptr);
        results_a.push_back(ra);
        std::cout << "    Build: " << std::fixed << std::setprecision(2) << ra.build_time_s << "s"
                  << "  Search: " << std::setprecision(1) << ra.search_time_ms << "ms"
                  << "  R@10=" << std::setprecision(2) << (ra.recall_10 * 100) << "%\n";

        std::cout << "  [B] Codebook...\n";
        auto rb = RunBenchmark(data, queries, centroids, cluster_ids, gt,
                                variances, bpd_f, K, nprobe, threads, "codebook", &codebooks);
        results_b.push_back(rb);
        std::cout << "    Build: " << std::fixed << std::setprecision(2) << rb.build_time_s << "s"
                  << "  Search: " << std::setprecision(1) << rb.search_time_ms << "ms"
                  << "  R@10=" << std::setprecision(2) << (rb.recall_10 * 100) << "%\n\n";
    }

    // Summary table
    std::cout << "\n================================================================\n";
    std::cout << "RESULTS SUMMARY (nprobe=" << nprobe << ")\n";
    std::cout << "================================================================\n\n";

    std::cout << std::setw(4) << "BPD"
              << " | " << std::setw(9) << "R@1 (A)"
              << std::setw(9) << "R@1 (B)"
              << " | " << std::setw(10) << "R@10 (A)"
              << std::setw(10) << "R@10 (B)"
              << " | " << std::setw(11) << "R@100 (A)"
              << std::setw(11) << "R@100 (B)"
              << " | " << std::setw(10) << "Build A"
              << std::setw(10) << "Build B"
              << " | " << std::setw(10) << "Srch A"
              << std::setw(10) << "Srch B"
              << "\n";
    std::cout << std::string(130, '-') << "\n";

    for (size_t i = 0; i < results_a.size(); i++) {
        auto& a = results_a[i];
        auto& b = results_b[i];
        std::cout << std::setw(4) << static_cast<int>(a.bpd)
                  << " | " << std::setw(8) << std::fixed << std::setprecision(2) << (a.recall_1*100) << "%"
                  << std::setw(8) << (b.recall_1*100) << "%"
                  << " | " << std::setw(9) << (a.recall_10*100) << "%"
                  << std::setw(9) << (b.recall_10*100) << "%"
                  << " | " << std::setw(10) << (a.recall_100*100) << "%"
                  << std::setw(10) << (b.recall_100*100) << "%"
                  << " | " << std::setw(8) << std::setprecision(1) << a.build_time_s << "s"
                  << std::setw(8) << b.build_time_s << "s"
                  << " | " << std::setw(8) << std::setprecision(0) << a.search_time_ms << "ms"
                  << std::setw(8) << b.search_time_ms << "ms"
                  << "\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "COSINE SIMILARITY (top-10, nprobe=" << nprobe << ")\n";
    std::cout << "================================================================\n";

    // Ground truth cosine
    size_t nq = static_cast<size_t>(queries.rows());
    std::vector<std::vector<PID>> gt_results(nq);
    for (size_t q = 0; q < nq; ++q) {
        size_t gt_k = std::min(static_cast<size_t>(10), static_cast<size_t>(gt.cols()));
        gt_results[q].resize(gt_k);
        for (size_t i = 0; i < gt_k; ++i) gt_results[q][i] = gt(q, i);
    }
    float cos_gt = MeanCosineSim(queries, data, gt_results, 10);

    std::cout << "  Ground truth cosine@10: " << std::fixed << std::setprecision(6) << cos_gt << "\n\n";

    std::cout << std::setw(4) << "BPD"
              << std::setw(14) << "Cos@10 (A)"
              << std::setw(14) << "Cos@10 (B)"
              << std::setw(14) << "B-A"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (size_t i = 0; i < results_a.size(); i++) {
        auto& a = results_a[i];
        auto& b = results_b[i];
        std::cout << std::setw(4) << static_cast<int>(a.bpd)
                  << std::setw(14) << std::fixed << std::setprecision(6) << a.cosine_10
                  << std::setw(14) << b.cosine_10
                  << std::setw(13) << std::showpos << (b.cosine_10 - a.cosine_10) << std::noshowpos
                  << "\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "QUANTIZATION PLANS\n";
    std::cout << "================================================================\n";

    for (size_t i = 0; i < results_a.size(); i++) {
        std::cout << "  BPD " << static_cast<int>(results_a[i].bpd) << ":\n";
        std::cout << "    A: " << results_a[i].plan_str << "\n";
        std::cout << "    B: " << results_b[i].plan_str << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
