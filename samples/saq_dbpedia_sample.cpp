/// @file saq_dbpedia_sample.cpp
/// @brief End-to-end SAQ-IVF benchmark on DBpedia 100K dataset.
///
/// Loads pre-computed PCA-transformed data, centroids, cluster assignments,
/// and ground truth produced by the Python preprocessing scripts. Builds
/// an IVF index and evaluates recall at multiple nprobe settings.
///
/// Usage: saq_dbpedia_sample [data_dir] [results_dir] [bpd] [num_clusters] [nprobe] [num_threads]
///   data_dir:      Path to dataset (default: data/datasets/dbpedia_100k)
///   results_dir:   Output directory (default: results/saq)
///   bpd:           Bits per dimension (default: 2.0)
///   num_clusters:  K for clustering (default: 4096)
///   nprobe:        Primary nprobe for search (default: 200)
///   num_threads:   Thread count for index construction (default: 8)

#include "index/ivf_index.h"
#include "saq/config.h"
#include "saq/defines.h"
#include "saq/io_utils.h"
#include "saq/stopw.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef SAQ_USE_OPENMP
#include <omp.h>
#endif

using namespace saq;

// ============================================================================
// Recall computation
// ============================================================================

/// @brief Compute recall@k for a single query.
///
/// Counts how many of the top-k returned results appear in the ground truth
/// top-k. The ground truth matrix has one row per query, each containing
/// the IDs of the 100 nearest neighbors.
static float ComputeRecallAtK(const std::vector<std::vector<PID>>& results,
                               const UintRowMat& gt, size_t k) {
    size_t nq = results.size();
    size_t total_correct = 0;
    size_t total_count = 0;

    for (size_t q = 0; q < nq; ++q) {
        size_t gt_k = std::min(k, static_cast<size_t>(gt.cols()));
        size_t res_k = std::min(k, results[q].size());

        // Build ground truth set for this query
        std::unordered_set<PID> gt_set;
        for (size_t i = 0; i < gt_k; ++i) {
            gt_set.insert(gt(q, i));
        }

        // Count hits
        for (size_t i = 0; i < res_k; ++i) {
            if (gt_set.count(results[q][i])) {
                total_correct++;
            }
        }
        total_count += gt_k;
    }

    return total_count > 0
        ? static_cast<float>(total_correct) / static_cast<float>(total_count)
        : 0.0f;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    std::cout << "========================================================================\n";
    std::cout << "SAQ-IVF Benchmark: DBpedia 100K Dataset\n";
    std::cout << "========================================================================\n\n";

    // -----------------------------------------------------------------------
    // 1. Parse command-line arguments
    // -----------------------------------------------------------------------
    std::string data_dir     = "data/datasets/dbpedia_100k";
    std::string results_dir  = "results/saq";
    float       bpd          = 2.0f;
    size_t      num_clusters = 4096;
    size_t      primary_nprobe = 200;
    int         num_threads  = 8;

    if (argc > 1) data_dir        = argv[1];
    if (argc > 2) results_dir     = argv[2];
    if (argc > 3) bpd             = std::stof(argv[3]);
    if (argc > 4) num_clusters    = std::stoul(argv[4]);
    if (argc > 5) primary_nprobe  = std::stoul(argv[5]);
    if (argc > 6) num_threads     = std::stoi(argv[6]);

    std::cout << "Configuration:\n";
    std::cout << "  data_dir:      " << data_dir << "\n";
    std::cout << "  results_dir:   " << results_dir << "\n";
    std::cout << "  bpd:           " << std::fixed << std::setprecision(2) << bpd << "\n";
    std::cout << "  num_clusters:  " << num_clusters << "\n";
    std::cout << "  nprobe:        " << primary_nprobe << "\n";
    std::cout << "  num_threads:   " << num_threads << "\n\n";

    // -----------------------------------------------------------------------
    // 2. Build file paths
    // -----------------------------------------------------------------------
    std::string k_str = std::to_string(num_clusters);

    std::string data_file      = data_dir + "/vectors_pca.fvecs";
    std::string query_file     = data_dir + "/queries_pca.fvecs";
    std::string centroid_file  = data_dir + "/centroids_" + k_str + "_pca.fvecs";
    std::string cids_file      = data_dir + "/cluster_ids_" + k_str + ".ivecs";
    std::string gt_file        = data_dir + "/groundtruth.ivecs";
    std::string variance_file  = data_dir + "/variances_pca.fvecs";

    // -----------------------------------------------------------------------
    // 3. Load data files
    // -----------------------------------------------------------------------
    std::cout << "[1/4] Loading data files...\n";

    // Check required files exist
    auto check_file = [](const std::string& path) {
        if (!file_exists(path.c_str())) {
            std::cerr << "ERROR: Required file not found: " << path << "\n";
            std::cerr << "Run the Python preprocessing scripts first:\n";
            std::cerr << "  python samples/preprocess_dbpedia.py\n";
            std::exit(1);
        }
    };

    check_file(data_file);
    check_file(query_file);
    check_file(centroid_file);
    check_file(cids_file);
    check_file(gt_file);

    FloatRowMat data, queries, centroids;
    UintRowMat  cluster_ids, gt;

    load_something<float, FloatRowMat>(data_file.c_str(), data);
    load_something<float, FloatRowMat>(query_file.c_str(), queries);
    load_something<float, FloatRowMat>(centroid_file.c_str(), centroids);
    load_something<uint32_t, UintRowMat>(cids_file.c_str(), cluster_ids);
    load_something<uint32_t, UintRowMat>(gt_file.c_str(), gt);

    // Optionally load variances
    FloatRowMat variances;
    bool have_variances = file_exists(variance_file.c_str());
    if (have_variances) {
        load_something<float, FloatRowMat>(variance_file.c_str(), variances);
    } else {
        std::cout << "  (variances file not found, will be computed from data)\n";
    }

    size_t num_vecs = static_cast<size_t>(data.rows());
    size_t num_dim  = static_cast<size_t>(data.cols());
    size_t num_q    = static_cast<size_t>(queries.rows());

    std::cout << "\n  Base vectors:  " << num_vecs << " x " << num_dim << "\n";
    std::cout << "  Queries:       " << num_q << " x " << queries.cols() << "\n";
    std::cout << "  Centroids:     " << centroids.rows() << " x " << centroids.cols() << "\n";
    std::cout << "  Ground truth:  " << gt.rows() << " x " << gt.cols() << "\n";
    std::cout << "  Compression:   " << std::fixed << std::setprecision(1)
              << (32.0f / bpd) << "x\n\n";

    // -----------------------------------------------------------------------
    // 4. Build the IVF index
    // -----------------------------------------------------------------------
    std::cout << "[2/4] Building IVF index (bpd=" << std::fixed << std::setprecision(2)
              << bpd << ", K=" << num_clusters << ")...\n";

    QuantizeConfig cfg;
    cfg.avg_bits              = bpd;
    cfg.single.quant_type     = BaseQuantType::CAQ;
    cfg.single.random_rotation = true;
    cfg.single.use_fastscan   = true;
    cfg.single.caq_adj_rd_lmt = 6;
    cfg.enable_segmentation   = true;

    IVF ivf(num_vecs, num_dim, num_clusters, cfg);

    if (have_variances && variances.rows() > 0) {
        // Convert the first row of the variance matrix to a FloatVec
        FloatVec var_vec = variances.row(0);
        ivf.set_variance(std::move(var_vec));
        std::cout << "  Variance data loaded and set.\n";
    }

    StopW build_timer;
    ivf.construct(data, centroids, cluster_ids.data(), num_threads);
    float build_time_s = build_timer.getElapsedTimeMili() / 1000.0f;

    std::cout << "  Build time: " << std::fixed << std::setprecision(2)
              << build_time_s << " seconds\n";

    // Print quantization plan
    const SaqData* saq_data = ivf.get_saq_data();
    if (saq_data) {
        std::cout << "  Quantization plan: ";
        size_t dims_sum = 0;
        for (auto& [dim_len, bits] : saq_data->quant_plan) {
            std::cout << "[" << dims_sum << ".." << (dims_sum + dim_len) << ")@" << bits << "b ";
            dims_sum += dim_len;
        }
        std::cout << "\n";
    }

    // Save index
    std::ostringstream bpd_ss;
    bpd_ss << std::fixed << std::setprecision(1) << bpd;

    // Create results directory
#ifdef _WIN32
    std::string mkdir_cmd = "mkdir \"" + results_dir + "\" 2>nul";
#else
    std::string mkdir_cmd = "mkdir -p \"" + results_dir + "\"";
#endif
    system(mkdir_cmd.c_str());

    std::string index_path = results_dir + "/ivf_k" + k_str + "_bpd" + bpd_ss.str() + ".index";
    ivf.save(index_path.c_str());
    std::cout << "  Index saved to: " << index_path << "\n";

    // Print build quality metrics
    auto& ip_metrics = ivf.quant_metrics_.norm_ip_o_oa;
    if (ip_metrics.cnt_ > 0) {
        std::cout << "  IP error: avg=" << std::fixed << std::setprecision(6)
                  << ip_metrics.avg() << "  max=" << ip_metrics.max() << "\n";
    }
    std::cout << "\n";

    // -----------------------------------------------------------------------
    // 5. Search with varying nprobe values
    // -----------------------------------------------------------------------
    std::cout << "[3/4] Running search benchmarks...\n";

    SearcherConfig searcher_cfg;
    searcher_cfg.dist_type = DistType::L2Sqr;

    constexpr size_t TOPK = 100;
    std::vector<size_t> nprobe_values = {1, 5, 10, 20, 50, 100, 200, 500};

    // Filter out nprobe values that exceed the number of clusters
    std::vector<size_t> valid_nprobes;
    for (size_t np : nprobe_values) {
        if (np <= num_clusters) {
            valid_nprobes.push_back(np);
        }
    }
    // Make sure the user-specified primary nprobe is included
    if (primary_nprobe <= num_clusters) {
        bool found = false;
        for (size_t np : valid_nprobes) {
            if (np == primary_nprobe) { found = true; break; }
        }
        if (!found) {
            valid_nprobes.push_back(primary_nprobe);
            std::sort(valid_nprobes.begin(), valid_nprobes.end());
        }
    }

    // Header
    std::cout << std::string(72, '-') << "\n";
    std::cout << std::setw(8) << "nprobe"
              << std::setw(12) << "Recall@1"
              << std::setw(12) << "Recall@10"
              << std::setw(12) << "Recall@100"
              << std::setw(14) << "Time(ms)"
              << std::setw(12) << "QPS"
              << "\n";
    std::cout << std::string(72, '-') << "\n";

    // Store results for file output
    struct SearchResult {
        size_t nprobe;
        float recall_at_1;
        float recall_at_10;
        float recall_at_100;
        float time_ms;
        float qps;
    };
    std::vector<SearchResult> all_results;

    for (size_t nprobe : valid_nprobes) {
        // Allocate result storage
        std::vector<std::vector<PID>> results(num_q, std::vector<PID>(TOPK));

        StopW search_timer;

        // Search all queries
        for (size_t q = 0; q < num_q; ++q) {
            ivf.search<DistType::L2Sqr>(
                queries.row(q), TOPK, nprobe, searcher_cfg,
                results[q].data());
        }

        float search_time_ms = search_timer.getElapsedTimeMili();
        float qps = static_cast<float>(num_q) / (search_time_ms / 1000.0f);

        // Compute recall at different k values
        float r1   = ComputeRecallAtK(results, gt, 1);
        float r10  = ComputeRecallAtK(results, gt, 10);
        float r100 = ComputeRecallAtK(results, gt, 100);

        std::cout << std::setw(8) << nprobe
                  << std::setw(11) << std::fixed << std::setprecision(2) << (r1 * 100) << "%"
                  << std::setw(11) << std::fixed << std::setprecision(2) << (r10 * 100) << "%"
                  << std::setw(11) << std::fixed << std::setprecision(2) << (r100 * 100) << "%"
                  << std::setw(14) << std::fixed << std::setprecision(1) << search_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps
                  << "\n";

        all_results.push_back({nprobe, r1, r10, r100, search_time_ms, qps});
    }
    std::cout << std::string(72, '-') << "\n\n";

    // -----------------------------------------------------------------------
    // 6. Write results to file
    // -----------------------------------------------------------------------
    std::cout << "[4/4] Writing results...\n";

    std::string results_file = results_dir + "/dbpedia_100k_k" + k_str
                             + "_bpd" + bpd_ss.str() + "_results.txt";

    std::ofstream out(results_file);
    if (out.is_open()) {
        out << "========================================================================\n";
        out << "SAQ-IVF Benchmark Results\n";
        out << "========================================================================\n\n";

        out << "Dataset:\n";
        out << "  Base vectors:  " << num_vecs << " x " << num_dim << "\n";
        out << "  Queries:       " << num_q << "\n";
        out << "  Centroids (K): " << num_clusters << "\n\n";

        out << "Index Configuration:\n";
        out << "  Bits/dim:      " << std::fixed << std::setprecision(2) << bpd << "\n";
        out << "  Quant type:    CAQ\n";
        out << "  Segmentation:  enabled\n";
        out << "  Fast scan:     enabled\n";
        out << "  Rotation:      random\n";
        out << "  Compression:   " << std::fixed << std::setprecision(1)
            << (32.0f / bpd) << "x\n";
        out << "  Build time:    " << std::fixed << std::setprecision(2)
            << build_time_s << " seconds\n";
        if (ip_metrics.cnt_ > 0) {
            out << "  IP error avg:  " << std::fixed << std::setprecision(6)
                << ip_metrics.avg() << "\n";
            out << "  IP error max:  " << std::fixed << std::setprecision(6)
                << ip_metrics.max() << "\n";
        }
        out << "\n";

        out << "Search Results (dist_type=L2Sqr):\n";
        out << std::string(72, '-') << "\n";
        out << std::setw(8) << "nprobe"
            << std::setw(12) << "Recall@1"
            << std::setw(12) << "Recall@10"
            << std::setw(12) << "Recall@100"
            << std::setw(14) << "Time(ms)"
            << std::setw(12) << "QPS"
            << "\n";
        out << std::string(72, '-') << "\n";

        for (const auto& r : all_results) {
            out << std::setw(8) << r.nprobe
                << std::setw(11) << std::fixed << std::setprecision(2) << (r.recall_at_1 * 100) << "%"
                << std::setw(11) << std::fixed << std::setprecision(2) << (r.recall_at_10 * 100) << "%"
                << std::setw(11) << std::fixed << std::setprecision(2) << (r.recall_at_100 * 100) << "%"
                << std::setw(14) << std::fixed << std::setprecision(1) << r.time_ms
                << std::setw(12) << std::fixed << std::setprecision(0) << r.qps
                << "\n";
        }
        out << std::string(72, '-') << "\n";

        out.close();
        std::cout << "  Results written to: " << results_file << "\n";
    } else {
        std::cerr << "  WARNING: Could not write results to: " << results_file << "\n";
    }

    std::cout << "\n========================================================================\n";
    std::cout << "Benchmark completed successfully!\n";
    std::cout << "  bpd=" << std::fixed << std::setprecision(2) << bpd
              << "  K=" << num_clusters
              << "  compression=" << std::fixed << std::setprecision(1) << (32.0f / bpd) << "x\n";
    std::cout << "========================================================================\n";

    return 0;
}
