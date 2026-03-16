/// @file gpu_benchmark_sample.cpp
/// @brief GPU vs CPU encode benchmark for SAQ-IVF on DBpedia 100K dataset.
///
/// Loads pre-computed PCA-transformed data and runs GPU encode via GpuIVF::construct,
/// comparing timing against CPU encode via IVF::construct.
///
/// Usage: gpu_benchmark_sample <data_dir> <bpd> <K> [num_threads] [nprobe]
///   data_dir:      Path to dataset (e.g., data/datasets/dbpedia_100k)
///   bpd:           Bits per dimension (e.g., 2.0)
///   K:             Number of clusters (e.g., 4096)
///   num_threads:   CPU thread count for comparison (default: 8)
///   nprobe:        Number of clusters to search (default: 200)

#ifdef SAQ_USE_CUDA

#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "saq/defines.h"
#include "saq/config.h"
#include "saq/io_utils.h"
#include "saq/stopw.h"
#include "saq/gpu/gpu_ivf.h"
#include "index/ivf_index.h"

using namespace saq;

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <data_dir> <bpd> <K> [num_threads]" << std::endl;
        return 1;
    }

    std::string data_dir = argv[1];
    float bpd = std::stof(argv[2]);
    int K = std::stoi(argv[3]);
    int num_threads = (argc > 4) ? std::stoi(argv[4]) : 8;
    int nprobe = (argc > 5) ? std::stoi(argv[5]) : 200;

    LOG(INFO) << "GPU Benchmark: data=" << data_dir << " bpd=" << bpd << " K=" << K
              << " nprobe=" << nprobe;

    // Load data
    std::string k_str = std::to_string(K);
    std::string data_file      = data_dir + "/vectors_pca.fvecs";
    std::string centroid_file  = data_dir + "/centroids_" + k_str + "_pca.fvecs";
    std::string cids_file      = data_dir + "/cluster_ids_" + k_str + ".ivecs";
    std::string variance_file  = data_dir + "/variances_pca.fvecs";

    FloatRowMat vectors, centroids, variances;
    UintRowMat cluster_ids_mat;
    load_something<float, FloatRowMat>(data_file.c_str(), vectors);
    load_something<float, FloatRowMat>(centroid_file.c_str(), centroids);
    load_something<uint32_t, UintRowMat>(cids_file.c_str(), cluster_ids_mat);
    load_something<float, FloatRowMat>(variance_file.c_str(), variances);

    size_t N = vectors.rows();
    size_t D = vectors.cols();

    LOG(INFO) << "Loaded: N=" << N << " D=" << D << " K=" << K;

    // Config
    QuantizeConfig cfg;
    cfg.avg_bits = bpd;
    cfg.enable_segmentation = true;
    cfg.single.quant_type = BaseQuantType::CAQ;
    cfg.single.caq_adj_rd_lmt = 6;
    cfg.single.use_fastscan = true;
    cfg.single.random_rotation = true;

    // Prepare cluster IDs
    std::vector<PID> cids(N);
    for (size_t i = 0; i < N; ++i)
        cids[i] = static_cast<PID>(cluster_ids_mat(i, 0));

    // ---- GPU Encode ----
    {
        LOG(INFO) << "--- GPU Encode ---";
        try {
            StopW sw;
            gpu::GpuIVF gpu_ivf(N, D, K, cfg);
            gpu_ivf.set_variance(variances.row(0));
            gpu_ivf.construct(vectors, centroids, cids.data());
            auto gpu_ms = sw.getElapsedTimeMicro() / 1000.0;
            LOG(INFO) << "GPU encode time: " << gpu_ms << " ms (" << gpu_ms / 1e3 << " s)";
        } catch (const std::exception& e) {
            LOG(ERROR) << "GPU encode failed: " << e.what();
        }
    }

    // ---- GPU Search ----
    {
        LOG(INFO) << "--- GPU Batch Search ---";

        // Load queries and ground truth
        std::string query_file = data_dir + "/queries_pca.fvecs";
        std::string gt_file = data_dir + "/groundtruth.ivecs";

        FloatRowMat queries;
        UintRowMat gt;
        load_something<float, FloatRowMat>(query_file.c_str(), queries);
        load_something<uint32_t, UintRowMat>(gt_file.c_str(), gt);

        size_t Q = queries.rows();
        size_t topk = std::min((size_t)100, (size_t)gt.cols());

        // Re-build GPU index for search (need the gpu_ivf object to persist)
        gpu::GpuIVF gpu_ivf2(N, D, K, cfg);
        gpu_ivf2.set_variance(variances.row(0));
        gpu_ivf2.construct(vectors, centroids, cids.data());

        SearcherConfig search_cfg;
        search_cfg.dist_type = DistType::L2Sqr;

        std::vector<PID> gpu_results(Q * topk);

        try {
            StopW sw;
            gpu_ivf2.search_batch(queries, topk, nprobe, search_cfg, gpu_results.data());
            auto gpu_search_ms = sw.getElapsedTimeMicro() / 1000.0;
            LOG(INFO) << "GPU batch search: Q=" << Q << " nprobe=" << nprobe
                      << " topk=" << topk
                      << " time=" << gpu_search_ms << " ms"
                      << " QPS=" << (Q * 1000.0 / gpu_search_ms);

            // Compute recall vs ground truth
            size_t correct = 0;
            for (size_t q = 0; q < Q; ++q) {
                for (size_t k = 0; k < topk && k < (size_t)gt.cols(); ++k) {
                    for (size_t r = 0; r < topk; ++r) {
                        if (gpu_results[q * topk + r] == (PID)gt(q, k)) {
                            correct++;
                            break;
                        }
                    }
                }
            }
            double recall = 100.0 * correct / (Q * std::min(topk, (size_t)gt.cols()));
            LOG(INFO) << "GPU Recall@" << topk << " = " << recall << "%";
        } catch (const std::exception& e) {
            LOG(ERROR) << "GPU search failed: " << e.what();
        }
    }

    // ---- CPU Encode (for comparison) ----
    {
        LOG(INFO) << "--- CPU Encode (" << num_threads << " threads) ---";
        StopW sw;
        IVF cpu_ivf(N, D, K, cfg);
        cpu_ivf.set_variance(variances.row(0));
        cpu_ivf.construct(vectors, centroids, cids.data(), num_threads);
        auto cpu_ms = sw.getElapsedTimeMicro() / 1000.0;
        LOG(INFO) << "CPU encode time: " << cpu_ms << " ms (" << cpu_ms / 1e3 << " s)";
    }

    return 0;
}

#else

#include <iostream>
int main() {
    std::cerr << "GPU benchmark requires SAQ_BUILD_CUDA=ON" << std::endl;
    return 1;
}

#endif
