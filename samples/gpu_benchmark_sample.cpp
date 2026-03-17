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

#include <cstdlib>
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
#include "saq/caq_estimator.h"
#include "saq/saq_estimator.h"
#include "saq/fast_scan.h"
#include "saq/lut.h"
#include "saq/code_helper.h"

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

    // ---- CPU Encode + Distance Diagnostic ----
    {
        LOG(INFO) << "--- CPU Encode (" << num_threads << " threads) ---";
        StopW sw;
        IVF cpu_ivf(N, D, K, cfg);
        std::srand(42);  // Seed BEFORE set_variance (which triggers rotation generation)
        cpu_ivf.set_variance(variances.row(0));
        cpu_ivf.construct(vectors, centroids, cids.data(), num_threads);
        auto cpu_ms = sw.getElapsedTimeMicro() / 1000.0;
        LOG(INFO) << "CPU encode time: " << cpu_ms << " ms (" << cpu_ms / 1e3 << " s)";

        // ---- Distance Diagnostic ----
        LOG(INFO) << "--- Distance Diagnostic ---";

        std::string query_file = data_dir + "/queries_pca.fvecs";
        FloatRowMat diag_queries;
        load_something<float, FloatRowMat>(query_file.c_str(), diag_queries);

        // Run CPU search for query 0
        SearcherConfig diag_cfg;
        diag_cfg.dist_type = DistType::L2Sqr;
        std::vector<PID> cpu_res(100);
        cpu_ivf.search<DistType::L2Sqr>(diag_queries.row(0), 100, nprobe, diag_cfg, cpu_res.data());
        LOG(INFO) << "CPU top-5 for query 0: " << cpu_res[0] << " " << cpu_res[1]
                  << " " << cpu_res[2] << " " << cpu_res[3] << " " << cpu_res[4];

        // Compute brute-force L2 distances for CPU top-5
        LOG(INFO) << "Brute-force L2 dists for CPU top-5:";
        for (int i = 0; i < 5; ++i) {
            float bf = (diag_queries.row(0) - vectors.row(cpu_res[i])).squaredNorm();
            LOG(INFO) << "  vec " << cpu_res[i] << ": bf_dist=" << bf;
        }

        // Now check GPU top-5 — re-read from the GPU search results above
        // (The GPU search was already run in the search section)
        // Re-run GPU search on a fresh index with SAME random seed as CPU
        gpu::GpuIVF gpu_diag(N, D, K, cfg);
        std::srand(42);  // Same seed BEFORE set_variance
        gpu_diag.set_variance(variances.row(0));
        gpu_diag.construct(vectors, centroids, cids.data());
        std::vector<PID> gpu_diag_res(100);
        gpu_diag.search_batch(diag_queries.topRows(1), 100, nprobe, diag_cfg, gpu_diag_res.data());

        LOG(INFO) << "GPU top-5 for query 0: " << gpu_diag_res[0] << " " << gpu_diag_res[1]
                  << " " << gpu_diag_res[2] << " " << gpu_diag_res[3] << " " << gpu_diag_res[4];

        LOG(INFO) << "Brute-force L2 dists for GPU top-5:";
        for (int i = 0; i < 5; ++i) {
            if (gpu_diag_res[i] == 0xFFFFFFFF) {
                LOG(INFO) << "  vec INVALID";
                continue;
            }
            float bf = (diag_queries.row(0) - vectors.row(gpu_diag_res[i])).squaredNorm();
            LOG(INFO) << "  vec " << gpu_diag_res[i] << ": bf_dist=" << bf;
        }

        // Check if the GPU encode produces the same factors as CPU encode
        // by running CPU search on GPU-encoded data
        // Simpler: just run the CPU search itself and see recall
        LOG(INFO) << "CPU search recall for reference:";
        {
            std::string gt_file = data_dir + "/groundtruth.ivecs";
            UintRowMat gt;
            load_something<uint32_t, UintRowMat>(gt_file.c_str(), gt);
            size_t topk2 = std::min((size_t)100, (size_t)gt.cols());
            std::vector<PID> cpu_batch_res(diag_queries.rows() * topk2);
            for (size_t q = 0; q < (size_t)diag_queries.rows(); ++q) {
                cpu_ivf.search<DistType::L2Sqr>(diag_queries.row(q), topk2, nprobe, diag_cfg,
                    cpu_batch_res.data() + q * topk2);
            }
            size_t correct = 0;
            for (size_t q = 0; q < (size_t)diag_queries.rows(); ++q) {
                for (size_t k = 0; k < topk2 && k < (size_t)gt.cols(); ++k) {
                    for (size_t r = 0; r < topk2; ++r) {
                        if (cpu_batch_res[q * topk2 + r] == (PID)gt(q, k)) { correct++; break; }
                    }
                }
            }
            double cpu_recall = 100.0 * correct / (diag_queries.rows() * std::min(topk2, (size_t)gt.cols()));
            LOG(INFO) << "CPU Recall@" << topk2 << " = " << cpu_recall << "%";
        }

        // Also check: how many of GPU top-100 are in CPU top-100?
        size_t overlap = 0;
        for (int i = 0; i < 100; ++i) {
            for (int j = 0; j < 100; ++j) {
                if (gpu_diag_res[i] != 0xFFFFFFFF && gpu_diag_res[i] == cpu_res[j]) {
                    overlap++;
                    break;
                }
            }
        }
        LOG(INFO) << "Overlap: GPU top-100 ∩ CPU top-100 = " << overlap << " vectors";

        // Check if CPU top-1 (vec 63099) appears anywhere in GPU top-100
        PID cpu_top1 = cpu_res[0];
        int gpu_rank_of_cpu_top1 = -1;
        for (int i = 0; i < 100; ++i) {
            if (gpu_diag_res[i] == cpu_top1) { gpu_rank_of_cpu_top1 = i; break; }
        }
        LOG(INFO) << "CPU top-1 (vec " << cpu_top1 << ") rank in GPU top-100: "
                  << (gpu_rank_of_cpu_top1 >= 0 ? std::to_string(gpu_rank_of_cpu_top1) : "NOT FOUND");

        // Check: which cluster does cpu_top1 belong to?
        PID cpu_top1_cluster = cids[cpu_top1];
        LOG(INFO) << "CPU top-1 belongs to cluster " << cpu_top1_cluster;

        // Check rotation matrices match
        {
            const auto& cpu_saq = *cpu_ivf.get_saq_data();
            const auto& gpu_saq = *gpu_diag.get_saq_data();
            for (size_t s = 0; s < 2 && s < cpu_saq.base_datas.size(); ++s) {
                const auto& cpu_rot = cpu_saq.base_datas[s].rotator;
                const auto& gpu_rot = gpu_saq.base_datas[s].rotator;
                if (cpu_rot && gpu_rot) {
                    auto cpu_p = cpu_rot->get_P();
                    auto gpu_p = gpu_rot->get_P();
                    LOG(INFO) << "Rotation seg " << s << " CPU P[0,0:3]="
                              << cpu_p(0,0) << " " << cpu_p(0,1) << " " << cpu_p(0,2);
                    LOG(INFO) << "Rotation seg " << s << " GPU P[0,0:3]="
                              << gpu_p(0,0) << " " << gpu_p(0,1) << " " << gpu_p(0,2);
                }
            }
        }

        // === CPU-side reference: manually walk through compAccurateDist ===
        {
            LOG(INFO) << "=== CPU-side compAccurateDist reference for vec " << cpu_res[0] << " ===";
            PID target = cpu_res[0];
            PID tclu = cids[target];
            const auto& saq_data = *cpu_ivf.get_saq_data();
            const auto& pclusters = cpu_ivf.get_pclusters();
            const auto& pcluster = pclusters[tclu];
            size_t num_segs = saq_data.quant_plan.size();

            // Find target vector's position within the cluster
            int target_pos = -1;
            for (size_t i = 0; i < pcluster.num_vec_; ++i) {
                if (pcluster.get_segment(0).ids()[i] == target) {
                    target_pos = (int)i;
                    break;
                }
            }
            LOG(INFO) << "  Target vec " << target << " at pos " << target_pos
                      << " in cluster " << tclu << " (size=" << pcluster.num_vec_ << ")";

            if (target_pos >= 0) {
                float total_cpu_dist = 0.0f;
                size_t dim_off = 0;

                for (size_t s = 0; s < num_segs; ++s) {
                    const auto& seg = pcluster.get_segment(s);
                    const auto& bdata = saq_data.base_datas[s];
                    size_t D_seg = saq_data.quant_plan[s].first;
                    size_t bits = saq_data.quant_plan[s].second;
                    float sq_delta = (bits > 0) ? 2.0f / (float)(1 << bits) : 0.0f;

                    // Rotate query segment
                    FloatVec q_seg = diag_queries.row(0).segment(dim_off, D_seg);
                    FloatVec q_rot = bdata.rotator ? (q_seg * bdata.rotator->get_P()).eval() : q_seg;

                    // For L2: query residual = rotated_query - centroid
                    FloatVec q_resid = q_rot - seg.centroid();
                    float q_l2sqr = q_resid.squaredNorm();
                    float sum_q = q_resid.sum();

                    // Get factors
                    size_t blk = target_pos / KFastScanSize;
                    size_t j = target_pos % KFastScanSize;
                    float o_l2norm = seg.factor_o_l2norm(blk)[j];
                    float o_l2sqr = o_l2norm * o_l2norm;

                    if (bits == 0) {
                        float seg_dist = o_l2sqr + q_l2sqr;
                        LOG(INFO) << "  Seg " << s << " (D=" << D_seg << " bits=0): "
                                  << "o_l2sqr=" << o_l2sqr << " q_l2sqr=" << q_l2sqr
                                  << " seg_dist=" << seg_dist;
                        total_cpu_dist += seg_dist;
                        dim_off += D_seg;
                        continue;
                    }

                    // Build LUT manually (same as Lut::prepare)
                    size_t num_codebooks = D_seg / 4;
                    constexpr int kPos[16] = {3,3,2,3,1,3,2,3,0,3,2,3,1,3,2,3};

                    // Compute ip_xb_qprime: LUT sum for this vector's short codes
                    // First need to build the float LUT
                    std::vector<float> lut_float(num_codebooks * 16);
                    for (size_t cb = 0; cb < num_codebooks; ++cb) {
                        float* lut16 = lut_float.data() + cb * 16;
                        const float* q4 = q_resid.data() + cb * 4;
                        lut16[0] = 0.0f;
                        for (int jj = 1; jj < 16; ++jj) {
                            int lb = jj & (-jj);
                            lut16[jj] = lut16[jj - lb] + q4[kPos[jj]];
                        }
                    }

                    // Now we need the short code for this vector to look up the LUT
                    // But CPU short codes are in fastscan-packed layout, not raw...
                    // Instead, let's use the Lut class directly
                    Lut lut(D_seg, bits > 1 ? bits - 1 : 0);
                    lut.prepare(q_resid);
                    float q_l2sqr_lut = lut.getQL2Sqr();

                    // compFastIP to get ip_xb_qprime
                    // This needs the short codes in fastscan layout
                    // Let's just use compAccurateDist via the estimator

                    // Actually, let's just build a CaqCluEstimator and call compAccurateDist
                    CaqCluEstimator<DistType::L2Sqr> estimator(bdata, diag_cfg, diag_queries.row(0));
                    estimator.prepare(&seg);

                    // compFastIP to fill ip_xb_qprime
                    float dummy_fast[32];
                    __m512 fst[2];
                    estimator.compFastDist(blk, fst);

                    float cpu_acc_dist = estimator.compAccurateDist(target_pos);

                    // Get the factors to print
                    const ExFactor& ex_fac = seg.long_factor(target_pos);

                    // Manually compute getExtIP
                    const uint8_t* long_code = seg.long_code(target_pos);
                    auto IP_FUNC = get_IP_FUNC(bits > 1 ? bits - 1 : 0);
                    float ext_ip = IP_FUNC(q_resid.data(), long_code, D_seg);

                    // Compute v_mx from rotated residual (same as encoder)
                    FloatVec v_seg_raw = vectors.row(target).segment(dim_off, D_seg);
                    FloatVec c_seg_raw = centroids.row(tclu).segment(dim_off, D_seg);
                    FloatVec resid_raw = v_seg_raw - c_seg_raw;
                    FloatVec resid_rot_v = bdata.rotator ? (resid_raw * bdata.rotator->get_P()).eval() : resid_raw;
                    float cpu_v_mx = resid_rot_v.cwiseAbs().maxCoeff();
                    float cpu_delta = 2.0f * cpu_v_mx / (float)((1 << bits));

                    LOG(INFO) << "  Seg " << s << " (D=" << D_seg << " bits=" << bits << "):";
                    LOG(INFO) << "    v_mx=" << cpu_v_mx << " delta=" << cpu_delta;
                    LOG(INFO) << "    o_l2norm=" << o_l2norm << " o_l2sqr=" << o_l2sqr
                              << " q_l2sqr=" << q_l2sqr << " (lut q_l2sqr=" << q_l2sqr_lut << ")";
                    LOG(INFO) << "    sum_q=" << sum_q << " sq_delta=" << sq_delta;
                    LOG(INFO) << "    rescale=" << ex_fac.rescale << " error=" << ex_fac.error;
                    LOG(INFO) << "    ext_ip(IP_FUNC)=" << ext_ip;
                    LOG(INFO) << "    compAccurateDist=" << cpu_acc_dist;

                    total_cpu_dist += cpu_acc_dist;
                    dim_off += D_seg;
                }
                LOG(INFO) << "  Total CPU compAccurateDist: " << total_cpu_dist;

                // Brute-force for reference
                float bf = (diag_queries.row(0) - vectors.row(target)).squaredNorm();
                LOG(INFO) << "  Brute-force L2: " << bf;
            }
        }

        // Compare CPU vs GPU short codes for vec 63099
        {
            PID target = cpu_res[0];
            PID tclu = cids[target];
            const auto& pclusters = cpu_ivf.get_pclusters();
            const auto& seg0 = pclusters[tclu].get_segment(0);
            const auto& gpu_pool = gpu_diag.get_pool();
            const auto& gpu_saq = *gpu_diag.get_saq_data();
            size_t D_seg0 = gpu_saq.quant_plan[0].first;
            size_t bits0 = gpu_saq.quant_plan[0].second;

            // Find CPU position of target
            int cpu_pos = -1;
            for (size_t i = 0; i < seg0.num_vec(); ++i) {
                if (seg0.ids()[i] == target) { cpu_pos = (int)i; break; }
            }

            // Find GPU position of target
            size_t gpu_clu_off = gpu_pool.cluster_offsets[tclu];
            size_t gpu_clu_sz = gpu_pool.cluster_offsets[tclu+1] - gpu_clu_off;
            std::vector<uint32_t> gpu_ids(gpu_clu_sz);
            gpu::download(gpu_ids.data(), gpu_pool.ids.get() + gpu_clu_off, gpu_clu_sz);
            int gpu_pos = -1;
            for (size_t i = 0; i < gpu_clu_sz; ++i) {
                if (gpu_ids[i] == (uint32_t)target) { gpu_pos = (int)i; break; }
            }

            LOG(INFO) << "Code comparison for vec " << target << " seg0 (D=" << D_seg0
                      << " bits=" << bits0 << "): cpu_pos=" << cpu_pos << " gpu_pos=" << gpu_pos;

            if (cpu_pos >= 0 && gpu_pos >= 0) {
                // Download GPU short codes (in GPU blocked layout)
                size_t num_cb = D_seg0 / 4;
                size_t gpu_blk_off = gpu_pool.block_offsets[tclu];
                size_t gpu_blk = gpu_pos / 32;
                size_t gpu_vec_in_blk = gpu_pos % 32;
                std::vector<uint8_t> gpu_sc(num_cb);
                gpu::download(gpu_sc.data(),
                    gpu_pool.segments[0].short_codes.get() + (gpu_blk_off + gpu_blk) * 32 * num_cb + gpu_vec_in_blk * num_cb,
                    num_cb);

                // CPU short codes are in fastscan-packed layout — can't easily compare byte-by-byte
                // Instead, download GPU rescale and compare with CPU
                float gpu_resc = 0;
                gpu::download(&gpu_resc, gpu_pool.segments[0].factor_rescale.get() + gpu_clu_off + gpu_pos, 1);
                ExFactor cpu_fac = seg0.long_factor(cpu_pos);

                LOG(INFO) << "  CPU rescale=" << cpu_fac.rescale << " GPU rescale=" << gpu_resc
                          << " CPU error=" << cpu_fac.error;
                LOG(INFO) << "  GPU warp_id for this vec = " << (gpu_clu_off + gpu_pos);
                LOG(INFO) << "  GPU short codes[0:7]=" << (int)gpu_sc[0] << " " << (int)gpu_sc[1]
                          << " " << (int)gpu_sc[2] << " " << (int)gpu_sc[3]
                          << " " << (int)gpu_sc[4] << " " << (int)gpu_sc[5]
                          << " " << (int)gpu_sc[6] << " " << (int)gpu_sc[7];
            }
        }

        // Check if GPU searched that cluster (it should if nprobe=200 and K=4096)
        // We can verify by checking if ANY vector from that cluster appears in GPU results
        int vecs_from_target_cluster = 0;
        for (int i = 0; i < 100; ++i) {
            if (gpu_diag_res[i] != 0xFFFFFFFF && cids[gpu_diag_res[i]] == cpu_top1_cluster) {
                vecs_from_target_cluster++;
            }
        }
        LOG(INFO) << "GPU results from cluster " << cpu_top1_cluster << ": " << vecs_from_target_cluster;

        // === Key test: download GPU rescale factors and compare with CPU ===
        // For cluster cpu_top1_cluster, segment 0, compare rescale values
        {
            const auto& pool = gpu_diag.get_pool();
            const auto& gpu_clusters = gpu_diag.get_gpu_clusters();
            const auto& saq_data = *gpu_diag.get_saq_data();
            size_t num_segs = saq_data.quant_plan.size();

            // Find position of cpu_top1 in sorted order
            // It's in cluster cpu_top1_cluster. Need to find its position within that cluster.
            // We can get cluster info from pool
            size_t clu_size = gpu_clusters[cpu_top1_cluster].num_vec;
            size_t clu_offset = pool.cluster_offsets[cpu_top1_cluster];

            // Download GPU IDs for this cluster to find which position is cpu_top1
            std::vector<uint32_t> gpu_ids(clu_size);
            gpu::download(gpu_ids.data(), pool.ids.get() + clu_offset, clu_size);

            int pos_in_cluster = -1;
            for (size_t i = 0; i < clu_size; ++i) {
                if (gpu_ids[i] == (uint32_t)cpu_top1) { pos_in_cluster = (int)i; break; }
            }
            LOG(INFO) << "Vec " << cpu_top1 << " is at position " << pos_in_cluster
                      << " in cluster " << cpu_top1_cluster << " (size=" << clu_size << ")";

            // Compute brute-force per-segment residual distances
            LOG(INFO) << "Pool: total_vecs=" << pool.total_vecs_ << " total_blocks=" << pool.total_blocks_ << " K=" << pool.K_;

        // Verify d_block_offsets matches host
        std::vector<uint32_t> dev_blk_off(pool.K_ + 1);
        gpu::download(dev_blk_off.data(), pool.d_block_offsets.get(), pool.K_ + 1);
        LOG(INFO) << "Host block_offsets[3975]=" << pool.block_offsets[3975]
                  << " Device d_block_offsets[3975]=" << dev_blk_off[3975];

        // Download o_l2norm for global_block 4967, position 0
        float test_o_l2n = -1.0f;
        gpu::download(&test_o_l2n, pool.segments[0].factor_o_l2norm.get() + 4967 * 32, 1);
        LOG(INFO) << "Direct download o_l2norm at pool.seg[0].factor_o_l2norm[4967*32+0] = " << test_o_l2n;

        // Verify: pointer values in the pool vs what we'd put in the descriptor
        LOG(INFO) << "pool.segments[0].factor_o_l2norm.get() = " << (void*)pool.segments[0].factor_o_l2norm.get();
        LOG(INFO) << "Expected read address: " << (void*)(pool.segments[0].factor_o_l2norm.get() + 4967 * 32);
        LOG(INFO) << "Per-segment brute-force residual distances for vec " << cpu_top1 << ":";
            float total_bf_resid = 0;
            size_t dim_off = 0;
            for (size_t s = 0; s < num_segs; ++s) {
                size_t D_seg = saq_data.quant_plan[s].first;
                const auto& bdata = saq_data.base_datas[s];

                // Rotate query and vector segments
                FloatVec q_seg = diag_queries.row(0).segment(dim_off, D_seg);
                FloatVec v_seg = vectors.row(cpu_top1).segment(dim_off, D_seg);
                FloatVec c_seg = centroids.row(cpu_top1_cluster).segment(dim_off, D_seg);

                FloatVec q_rot, v_rot, c_rot;
                if (bdata.rotator) {
                    q_rot = q_seg * bdata.rotator->get_P();
                    v_rot = v_seg * bdata.rotator->get_P();
                    c_rot = c_seg * bdata.rotator->get_P();
                } else {
                    q_rot = q_seg; v_rot = v_seg; c_rot = c_seg;
                }

                FloatVec q_resid = q_rot - c_rot;
                FloatVec v_resid = v_rot - c_rot;
                float seg_bf = (q_resid - v_resid).squaredNorm();
                total_bf_resid += seg_bf;

                float q_resid_l2sqr = q_resid.squaredNorm();
                float v_resid_l2sqr = v_resid.squaredNorm();
                float ip_qv = q_resid.dot(v_resid);

                LOG(INFO) << "  Seg " << s << " (D=" << D_seg << "): bf_resid_dist=" << seg_bf
                          << " q_l2sqr=" << q_resid_l2sqr << " v_l2sqr=" << v_resid_l2sqr
                          << " ip=" << ip_qv << " (check: " << (q_resid_l2sqr + v_resid_l2sqr - 2*ip_qv) << ")";
                dim_off += D_seg;
            }
            LOG(INFO) << "Total brute-force residual dist: " << total_bf_resid;

            if (pos_in_cluster >= 0) {
                size_t vec_offset = clu_offset + pos_in_cluster;
                size_t blk_off = pool.block_offsets[cpu_top1_cluster];
                size_t blk_in_clu = pos_in_cluster / 32;
                size_t vec_in_blk = pos_in_cluster % 32;
                size_t global_blk = blk_off + blk_in_clu;

                for (size_t s = 0; s < num_segs && s < 2; ++s) {
                    size_t D_seg = saq_data.quant_plan[s].first;
                    size_t bits = saq_data.quant_plan[s].second;
                    size_t ex_bits = bits > 1 ? bits - 1 : 0;
                    size_t long_bpv = ex_bits > 0 ? D_seg * ex_bits / 8 : 0;

                    float gpu_rescale = 0, gpu_o_l2norm = 0;
                    gpu::download(&gpu_rescale, pool.segments[s].factor_rescale.get() + vec_offset, 1);
                    gpu::download(&gpu_o_l2norm, pool.segments[s].factor_o_l2norm.get() + global_blk * 32 + vec_in_blk, 1);

                    LOG(INFO) << "Seg " << s << " (D=" << D_seg << " bits=" << bits
                              << "): GPU rescale=" << gpu_rescale << " o_l2norm=" << gpu_o_l2norm;

                    // Download and decode first 8 long code values
                    if (long_bpv > 0) {
                        std::vector<uint8_t> lc(long_bpv);
                        gpu::download(lc.data(), pool.segments[s].long_codes.get() + vec_offset * long_bpv, long_bpv);
                        std::string vals;
                        for (size_t d = 0; d < 8 && d < D_seg; ++d) {
                            size_t bit_off = d * ex_bits;
                            int code_val = 0;
                            for (size_t b = 0; b < ex_bits; ++b) {
                                size_t gb = bit_off + b;
                                if ((lc[gb/8] >> (gb%8)) & 1) code_val |= (1 << b);
                            }
                            vals += std::to_string(code_val) + " ";
                        }
                        LOG(INFO) << "  Long codes (first 8): " << vals
                                  << " (max=" << ((1<<ex_bits)-1) << ")";
                    }

                    // Also download and show short code for this vector's first codebook
                    if (bits > 0) {
                        size_t num_cb = D_seg / 4;
                        std::vector<uint8_t> sc(num_cb);
                        gpu::download(sc.data(),
                            pool.segments[s].short_codes.get() + global_blk * 32 * num_cb + vec_in_blk * num_cb,
                            num_cb);
                        LOG(INFO) << "  Short codes (first 8 codebooks): "
                                  << (int)sc[0] << " " << (int)sc[1] << " " << (int)sc[2] << " "
                                  << (int)sc[3] << " " << (int)sc[4] << " " << (int)sc[5] << " "
                                  << (int)sc[6] << " " << (int)sc[7];
                    }
                }
            }
        }
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
