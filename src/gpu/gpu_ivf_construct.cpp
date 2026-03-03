#ifdef SAQ_USE_CUDA

#include "saq/gpu/gpu_ivf.h"
#include "saq/gpu/gpu_utils.cuh"
#include "saq/gpu/gpu_encoder.cuh"
#include "saq/gpu/gpu_packer.cuh"
#include "saq/gpu/gpu_cluster_data.cuh"
#include "saq/initializer.h"
#include "saq/stopw.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include <cublas_v2.h>
#include <glog/logging.h>

namespace saq::gpu {

GpuIVF::GpuIVF(size_t n, size_t num_dim, size_t k, QuantizeConfig cfg)
    : num_data_(n), num_dim_(num_dim), num_cen_(k), cfg_(std::move(cfg)),
      saq_data_maker_(std::make_unique<SaqDataMaker>(cfg_, num_dim)) {}

GpuIVF::~GpuIVF() = default;

void GpuIVF::set_variance(FloatVec vars) {
    saq_data_maker_->set_variance(std::move(vars));
}

void GpuIVF::construct(const FloatRowMat& data,
                       const FloatRowMat& centroids,
                       const PID* cluster_ids) {
    LOG(INFO) << "Starting GPU IVF construction...";
    StopW stopw;
    StopW phase_timer;

    const size_t N = num_data_;
    const size_t D = num_dim_;
    const size_t K = num_cen_;

    // 1. Prepare CPU-side metadata
    // Initializer (centroid search stays on CPU for now)
    if (K < 20000ul) {
        initer_ = std::make_unique<FlatInitializer>(D, K);
    } else {
        LOG(FATAL) << "HNSW initializer not implemented";
    }
    initer_->set_centroids(centroids);

    // SaqData (quantization plan)
    if (!saq_data_maker_->is_variance_set()) {
        saq_data_maker_->compute_variance(data);
    }
    saq_data_ = saq_data_maker_->return_data();
    const auto& quant_plan = saq_data_->quant_plan;
    const auto& base_datas = saq_data_->base_datas;
    size_t num_segments = quant_plan.size();

    LOG(INFO) << "Quantization plan: " << num_segments << " segments";

    // 2. Compute cluster sizes and offsets (CPU-side)
    std::vector<size_t> cluster_sizes(K, 0);
    for (size_t i = 0; i < N; ++i) {
        cluster_sizes[cluster_ids[i]]++;
    }
    std::vector<size_t> cluster_offsets(K + 1, 0);
    std::partial_sum(cluster_sizes.begin(), cluster_sizes.end(), cluster_offsets.begin() + 1);

    // Sort vectors by cluster ID
    std::vector<size_t> order(N);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
        [&](size_t a, size_t b) { return cluster_ids[a] < cluster_ids[b]; });

    // Build sorted arrays
    std::vector<uint32_t> h_sorted_cids(N);
    std::vector<uint32_t> h_sorted_original_ids(N);
    FloatRowMat sorted_data(N, D);
    for (size_t i = 0; i < N; ++i) {
        h_sorted_cids[i] = cluster_ids[order[i]];
        h_sorted_original_ids[i] = static_cast<uint32_t>(order[i]);
        sorted_data.row(i) = data.row(order[i]);
    }

    auto cpu_prep_ms = stopw.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "[TIMING] CPU prep (sort+metadata): " << cpu_prep_ms << " ms";

    // 3. Upload to GPU
    phase_timer.reset();
    auto d_vectors = device_alloc<float>(N * D);
    auto d_centroids = device_alloc<float>(K * D);
    auto d_cluster_ids = device_alloc<uint32_t>(N);

    upload(d_vectors.get(), sorted_data.data(), N * D);
    upload(d_centroids.get(), centroids.data(), K * D);
    upload(d_cluster_ids.get(), h_sorted_cids.data(), N);
    SAQ_CUDA_CHECK(cudaDeviceSynchronize());
    auto upload_ms = phase_timer.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "[TIMING] H2D upload (vectors+centroids+cids): " << upload_ms << " ms";

    // 4. Allocate GPU cluster data
    phase_timer.reset();
    gpu_clusters_.clear();
    gpu_clusters_.resize(K);
    for (size_t c = 0; c < K; ++c) {
        gpu_clusters_[c].allocate(cluster_sizes[c], quant_plan);
        // Upload original IDs for this cluster
        if (cluster_sizes[c] > 0) {
            upload(gpu_clusters_[c].d_ids.get(),
                   h_sorted_original_ids.data() + cluster_offsets[c],
                   cluster_sizes[c]);
        }
    }
    SAQ_CUDA_CHECK(cudaDeviceSynchronize());
    auto alloc_ms = phase_timer.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "[TIMING] GPU cluster alloc + ID upload: " << alloc_ms << " ms";

    // cuBLAS handle
    CublasHandle cublas;

    // 5. Process each segment
    double total_kernel_ms = 0.0;
    double total_scatter_ms = 0.0;
    size_t dim_offset = 0;
    for (size_t seg = 0; seg < num_segments; ++seg) {
        size_t D_seg = quant_plan[seg].first;
        size_t num_bits = quant_plan[seg].second;
        const auto& bdata = base_datas[seg];

        LOG(INFO) << "Segment " << seg << ": dim=" << D_seg << " bits=" << num_bits;
        phase_timer.reset();

        // Allocate per-segment temporaries
        auto d_residuals = device_alloc<float>(N * D_seg);
        auto d_rotated = device_alloc<float>(N * D_seg);

        // 5a. Subtract centroids
        launch_subtract_centroid(
            d_vectors.get(), d_centroids.get(), d_cluster_ids.get(),
            d_residuals.get(), dim_offset, D_seg, D, N);

        // 5b. Rotation: d_rotated = d_residuals * P^T
        if (bdata.rotator) {
            // Upload rotation matrix
            auto d_P = device_alloc<float>(D_seg * D_seg);
            upload(d_P.get(), bdata.rotator->get_P().data(), D_seg * D_seg);

            // cuBLAS GEMM: C = A * B
            // A = d_residuals [N x D_seg], B = P [D_seg x D_seg]
            // In row-major: cublasSgemm with transposed args
            float alpha = 1.0f, beta = 0.0f;
            SAQ_CUBLAS_CHECK(cublasSgemm(
                cublas.get(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                (int)D_seg, (int)N, (int)D_seg,
                &alpha,
                d_P.get(), (int)D_seg,         // B (column-major = P^T row-major)
                d_residuals.get(), (int)D_seg,  // A (column-major view)
                &beta,
                d_rotated.get(), (int)D_seg));  // C
        } else {
            // No rotation: copy residuals to rotated
            SAQ_CUDA_CHECK(cudaMemcpy(d_rotated.get(), d_residuals.get(),
                N * D_seg * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        // Also rotate centroids for ip_cent_oa computation
        auto d_centroids_seg = device_alloc<float>(K * D_seg);
        if (bdata.rotator) {
            // Extract centroid segment and rotate on CPU, then upload
            FloatRowMat cent_seg(K, D_seg);
            for (size_t c = 0; c < K; ++c) {
                cent_seg.row(c) = centroids.row(c).segment(dim_offset, D_seg);
            }
            FloatRowMat cent_rotated = cent_seg * bdata.rotator->get_P();
            upload(d_centroids_seg.get(), cent_rotated.data(), K * D_seg);
        } else {
            // Extract and upload unrotated centroid segments
            FloatRowMat cent_seg(K, D_seg);
            for (size_t c = 0; c < K; ++c) {
                cent_seg.row(c) = centroids.row(c).segment(dim_offset, D_seg);
            }
            upload(d_centroids_seg.get(), cent_seg.data(), K * D_seg);
        }

        // 5c. CAQ Encode
        auto d_codes = device_alloc<int>(N * D_seg);
        auto d_o_l2norm = device_alloc<float>(N);
        auto d_fac_rescale = device_alloc<float>(N);
        auto d_fac_error = device_alloc<float>(N);
        auto d_ip_cent_oa = device_alloc<float>(N);

        uint16_t code_max = (1 << num_bits) - 1;
        if (bdata.cfg.caq_ori_qB) {
            code_max = (1 << bdata.cfg.caq_ori_qB) - 1;
        }

        launch_caq_encode(
            d_rotated.get(), d_codes.get(),
            d_o_l2norm.get(), d_fac_rescale.get(), d_fac_error.get(), d_ip_cent_oa.get(),
            d_centroids_seg.get(), d_cluster_ids.get(),
            D_seg, N, K, num_bits, code_max,
            bdata.cfg.caq_adj_rd_lmt, bdata.cfg.caq_adj_eps, bdata.cfg.caq_ori_qB);

        // 5d. Pack short codes (per-vector)
        size_t short_code_bytes = D_seg / 8;
        auto d_short_raw = device_alloc<uint8_t>(N * short_code_bytes);
        launch_pack_short_codes(d_codes.get(), d_short_raw.get(), D_seg, N, num_bits);

        // 5e. Pack long codes (per-vector)
        size_t long_code_bytes = (num_bits > 1) ? D_seg * (num_bits - 1) / 8 : 0;
        auto d_long_raw = device_alloc<uint8_t>(N * (long_code_bytes > 0 ? long_code_bytes : 1));
        launch_pack_long_codes(d_codes.get(), d_long_raw.get(), D_seg, N, num_bits);

        // 5f. Scatter to per-cluster GpuSaqCluData
        SAQ_CUDA_CHECK(cudaDeviceSynchronize());
        auto seg_kernel_ms = phase_timer.getElapsedTimeMicro() / 1000.0;
        total_kernel_ms += seg_kernel_ms;
        LOG(INFO) << "[TIMING] Segment " << seg << " kernels: " << seg_kernel_ms << " ms";

        phase_timer.reset();
        for (size_t c = 0; c < K; ++c) {
            size_t clu_size = cluster_sizes[c];
            if (clu_size == 0) continue;
            size_t clu_offset = cluster_offsets[c];
            auto& gpu_seg = gpu_clusters_[c].segments[seg];

            // Copy centroid
            SAQ_CUDA_CHECK(cudaMemcpy(
                gpu_seg.d_centroid,
                d_centroids_seg.get() + c * D_seg,
                D_seg * sizeof(float), cudaMemcpyDeviceToDevice));

            // Copy factors
            launch_store_factors(
                d_o_l2norm.get(), d_ip_cent_oa.get(),
                d_fac_rescale.get(), d_fac_error.get(),
                gpu_seg.d_factor_o_l2norm, gpu_seg.d_factor_ip_cent_oa,
                gpu_seg.d_long_factor_rescale, gpu_seg.d_long_factor_error,
                clu_size, clu_offset);

            // Copy short codes
            if (num_bits > 0 && gpu_seg.d_short_codes) {
                // TODO: fastscan reorder needed for GPU search
                // For now, copy raw short codes
                SAQ_CUDA_CHECK(cudaMemcpy(
                    gpu_seg.d_short_codes,
                    d_short_raw.get() + clu_offset * short_code_bytes,
                    clu_size * short_code_bytes, cudaMemcpyDeviceToDevice));
            }

            // Copy long codes
            if (long_code_bytes > 0 && gpu_seg.d_long_codes) {
                SAQ_CUDA_CHECK(cudaMemcpy(
                    gpu_seg.d_long_codes,
                    d_long_raw.get() + clu_offset * long_code_bytes,
                    clu_size * long_code_bytes, cudaMemcpyDeviceToDevice));
            }
        }

        SAQ_CUDA_CHECK(cudaDeviceSynchronize());
        auto seg_scatter_ms = phase_timer.getElapsedTimeMicro() / 1000.0;
        total_scatter_ms += seg_scatter_ms;
        LOG(INFO) << "[TIMING] Segment " << seg << " scatter: " << seg_scatter_ms << " ms";

        dim_offset += D_seg;
    }

    SAQ_CUDA_CHECK(cudaDeviceSynchronize());
    auto tm_ms = stopw.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "GPU IVF construction done. Time: " << tm_ms / 1e3 << " s";
    LOG(INFO) << "[TIMING] === Summary ===";
    LOG(INFO) << "[TIMING] CPU prep:       " << cpu_prep_ms << " ms";
    LOG(INFO) << "[TIMING] H2D upload:     " << upload_ms << " ms";
    LOG(INFO) << "[TIMING] GPU alloc:      " << alloc_ms << " ms";
    LOG(INFO) << "[TIMING] GPU kernels:    " << total_kernel_ms << " ms";
    LOG(INFO) << "[TIMING] GPU scatter:    " << total_scatter_ms << " ms";
    LOG(INFO) << "[TIMING] Kernel+scatter: " << (total_kernel_ms + total_scatter_ms) << " ms";
    LOG(INFO) << "[TIMING] Total wall:     " << tm_ms << " ms";
}

} // namespace saq::gpu

#endif // SAQ_USE_CUDA
