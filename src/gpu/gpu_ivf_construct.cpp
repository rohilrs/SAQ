#ifdef SAQ_USE_CUDA

#include <cstdlib>
#include "saq/gpu/gpu_ivf.h"
#include "saq/gpu/gpu_utils.cuh"
#include "saq/gpu/gpu_encoder.cuh"
#include "saq/gpu/gpu_scatter.cuh"
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

void GpuIVF::set_quant_plan(SaqData::QuantPlanT plan) {
    saq_data_maker_->set_quant_plan(std::move(plan));
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

    // Heterogeneous byte layout: validate, then build the pattern-sorted
    // column metadata the pack and decode use. Pattern key = offset-ordered
    // field widths packed 4 bits each (MSB-side field in the low nibble).
    std::vector<uint16_t> h_target_col, h_slot_dims;
    std::vector<uint8_t> h_shift, h_width;
    std::vector<uint32_t> h_runs;
    if (has_byte_layout_) {
        CHECK(has_codebooks_) << "byte layout requires codebook mode";
        CHECK_EQ(num_segments, 1u) << "byte layout requires a single-segment plan";
        CHECK_EQ(quant_plan[0].first, D) << "byte layout plan must cover all dims";
        CHECK_EQ(layout_widths_.size(), D);
        layout_n_bytes_ = 0;
        for (size_t j = 0; j < D; ++j)
            if (layout_byte_idx_[j] >= 0)
                layout_n_bytes_ = std::max(layout_n_bytes_, (size_t)layout_byte_idx_[j] + 1);
        // fields per original byte, offset-ordered
        std::vector<std::vector<std::pair<int, int>>> fields(layout_n_bytes_); // (off, dim)
        for (size_t j = 0; j < D; ++j)
            if (layout_widths_[j] > 0)
                fields[layout_byte_idx_[j]].push_back({layout_bit_off_[j], (int)j});
        std::vector<uint32_t> pat(layout_n_bytes_, 0);
        for (size_t t = 0; t < layout_n_bytes_; ++t) {
            std::sort(fields[t].begin(), fields[t].end());
            uint32_t key = 0;
            for (size_t k = 0; k < fields[t].size(); ++k)
                key |= (uint32_t)layout_widths_[fields[t][k].second] << (4 * k);
            pat[t] = key;
        }
        std::vector<size_t> col_order(layout_n_bytes_);
        std::iota(col_order.begin(), col_order.end(), 0u);
        std::stable_sort(col_order.begin(), col_order.end(),
                         [&](size_t a, size_t b) { return pat[a] < pat[b]; });
        std::vector<uint16_t> new_col(layout_n_bytes_);
        for (size_t nc = 0; nc < layout_n_bytes_; ++nc)
            new_col[col_order[nc]] = (uint16_t)nc;
        h_target_col.resize(D, 0); h_shift.resize(D, 0); h_width.resize(D, 0);
        for (size_t j = 0; j < D; ++j) {
            h_width[j] = (uint8_t)layout_widths_[j];
            if (layout_widths_[j] > 0) {
                h_target_col[j] = new_col[layout_byte_idx_[j]];
                h_shift[j] = (uint8_t)(8 - layout_bit_off_[j] - layout_widths_[j]);
            }
        }
        for (size_t nc = 0; nc < layout_n_bytes_; ++nc) {
            size_t t = col_order[nc];
            for (auto& f : fields[t]) h_slot_dims.push_back((uint16_t)f.second);
            if (h_runs.size() >= 2 && h_runs[h_runs.size() - 2] == pat[t])
                h_runs.back() += 1;
            else { h_runs.push_back(pat[t]); h_runs.push_back(1); }
        }
        n_layout_runs_ = h_runs.size() / 2;
        LOG(INFO) << "Byte layout: " << layout_n_bytes_ << " bytes/vec, "
                  << h_slot_dims.size() << " slots, " << n_layout_runs_
                  << " pattern runs";
        d_layout_target_col_ = device_alloc<uint16_t>(D);
        d_layout_shift_ = device_alloc<uint8_t>(D);
        d_layout_widths_ = device_alloc<uint8_t>(D);
        d_layout_slot_dims_ = device_alloc<uint16_t>(h_slot_dims.size());
        d_layout_runs_ = device_alloc<uint32_t>(h_runs.size());
        upload(d_layout_target_col_.get(), h_target_col.data(), D);
        upload(d_layout_shift_.get(), h_shift.data(), D);
        upload(d_layout_widths_.get(), h_width.data(), D);
        upload(d_layout_slot_dims_.get(), h_slot_dims.data(), h_slot_dims.size());
        upload(d_layout_runs_.get(), h_runs.data(), h_runs.size());
    }

    // Build segment codebooks if codebook mode is active
    if (has_codebooks_) {
        // Codebooks are trained/injected in the unrotated (PCA) basis; encoding
        // through a per-segment random rotation would quantize rotated data
        // with mismatched codebooks (measured: error floor ~8x CAQ's at 8 bpd,
        // recall below the method's own 4-bpd result). Disable random_rotation
        // in codebook mode.
        for (const auto& bdata : base_datas) {
            CHECK(!bdata.rotator) << "codebook mode requires random_rotation=false: "
                "segment rotators would encode rotated data against unrotated codebooks";
        }
        // Native data-driven derivation: build per-dim Lloyd codebooks from the
        // (PCA-transformed) data, then select each dimension's codebook at the
        // bit-count its segment was allocated in quant_plan.
        if (derive_codebooks_ && codebooks_explicit_.empty()) {
            std::vector<CodebookResult> per_dim = build_all_dims(data, lloyd_opts_);

            // Stash per-dim costs for the future allocation sub-project.
            saq_data_->codebook_costs.resize(per_dim.size());
            for (size_t d = 0; d < per_dim.size(); ++d) {
                saq_data_->codebook_costs[d] = per_dim[d].costs;
            }

            // Assemble explicit codebooks[seg][dim_in_seg] at each segment's bits.
            codebooks_explicit_.clear();
            codebooks_explicit_.resize(quant_plan.size());
            size_t gdim = 0;
            for (size_t s = 0; s < quant_plan.size(); ++s) {
                const size_t dim_len = quant_plan[s].first;
                codebooks_explicit_[s].reserve(dim_len);
                for (size_t j = 0; j < dim_len; ++j, ++gdim) {
                    // Heterogeneous layout: each dim at its own width;
                    // otherwise the segment's uniform width.
                    const size_t bits = has_byte_layout_
                        ? (size_t)layout_widths_[gdim] : quant_plan[s].second;
                    CHECK_LT(gdim, per_dim.size());
                    CHECK_LT(bits, per_dim[gdim].codebooks.size())
                        << "lloyd_opts_.max_bits too small for allocated bits=" << bits;
                    codebooks_explicit_[s].push_back(per_dim[gdim].codebooks[bits]);
                }
            }
        }
        saq_data_->segment_codebooks = build_segment_codebooks(
            quant_plan, codebooks_explicit_,
            gaussian_codebook_centroids_, residual_stds_);
        LOG(INFO) << "Built codebooks for " << num_segments << " segments"
                  << (derive_codebooks_ ? " (native Lloyd)" : "");
    }

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
    d_centroids_raw_ = device_alloc<float>(K * D);  // kept as member for GPU search
    auto d_cluster_ids = device_alloc<uint32_t>(N);

    upload(d_vectors.get(), sorted_data.data(), N * D);
    upload(d_centroids_raw_.get(), centroids.data(), K * D);
    upload(d_cluster_ids.get(), h_sorted_cids.data(), N);
    SAQ_CUDA_CHECK(cudaDeviceSynchronize());
    auto upload_ms = phase_timer.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "[TIMING] H2D upload (vectors+centroids+cids): " << upload_ms << " ms";

    // 4. Allocate GPU memory via pool
    phase_timer.reset();
    pool_ = GpuMemoryPool{};  // reset if previously used
    pool_.allocate(K, cluster_sizes, quant_plan, /*fused_codes_eligible=*/has_codebooks_,
                   /*fused_bpv_override=*/has_byte_layout_ ? layout_n_bytes_ : 0);

    // Set up cluster views
    gpu_clusters_.clear();
    gpu_clusters_.resize(K);
    for (size_t c = 0; c < K; ++c) {
        pool_.assign_pointers(gpu_clusters_[c], c);
    }

    // Upload original IDs into pool
    upload(pool_.ids.get(), h_sorted_original_ids.data(), N);

    // Upload codebooks to GPU
    if (has_codebooks_ && !saq_data_->segment_codebooks.empty()) {
        for (size_t s = 0; s < num_segments; ++s) {
            const auto& seg_cbs = saq_data_->segment_codebooks[s];
            if (!seg_cbs.empty()) {
                if (has_byte_layout_)
                    pool_.upload_segment_codebooks_padded(s, seg_cbs, 256);
                else
                    pool_.upload_segment_codebooks(s, seg_cbs);
            }
        }
        LOG(INFO) << "Uploaded codebooks to GPU";
    }

    SAQ_CUDA_CHECK(cudaDeviceSynchronize());
    auto alloc_ms = phase_timer.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "[TIMING] GPU pool alloc + ID upload: " << alloc_ms << " ms";

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

        // Allocate outputs
        size_t short_code_bytes = D_seg / 8;
        size_t long_code_bytes = (num_bits > 1) ? D_seg * (num_bits - 1) / 8 : 0;
        size_t short_alloc = N * (short_code_bytes > 0 ? short_code_bytes : 1);
        size_t long_alloc = N * (long_code_bytes > 0 ? long_code_bytes : 1);
        auto d_short_raw = device_alloc<uint8_t>(short_alloc);
        auto d_long_raw = device_alloc<uint8_t>(long_alloc);
        // Zero buffers — encode kernels use atomicOr to set bits
        SAQ_CUDA_CHECK(cudaMemset(d_short_raw.get(), 0, short_alloc));
        SAQ_CUDA_CHECK(cudaMemset(d_long_raw.get(), 0, long_alloc));
        auto d_o_l2norm = device_alloc<float>(N);
        auto d_fac_rescale = device_alloc<float>(N);
        auto d_fac_error = device_alloc<float>(N);
        auto d_ip_cent_oa = device_alloc<float>(N);

        uint16_t code_max = (1 << num_bits) - 1;
        if (bdata.cfg.caq_ori_qB) {
            code_max = (1 << bdata.cfg.caq_ori_qB) - 1;
        }

        if (bdata.rotator) {
            // L1: GEMM on raw vector segment, fused encode subtracts rotated centroid
            auto d_vec_seg = device_alloc<float>(N * D_seg);
            {
                float alpha = 1.0f, beta = 0.0f;
                SAQ_CUBLAS_CHECK(cublasSgeam(cublas.get(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)D_seg, (int)N,
                    &alpha, d_vectors.get() + dim_offset, (int)D,
                    &beta,  d_vec_seg.get(), (int)D_seg,
                    d_vec_seg.get(), (int)D_seg));
            }

            auto d_rotated = device_alloc<float>(N * D_seg);
            auto d_P = device_alloc<float>(D_seg * D_seg);
            upload(d_P.get(), bdata.rotator->get_P().data(), D_seg * D_seg);
            {
                float alpha = 1.0f, beta = 0.0f;
                SAQ_CUBLAS_CHECK(cublasSgemm(cublas.get(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)D_seg, (int)N, (int)D_seg,
                    &alpha, d_P.get(), (int)D_seg,
                    d_vec_seg.get(), (int)D_seg,
                    &beta, d_rotated.get(), (int)D_seg));
            }

            auto d_rotated_centroids = device_alloc<float>(K * D_seg);
            FloatRowMat cent_seg(K, D_seg);
            for (size_t c = 0; c < K; ++c)
                cent_seg.row(c) = centroids.row(c).segment(dim_offset, D_seg);
            FloatRowMat cent_rotated = cent_seg * bdata.rotator->get_P();
            upload(d_rotated_centroids.get(), cent_rotated.data(), K * D_seg);

            if (has_codebooks_ && pool_.segments[seg].codebook_entries_per_dim > 0) {
                launch_fused_codebook_encode(
                    d_rotated.get(), d_rotated_centroids.get(), d_cluster_ids.get(),
                    pool_.segments[seg].codebook_centroids.get(),
                    pool_.segments[seg].codebook_entries_per_dim,
                    d_o_l2norm.get(), d_fac_rescale.get(), d_fac_error.get(), d_ip_cent_oa.get(),
                    d_short_raw.get(), d_long_raw.get(),
                    D_seg, N, K, num_bits, code_max,
                    bdata.cfg.caq_adj_rd_lmt, bdata.cfg.caq_adj_eps);
            } else {
                // SAQ_GPU_CAQ_SEQUENTIAL=1 -> faithful sequential Gauss-Seidel CAQ that
                // reproduces the reference CPU encoder (block-Jacobi otherwise).
                static const int kCaqSeq = (std::getenv("SAQ_GPU_CAQ_SEQUENTIAL") != nullptr);
                launch_fused_caq_encode(
                    d_rotated.get(), d_rotated_centroids.get(), d_cluster_ids.get(),
                    d_o_l2norm.get(), d_fac_rescale.get(), d_fac_error.get(), d_ip_cent_oa.get(),
                    d_short_raw.get(), d_long_raw.get(),
                    D_seg, N, K, num_bits, code_max,
                    bdata.cfg.caq_adj_rd_lmt, bdata.cfg.caq_adj_eps, bdata.cfg.caq_ori_qB,
                    kCaqSeq);
            }
        } else {
            // No rotation: fused encode subtracts centroid inline from raw vectors
            if (has_codebooks_ && pool_.segments[seg].codebook_entries_per_dim > 0) {
                // For no-rotation codebook path, extract segment and use the rotation variant
                // with identity (the vectors are already in the right space)
                auto d_vec_seg = device_alloc<float>(N * D_seg);
                {
                    float alpha = 1.0f, beta = 0.0f;
                    SAQ_CUBLAS_CHECK(cublasSgeam(cublas.get(),
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)D_seg, (int)N,
                        &alpha, d_vectors.get() + dim_offset, (int)D,
                        &beta,  d_vec_seg.get(), (int)D_seg,
                        d_vec_seg.get(), (int)D_seg));
                }
                auto d_cent_seg = device_alloc<float>(K * D_seg);
                FloatRowMat cent_seg(K, D_seg);
                for (size_t c = 0; c < K; ++c)
                    cent_seg.row(c) = centroids.row(c).segment(dim_offset, D_seg);
                upload(d_cent_seg.get(), cent_seg.data(), K * D_seg);

                if (has_byte_layout_) {
                    // Heterogeneous widths: per-dim nearest-codeword encode to
                    // a transient code array; bytes are packed after scatter.
                    if (!d_het_codes_) d_het_codes_ = device_alloc<uint8_t>(N * D_seg);
                    launch_het_codebook_encode(
                        d_vec_seg.get(), d_cent_seg.get(), d_cluster_ids.get(),
                        pool_.segments[seg].codebook_centroids.get(),
                        d_layout_widths_.get(),
                        pool_.segments[seg].codebook_entries_per_dim,
                        d_o_l2norm.get(), d_fac_rescale.get(), d_fac_error.get(), d_ip_cent_oa.get(),
                        d_het_codes_.get(),
                        D_seg, N, K,
                        bdata.cfg.caq_adj_rd_lmt, bdata.cfg.caq_adj_eps);
                } else {
                launch_fused_codebook_encode(
                    d_vec_seg.get(), d_cent_seg.get(), d_cluster_ids.get(),
                    pool_.segments[seg].codebook_centroids.get(),
                    pool_.segments[seg].codebook_entries_per_dim,
                    d_o_l2norm.get(), d_fac_rescale.get(), d_fac_error.get(), d_ip_cent_oa.get(),
                    d_short_raw.get(), d_long_raw.get(),
                    D_seg, N, K, num_bits, code_max,
                    bdata.cfg.caq_adj_rd_lmt, bdata.cfg.caq_adj_eps);
                }
            } else {
                launch_fused_caq_encode_no_rotation(
                    d_vectors.get(), d_centroids_raw_.get(), d_cluster_ids.get(),
                    dim_offset, D_seg, D,
                    d_o_l2norm.get(), d_fac_rescale.get(), d_fac_error.get(), d_ip_cent_oa.get(),
                    d_short_raw.get(), d_long_raw.get(),
                    N, K, num_bits, code_max,
                    bdata.cfg.caq_adj_rd_lmt, bdata.cfg.caq_adj_eps, bdata.cfg.caq_ori_qB);
            }
        }

        // Compute rotated centroids for scatter (needed for pool centroid storage)
        auto d_centroids_seg = device_alloc<float>(K * D_seg);
        if (bdata.rotator) {
            FloatRowMat cent_seg(K, D_seg);
            for (size_t c = 0; c < K; ++c)
                cent_seg.row(c) = centroids.row(c).segment(dim_offset, D_seg);
            FloatRowMat cent_rotated = cent_seg * bdata.rotator->get_P();
            upload(d_centroids_seg.get(), cent_rotated.data(), K * D_seg);
        } else {
            FloatRowMat cent_seg(K, D_seg);
            for (size_t c = 0; c < K; ++c)
                cent_seg.row(c) = centroids.row(c).segment(dim_offset, D_seg);
            upload(d_centroids_seg.get(), cent_seg.data(), K * D_seg);
        }

        // 5f. Scatter to per-cluster GpuSaqCluData
        SAQ_CUDA_CHECK(cudaDeviceSynchronize());
        auto seg_kernel_ms = phase_timer.getElapsedTimeMicro() / 1000.0;
        total_kernel_ms += seg_kernel_ms;
        LOG(INFO) << "[TIMING] Segment " << seg << " kernels: " << seg_kernel_ms << " ms";

        phase_timer.reset();

        // Scatter centroids (simple D2D memcpy — same contiguous layout)
        copy_centroids_to_pool(
            d_centroids_seg.get(), pool_.segments[seg].centroids.get(),
            D_seg, K);

        // Scatter factors (blocked + per-vector layout)
        launch_scatter_factors(
            d_o_l2norm.get(), d_ip_cent_oa.get(),
            d_fac_rescale.get(), d_fac_error.get(),
            pool_.segments[seg].factor_o_l2norm.get(),
            pool_.segments[seg].factor_ip_cent_oa.get(),
            pool_.segments[seg].factor_rescale.get(),
            pool_.segments[seg].factor_error.get(),
            pool_.d_cluster_offsets.get(), pool_.d_block_offsets.get(),
            d_cluster_ids.get(), N);

        // Scatter short codes (with fastscan reorder to GPU blocked layout)
        if (num_bits > 0) {
            launch_scatter_short_codes(
                d_short_raw.get(), pool_.segments[seg].short_codes.get(),
                pool_.d_cluster_offsets.get(), pool_.d_block_offsets.get(),
                d_cluster_ids.get(), D_seg, N, num_bits);
        }

        // Scatter long codes
        if (long_code_bytes > 0) {
            launch_scatter_long_codes(
                d_long_raw.get(), pool_.segments[seg].long_codes.get(),
                pool_.d_cluster_offsets.get(), pool_.d_block_offsets.get(),
                d_cluster_ids.get(), long_code_bytes, N);
        }

        // Heterogeneous layout: pack per-dim codes into the fused byte layout.
        if (has_byte_layout_ && seg == 0) {
            launch_het_pack(
                d_het_codes_.get(), pool_.segments[seg].fused_codes.get(),
                d_layout_target_col_.get(), d_layout_shift_.get(), d_layout_widths_.get(),
                pool_.d_cluster_offsets.get(), pool_.d_block_offsets.get(),
                d_cluster_ids.get(), N, D_seg, layout_n_bytes_);
        }

        // Byte-fused repack (codebook mode, width dividing 8): byte-aligned
        // codes for the fused decode path, from the just-scattered streams.
        if (!has_byte_layout_ && pool_.segments[seg].fused_bytes_per_vec > 0) {
            size_t ncb_groups = ((D_seg / 4) + 3) / 4;
            size_t lwpv = (long_code_bytes + 3) / 4;
            launch_fused_repack(
                pool_.segments[seg].short_codes.get(),
                pool_.segments[seg].long_codes.get(),
                pool_.segments[seg].fused_codes.get(),
                pool_.total_blocks_, D_seg, num_bits, ncb_groups, lwpv);
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
