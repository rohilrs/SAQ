#ifdef SAQ_USE_CUDA

#include "saq/gpu/gpu_ivf.h"
#include "saq/gpu/gpu_utils.cuh"
#include "saq/gpu/gpu_searcher.cuh"
#include "saq/gpu/gpu_memory_pool.h"
#include "saq/initializer.h"
#include "saq/stopw.h"

#include <vector>
#include <algorithm>

#include <glog/logging.h>

namespace saq::gpu {

void GpuIVF::search_batch(const FloatRowMat& queries,
                           size_t topk, size_t nprobe,
                           SearcherConfig cfg,
                           PID* results,
                           float* out_dists) {
    const size_t Q = queries.rows();
    const size_t D = num_dim_;
    const size_t K = num_cen_;
    const auto& quant_plan = saq_data_->quant_plan;
    const auto& base_datas = saq_data_->base_datas;
    size_t num_segments = quant_plan.size();

    LOG(INFO) << "GPU batch search: Q=" << Q << " topk=" << topk << " nprobe=" << nprobe;
    StopW stopw;

    // 1. Find nprobe nearest centroids per query (GPU)
    auto d_queries_full = device_alloc<float>(Q * D);
    upload(d_queries_full.get(), queries.data(), Q * D);

    auto d_centroid_ids = device_alloc<uint32_t>(Q * nprobe);
    launch_batch_centroid_search(
        d_queries_full.get(), d_centroids_raw_.get(), d_centroid_ids.get(),
        Q, K, D, nprobe);
    SAQ_CUDA_CHECK(cudaDeviceSynchronize());

    // Download centroid IDs for host-side query prep
    std::vector<uint32_t> h_centroid_ids(Q * nprobe);
    download(h_centroid_ids.data(), d_centroid_ids.get(), Q * nprobe);

    auto centroid_ms = stopw.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "[SEARCH TIMING] Centroid search (GPU): " << centroid_ms << " ms";

    // 2. Rotate queries per segment and compute constants (CPU)
    size_t total_D_seg = 0;
    std::vector<size_t> seg_dim_offsets(num_segments);
    for (size_t s = 0; s < num_segments; ++s) {
        seg_dim_offsets[s] = total_D_seg;
        total_D_seg += quant_plan[s].first;
    }

    std::vector<float> h_rotated_queries(Q * total_D_seg);
    std::vector<QuerySegmentConstants> h_query_consts(Q * num_segments);

    for (size_t q = 0; q < Q; ++q) {
        size_t dim_offset = 0;
        for (size_t s = 0; s < num_segments; ++s) {
            size_t D_seg = quant_plan[s].first;
            const auto& bdata = base_datas[s];

            // Extract and rotate query segment
            FloatVec query_seg = queries.row(q).segment(dim_offset, D_seg);
            FloatVec rotated_seg;
            if (bdata.rotator) {
                rotated_seg = query_seg * bdata.rotator->get_P();
            } else {
                rotated_seg = query_seg;
            }

            // Copy to flat array
            float* dst = h_rotated_queries.data() + q * total_D_seg + seg_dim_offsets[s];
            for (size_t d = 0; d < D_seg; ++d)
                dst[d] = rotated_seg(d);

            // Compute LUT constants (same as CPU Lut::prepare but just the metadata)
            float sum_q = rotated_seg.sum();
            float q_l2sqr = rotated_seg.squaredNorm();
            float q_l2norm = std::sqrt(q_l2sqr);

            // Build float LUT to get delta and sum_vl_lut
            // pack_lut equivalent: compute subset sums for each codebook
            size_t num_codebooks = D_seg / 4;
            float vl = std::numeric_limits<float>::max();
            float vr = std::numeric_limits<float>::lowest();

            // We need the min/max of the LUT entries to compute delta
            for (size_t cb = 0; cb < num_codebooks; ++cb) {
                const float* q4 = dst + cb * 4;
                // 16 subset sums
                float lut[16];
                lut[0] = 0.0f;
                constexpr int kPos[16] = {3,3,2,3,1,3,2,3,0,3,2,3,1,3,2,3};
                for (int j = 1; j < 16; ++j) {
                    int lb = j & (-j);
                    lut[j] = lut[j - lb] + q4[kPos[j]];
                }
                for (int j = 0; j < 16; ++j) {
                    vl = std::min(vl, lut[j]);
                    vr = std::max(vr, lut[j]);
                }
            }

            // Note: for the GPU float LUT, delta/sum_vl_lut are not used for quantization
            // (we use float LUT directly). But we still need them for the fast distance formula.
            // Actually, since the GPU LUT stores raw float values (not quantized), the
            // distance formula needs to be adjusted. The raw LUT sum IS the approximate IP.
            // delta and sum_vl_lut are used in the CPU's quantized LUT path but not needed
            // when using float LUT. We store them for potential future use.

            size_t bits = quant_plan[s].second;

            auto& qc = h_query_consts[q * num_segments + s];
            qc.delta = (vr - vl) / (65535.0f - 0.01f);
            qc.sum_vl_lut = (vl + 0.5f * qc.delta) * (float)num_codebooks;
            qc.sum_q = sum_q;
            qc.q_l2sqr = q_l2sqr;
            qc.q_l2norm = q_l2norm;
            qc.one_over_sqrtD = 1.0f / std::sqrt((float)D_seg);
            qc.sq_delta = (bits > 0) ? 2.0f / (float)(1 << bits) : 0.0f;

            dim_offset += D_seg;
        }
    }
    auto prep_ms = stopw.getElapsedTimeMicro() / 1000.0 - centroid_ms;
    LOG(INFO) << "[SEARCH TIMING] Query prep: " << prep_ms << " ms";

    // 3. Build and upload descriptor tables
    std::vector<GpuSegmentDescriptor> h_seg_descs(num_segments);
    for (size_t s = 0; s < num_segments; ++s) {
        auto& sd = h_seg_descs[s];
        const auto& sp = pool_.segments[s];
        sd.short_codes = sp.short_codes.get();
        sd.long_codes = sp.long_codes.get();
        sd.factor_o_l2norm = sp.factor_o_l2norm.get();
        sd.factor_ip_cent_oa = sp.factor_ip_cent_oa.get();
        sd.factor_rescale = sp.factor_rescale.get();
        sd.factor_error = sp.factor_error.get();
        sd.centroids = sp.centroids.get();
        sd.num_codebooks = quant_plan[s].first / 4;
        sd.D_seg = quant_plan[s].first;
        sd.num_bits = quant_plan[s].second;
        sd.long_bytes_per_vec = (quant_plan[s].second > 1)
            ? quant_plan[s].first * (quant_plan[s].second - 1) / 8 : 0;

        // Codebook pointers (nullptr if no codebook for this segment)
        sd.codebook_centroids = sp.codebook_centroids.get();
        sd.codebook_entries_per_dim = sp.codebook_entries_per_dim;
    }

    std::vector<GpuClusterDescriptor> h_clu_descs(K);
    for (size_t c = 0; c < K; ++c) {
        h_clu_descs[c].num_vec = gpu_clusters_[c].num_vec;
        h_clu_descs[c].num_blocks = gpu_clusters_[c].num_blocks;
        h_clu_descs[c].ids = pool_.ids.get() + pool_.cluster_offsets[c];
    }

    // Upload everything to device
    auto d_seg_descs = device_alloc<GpuSegmentDescriptor>(num_segments);
    upload(d_seg_descs.get(), h_seg_descs.data(), num_segments);

    auto d_clu_descs = device_alloc<GpuClusterDescriptor>(K);
    upload(d_clu_descs.get(), h_clu_descs.data(), K);

    auto d_rotated_queries = device_alloc<float>(Q * total_D_seg);
    upload(d_rotated_queries.get(), h_rotated_queries.data(), Q * total_D_seg);

    auto d_query_consts = device_alloc<QuerySegmentConstants>(Q * num_segments);
    upload(d_query_consts.get(), h_query_consts.data(), Q * num_segments);

    // d_centroid_ids already on GPU from centroid search step

    // Allocate output buffers
    size_t cand_buf_size = Q * nprobe * kMaxCandidatesPerBlock;
    auto d_candidate_dists = device_alloc<float>(cand_buf_size);
    auto d_candidate_ids = device_alloc<uint32_t>(cand_buf_size);
    auto d_candidate_counts = device_alloc<uint32_t>(Q * nprobe);
    auto d_results = device_alloc<uint32_t>(Q * topk);
    auto d_results_dists = device_alloc<float>(Q * topk);

    // Workspace for merge kernel
    size_t max_total_cands = nprobe * kMaxCandidatesPerBlock;
    auto d_work_dists = device_alloc<float>(Q * max_total_cands);
    auto d_work_ids = device_alloc<uint32_t>(Q * max_total_cands);

    SAQ_CUDA_CHECK(cudaDeviceSynchronize());
    auto upload_ms = stopw.getElapsedTimeMicro() / 1000.0 - centroid_ms - prep_ms;
    LOG(INFO) << "[SEARCH TIMING] Upload: " << upload_ms << " ms";

    // 4. Launch search kernel
    StopW kernel_timer;
    launch_search(
        d_seg_descs.get(), d_clu_descs.get(),
        pool_.d_block_offsets.get(), pool_.d_cluster_offsets.get(),
        d_rotated_queries.get(), d_query_consts.get(), d_centroid_ids.get(),
        Q, nprobe, topk, num_segments, total_D_seg,
        d_candidate_dists.get(), d_candidate_ids.get(), d_candidate_counts.get());

    SAQ_CUDA_CHECK(cudaDeviceSynchronize());
    auto search_ms = kernel_timer.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "[SEARCH TIMING] Search kernel: " << search_ms << " ms";

    // 5. Launch merge kernel
    kernel_timer.reset();
    launch_merge_topk(
        d_candidate_dists.get(), d_candidate_ids.get(), d_candidate_counts.get(),
        d_work_dists.get(), d_work_ids.get(),
        d_results.get(), d_results_dists.get(), Q, nprobe, topk, max_total_cands);

    SAQ_CUDA_CHECK(cudaDeviceSynchronize());
    auto merge_ms = kernel_timer.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "[SEARCH TIMING] Merge kernel: " << merge_ms << " ms";

    // 6. Download results
    std::vector<uint32_t> h_results(Q * topk);
    download(h_results.data(), d_results.get(), Q * topk);

    for (size_t i = 0; i < Q * topk; ++i) {
        results[i] = static_cast<PID>(h_results[i]);
    }

    if (out_dists != nullptr) {
        download(out_dists, d_results_dists.get(), Q * topk);
    }

    auto total_ms = stopw.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "[SEARCH TIMING] Total: " << total_ms << " ms"
              << " (" << (Q * 1000.0 / total_ms) << " QPS)";
}

} // namespace saq::gpu

#endif // SAQ_USE_CUDA
