#include "saq/gpu/gpu_searcher.cuh"
#include "saq/gpu/gpu_utils.cuh"

#include <cfloat>

namespace saq::gpu {

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float warp_reduce_min(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

/// Build LUT for one codebook (4 query dims → 16 entries via subset sums).
/// Same as CPU pack_lut: LUT[j] = LUT[j - lowbit(j)] + query[kPos[j]]
__device__ void build_codebook_lut(const float* query4, float* lut16) {
    // kPos maps 4-bit pattern to which query dim to add
    constexpr int kPos[16] = {3,3,2,3,1,3,2,3,0,3,2,3,1,3,2,3};
    lut16[0] = 0.0f;
    for (int j = 1; j < 16; ++j) {
        int lb = j & (-j);  // lowbit
        lut16[j] = lut16[j - lb] + query4[kPos[j]];
    }
}

/// Unpack and compute IP between query and variable-bit long code.
/// Long codes store (num_bits-1) bits per dim, bit-compacted.
__device__ float gpu_long_code_ip(
    const float* query_seg,
    const uint8_t* long_code,
    size_t D_seg, size_t num_bits)
{
    if (num_bits <= 1) return 0.0f;

    size_t ex_bits = num_bits - 1;
    float ip = 0.0f;

    for (size_t d = 0; d < D_seg; ++d) {
        // Extract ex_bits starting at bit position d * ex_bits
        size_t bit_offset = d * ex_bits;
        int code_val = 0;
        for (size_t b = 0; b < ex_bits; ++b) {
            size_t global_bit = bit_offset + b;
            size_t byte_pos = global_bit / 8;
            size_t bit_pos = global_bit % 8;
            if ((long_code[byte_pos] >> bit_pos) & 1)
                code_val |= (1 << b);
        }
        ip += query_seg[d] * (float)code_val;
    }
    return ip;
}

// ============================================================================
// Main search kernel: build LUT + 3-stage search
// Grid: dim3(Q, nprobe), Block: 128 threads (4 warps)
// ============================================================================

__global__ void kernel_search(
    const GpuSegmentDescriptor* __restrict__ d_seg_descs,
    const GpuClusterDescriptor* __restrict__ d_clu_descs,
    const uint32_t* __restrict__ d_block_offsets,
    const uint32_t* __restrict__ d_cluster_offsets,
    const float* __restrict__ d_rotated_queries,
    const QuerySegmentConstants* __restrict__ d_query_consts,
    const uint32_t* __restrict__ d_centroid_ids,
    size_t Q, size_t nprobe, size_t topk,
    size_t num_segments, size_t total_D_seg,
    float* __restrict__ d_candidate_dists,
    uint32_t* __restrict__ d_candidate_ids,
    uint32_t* __restrict__ d_candidate_counts)
{
    // Dynamic shared memory layout:
    // [0..lut_bytes): int16 LUT entries
    // [lut_bytes..lut_bytes+const_bytes): QuerySegmentConstants per segment
    // [after constants): work counter (int)
    extern __shared__ char smem_raw[];

    size_t q_idx = blockIdx.x;
    size_t cluster_rank = blockIdx.y;
    if (q_idx >= Q || cluster_rank >= nprobe) return;

    uint32_t c = d_centroid_ids[q_idx * nprobe + cluster_rank];
    const auto& clu = d_clu_descs[c];
    if (clu.num_blocks == 0) {
        if (threadIdx.x == 0)
            d_candidate_counts[q_idx * nprobe + cluster_rank] = 0;
        return;
    }

    // Compute shared memory layout
    size_t total_codebooks = 0;
    size_t seg_cb_offsets[8]; // max 8 segments
    size_t seg_dim_offsets[8];
    size_t dim_offset = 0;
    for (size_t s = 0; s < num_segments; ++s) {
        seg_cb_offsets[s] = total_codebooks;
        seg_dim_offsets[s] = dim_offset;
        total_codebooks += d_seg_descs[s].num_codebooks;
        dim_offset += d_seg_descs[s].D_seg;
    }

    // Shared memory layout:
    // [0..total_codebooks*16): float LUT (24KB for D=1536)
    // [after LUT): per-segment constants (7 floats per segment)
    // [after consts): work-stealing counter
    // [after counter): per-segment residual query (for stage 3 accurate distance)
    constexpr int kConstsPerSeg = 7;  // delta, sum_vl_lut, sum_q_resid, q_l2sqr_resid, q_l2norm_resid, one_over_sqrtD, sq_delta
    float* smem_lut_f = (float*)smem_raw;
    float* smem_consts_f = smem_lut_f + total_codebooks * 16;
    int* smem_work = (int*)(smem_consts_f + num_segments * kConstsPerSeg);
    float* smem_resid_query = (float*)((char*)(smem_work + 1));
    // smem_resid_query: [total_D_seg] floats — residual query = rotated_query - centroid

    // ---- Phase 0: Build LUT from (query - centroid) ----
    // For L2 distance, the CPU builds LUT from the residual query.
    // We subtract the cluster's centroid per segment.
    for (size_t s = 0; s < num_segments; ++s) {
        const auto& seg = d_seg_descs[s];
        const auto& qc = d_query_consts[q_idx * num_segments + s];
        const float* query_seg = d_rotated_queries + q_idx * total_D_seg + seg_dim_offsets[s];
        const float* centroid_seg = seg.centroids + c * seg.D_seg;

        // Cooperatively compute residual query = query - centroid, store in shared memory
        float* resid_seg = smem_resid_query + seg_dim_offsets[s];
        for (size_t d = threadIdx.x; d < seg.D_seg; d += blockDim.x) {
            resid_seg[d] = query_seg[d] - centroid_seg[d];
        }
    }
    __syncthreads();

    // Now build LUT from residual query and compute per-segment constants
    for (size_t s = 0; s < num_segments; ++s) {
        const auto& seg = d_seg_descs[s];
        const auto& qc = d_query_consts[q_idx * num_segments + s];
        float* resid_seg = smem_resid_query + seg_dim_offsets[s];

        // Build LUT from residual query
        for (size_t cb = threadIdx.x; cb < seg.num_codebooks; cb += blockDim.x) {
            float* dst = smem_lut_f + (seg_cb_offsets[s] + cb) * 16;
            build_codebook_lut(resid_seg + cb * 4, dst);
        }

        // Compute residual query constants (thread 0 does sequential sum)
        if (threadIdx.x == 0) {
            float sum_q_resid = 0.0f;
            float q_l2sqr_resid = 0.0f;
            for (size_t d = 0; d < seg.D_seg; ++d) {
                sum_q_resid += resid_seg[d];
                q_l2sqr_resid += resid_seg[d] * resid_seg[d];
            }
            float q_l2norm_resid = sqrtf(q_l2sqr_resid);

            smem_consts_f[s * kConstsPerSeg + 0] = qc.delta;
            smem_consts_f[s * kConstsPerSeg + 1] = qc.sum_vl_lut;
            smem_consts_f[s * kConstsPerSeg + 2] = sum_q_resid;
            smem_consts_f[s * kConstsPerSeg + 3] = q_l2sqr_resid;
            smem_consts_f[s * kConstsPerSeg + 4] = q_l2norm_resid;
            smem_consts_f[s * kConstsPerSeg + 5] = qc.one_over_sqrtD;
            smem_consts_f[s * kConstsPerSeg + 6] = qc.sq_delta;
        }
    }

    if (threadIdx.x == 0) *smem_work = 0;
    __syncthreads();

    // ---- Phase 1: Compute accurate distance for ALL vectors ----
    // Skip stage 2 fast distance filtering. Compute full accurate distance
    // using LUT (short codes) + long code IP for every valid vector.
    // This guarantees correct recall at the cost of computing more distances.
    int lane = threadIdx.x % 32;

    // Per-warp candidate buffer
    constexpr int kWarpMaxCandidates = 64;
    float warp_cand_dists[kWarpMaxCandidates];
    uint32_t warp_cand_ids[kWarpMaxCandidates];
    int warp_cand_count = 0;
    float distk = FLT_MAX;

    uint32_t blk_off_c = d_block_offsets[c];

    while (true) {
        // Claim next block (work-stealing)
        int block_idx;
        if (lane == 0) block_idx = atomicAdd(smem_work, 1);
        block_idx = __shfl_sync(0xFFFFFFFF, block_idx, 0);
        if ((size_t)block_idx >= clu.num_blocks) break;

        uint32_t global_block = blk_off_c + block_idx;
        uint32_t vec_pos = block_idx * 32 + lane;
        bool valid_vec = (vec_pos < clu.num_vec);

        // Compute accurate distance for this vector across all segments
        float acc_dist = 0.0f;
        if (valid_vec) {
            uint32_t vec_offset = d_cluster_offsets[c] + vec_pos;

            for (size_t s = 0; s < num_segments; ++s) {
                const auto& seg = d_seg_descs[s];
                float q_l2sqr_s = smem_consts_f[s * kConstsPerSeg + 3];
                float o_l2n = seg.factor_o_l2norm[global_block * 32 + lane];
                float o_l2sqr = o_l2n * o_l2n;

                if (seg.num_bits == 0) {
                    // Zero-bit: distance = o_l2sqr + q_l2sqr (no IP approximation)
                    acc_dist += o_l2sqr + q_l2sqr_s;
                    continue;
                }

                float sum_q_s = smem_consts_f[s * kConstsPerSeg + 2];
                float sq_delta_s = smem_consts_f[s * kConstsPerSeg + 6];

                // LUT sum: approximate IP from 1-bit (short) codes
                float lut_sum = 0.0f;
                {
                    const uint8_t* short_base = seg.short_codes
                        + (size_t)global_block * 32 * seg.num_codebooks;
                    const uint8_t* my_codes = short_base + lane * seg.num_codebooks;
                    for (size_t cb = 0; cb < seg.num_codebooks; ++cb) {
                        lut_sum += smem_lut_f[(seg_cb_offsets[s] + cb) * 16 + my_codes[cb]];
                    }
                }

                float rescale = seg.factor_rescale[vec_offset];
                float full_ip;

                if (seg.num_bits > 1 && seg.long_bytes_per_vec > 0) {
                    // Full IP using long codes
                    const float* resid_seg = smem_resid_query + seg_dim_offsets[s];
                    const uint8_t* long_code = seg.long_codes
                        + vec_offset * seg.long_bytes_per_vec;
                    float ext_ip = gpu_long_code_ip(resid_seg, long_code,
                                                     seg.D_seg, seg.num_bits);
                    full_ip = lut_sum + ext_ip * sq_delta_s
                            + (-1.0f + sq_delta_s / 2.0f) * sum_q_s;
                } else {
                    // 1-bit or no long codes: still apply the bias term
                    // ext_ip = 0, so full_ip = lut_sum + 0 + (vl + sq_delta/2) * sum_q
                    full_ip = lut_sum + (-1.0f + sq_delta_s / 2.0f) * sum_q_s;
                }

                float ip_o_q = rescale * full_ip;
                float seg_dist = o_l2sqr + q_l2sqr_s - 2.0f * ip_o_q;
                acc_dist += seg_dist;

                // Debug output removed for commit
            }
        }

        float my_dist = valid_vec ? fmaxf(0.0f, acc_dist) : FLT_MAX;

        // All valid vectors are candidates — collect via shuffle
        uint32_t valid_mask = __ballot_sync(0xFFFFFFFF, valid_vec);

        for (int src_lane = 0; src_lane < 32; ++src_lane) {
            if (!(valid_mask & (1u << src_lane))) continue;

            float dist_val = __shfl_sync(0xFFFFFFFF, my_dist, src_lane);
            if (dist_val >= distk) continue;

            if (lane == 0 && warp_cand_count < kWarpMaxCandidates) {
                warp_cand_dists[warp_cand_count] = dist_val;
                uint32_t vec_p = block_idx * 32 + src_lane;
                warp_cand_ids[warp_cand_count] = clu.ids[vec_p];
                warp_cand_count++;

                // Evict worst when buffer full
                if (warp_cand_count >= kWarpMaxCandidates) {
                    int worst_idx = 0;
                    float worst_dist = warp_cand_dists[0];
                    for (int k = 1; k < warp_cand_count; ++k) {
                        if (warp_cand_dists[k] > worst_dist) {
                            worst_dist = warp_cand_dists[k];
                            worst_idx = k;
                        }
                    }
                    warp_cand_count--;
                    warp_cand_dists[worst_idx] = warp_cand_dists[warp_cand_count];
                    warp_cand_ids[worst_idx] = warp_cand_ids[warp_cand_count];

                    worst_dist = -FLT_MAX;
                    for (int k = 0; k < warp_cand_count; ++k)
                        worst_dist = fmaxf(worst_dist, warp_cand_dists[k]);
                    distk = worst_dist;
                }
            }
            distk = __shfl_sync(0xFFFFFFFF, distk, 0);
        }
    }

    // ---- Phase 3: Output per-block candidates ----
    // Merge 4 warps' candidates. For simplicity, warp 0 writes first,
    // then warp 1, etc. Each warp writes its candidates sequentially.
    __shared__ float block_cand_dists[kMaxCandidatesPerBlock];
    __shared__ uint32_t block_cand_ids[kMaxCandidatesPerBlock];
    __shared__ int block_cand_total;

    if (threadIdx.x == 0) block_cand_total = 0;
    __syncthreads();

    // Lane 0 of each warp writes its candidates
    if (lane == 0 && warp_cand_count > 0) {
        int start_pos = atomicAdd(&block_cand_total, warp_cand_count);
        for (int k = 0; k < warp_cand_count && start_pos + k < (int)kMaxCandidatesPerBlock; ++k) {
            block_cand_dists[start_pos + k] = warp_cand_dists[k];
            block_cand_ids[start_pos + k] = warp_cand_ids[k];
        }
    }
    __syncthreads();

    // Write to global output
    size_t out_base = (q_idx * nprobe + cluster_rank) * kMaxCandidatesPerBlock;
    int count = min(block_cand_total, (int)kMaxCandidatesPerBlock);

    for (int k = threadIdx.x; k < count; k += blockDim.x) {
        d_candidate_dists[out_base + k] = block_cand_dists[k];
        d_candidate_ids[out_base + k] = block_cand_ids[k];
    }
    if (threadIdx.x == 0) {
        d_candidate_counts[q_idx * nprobe + cluster_rank] = count;
    }
}

// ============================================================================
// Top-K merge kernel: one block per query
// ============================================================================

// Merge kernel uses candidate buffers directly from global memory (no shared memory needed).
// Thread 0 of each block does sequential selection sort — sufficient for ~1K candidates.
__global__ void kernel_merge_topk(
    const float* __restrict__ d_candidate_dists,
    const uint32_t* __restrict__ d_candidate_ids,
    const uint32_t* __restrict__ d_candidate_counts,
    float* __restrict__ d_work_dists,          // [Q * max_total_cands] workspace
    uint32_t* __restrict__ d_work_ids,         // [Q * max_total_cands] workspace
    uint32_t* __restrict__ d_results,
    size_t Q, size_t nprobe, size_t topk,
    size_t max_total_cands)
{
    size_t q = blockIdx.x;
    if (q >= Q) return;

    if (threadIdx.x == 0) {
        float* all_dists = d_work_dists + q * max_total_cands;
        uint32_t* all_ids = d_work_ids + q * max_total_cands;
        int total_cands = 0;

        for (size_t cr = 0; cr < nprobe; ++cr) {
            int cnt = d_candidate_counts[q * nprobe + cr];
            size_t base = (q * nprobe + cr) * kMaxCandidatesPerBlock;
            for (int k = 0; k < cnt && total_cands < (int)max_total_cands; ++k) {
                all_dists[total_cands] = d_candidate_dists[base + k];
                all_ids[total_cands] = d_candidate_ids[base + k];
                total_cands++;
            }
        }

        // Simple selection sort for top-K
        for (size_t i = 0; i < topk && i < (size_t)total_cands; ++i) {
            int best = (int)i;
            for (int j = (int)i + 1; j < total_cands; ++j) {
                if (all_dists[j] < all_dists[best])
                    best = j;
            }
            if (best != (int)i) {
                float tmp_d = all_dists[i]; all_dists[i] = all_dists[best]; all_dists[best] = tmp_d;
                uint32_t tmp_id = all_ids[i]; all_ids[i] = all_ids[best]; all_ids[best] = tmp_id;
            }
            d_results[q * topk + i] = all_ids[i];
        }
        for (size_t i = total_cands; i < topk; ++i)
            d_results[q * topk + i] = 0xFFFFFFFF;
    }
}

// ============================================================================
// Launch wrappers
// ============================================================================

void launch_search(
    const GpuSegmentDescriptor* d_seg_descs,
    const GpuClusterDescriptor* d_clu_descs,
    const uint32_t* d_block_offsets,
    const uint32_t* d_cluster_offsets,
    const float* d_rotated_queries,
    const QuerySegmentConstants* d_query_consts,
    const uint32_t* d_centroid_ids,
    size_t Q, size_t nprobe, size_t topk,
    size_t num_segments, size_t total_D_seg,
    float* d_candidate_dists,
    uint32_t* d_candidate_ids,
    uint32_t* d_candidate_counts,
    cudaStream_t stream)
{
    if (Q == 0 || nprobe == 0) return;

    // Compute shared memory size
    size_t total_codebooks = total_D_seg / 4;
    constexpr int kConstsPerSeg = 7;
    size_t shmem_bytes = total_codebooks * 16 * sizeof(float)          // LUT
                       + num_segments * kConstsPerSeg * sizeof(float)  // constants
                       + sizeof(int)                                    // work counter
                       + total_D_seg * sizeof(float)                    // residual query
                       + kMaxCandidatesPerBlock * (sizeof(float) + sizeof(uint32_t)); // candidate buffer

    dim3 grid(Q, nprobe);
    int block_size = 128;

    kernel_search<<<grid, block_size, shmem_bytes, stream>>>(
        d_seg_descs, d_clu_descs, d_block_offsets, d_cluster_offsets,
        d_rotated_queries, d_query_consts, d_centroid_ids,
        Q, nprobe, topk, num_segments, total_D_seg,
        d_candidate_dists, d_candidate_ids, d_candidate_counts);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

void launch_merge_topk(
    const float* d_candidate_dists,
    const uint32_t* d_candidate_ids,
    const uint32_t* d_candidate_counts,
    float* d_work_dists,
    uint32_t* d_work_ids,
    uint32_t* d_results,
    size_t Q, size_t nprobe, size_t topk,
    size_t max_total_cands,
    cudaStream_t stream)
{
    if (Q == 0) return;

    kernel_merge_topk<<<Q, 1, 0, stream>>>(
        d_candidate_dists, d_candidate_ids, d_candidate_counts,
        d_work_dists, d_work_ids, d_results,
        Q, nprobe, topk, max_total_cands);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace saq::gpu
