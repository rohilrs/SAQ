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

/// Compute inner product between codebook-quantized vector and query segment.
/// Each dimension's code indexes into the codebook to get the centroid value,
/// then we compute dot product with the residual query.
///
/// Short codes are in fastscan layout: num_codebooks nibbles per vector.
/// Each nibble packs 4 dims: dim0→bit3, dim1→bit2, dim2→bit1, dim3→bit0.
/// Long codes are bit-compacted: (num_bits-1) bits per dim.
__device__ float gpu_codebook_ip(
    const float* __restrict__ resid_query,    // [D_seg]
    const float* __restrict__ codebook,       // [D_seg * entries_per_dim]
    const uint8_t* __restrict__ short_code,   // fastscan nibbles [num_codebooks]
    const uint8_t* __restrict__ long_code,    // bit-packed lower bits
    size_t D_seg, size_t num_bits,
    size_t entries_per_dim)
{
    size_t ex_bits = (num_bits > 1) ? num_bits - 1 : 0;
    size_t num_codebooks = D_seg / 4;
    float ip = 0.0f;

    for (size_t d = 0; d < D_seg; ++d) {
        // Extract MSB from fastscan nibble layout
        size_t cb_idx = d / 4;          // which codebook (nibble)
        int sub = (int)(d % 4);         // position within the 4-dim group
        int bit_in_nibble = 3 - sub;    // dim0→bit3, dim1→bit2, etc.
        int msb = (short_code[cb_idx] >> bit_in_nibble) & 1;

        int code_low = 0;
        if (ex_bits > 0 && long_code) {
            size_t bit_offset = d * ex_bits;
            for (size_t b = 0; b < ex_bits; ++b) {
                size_t gbit = bit_offset + b;
                if ((long_code[gbit / 8] >> (gbit % 8)) & 1)
                    code_low |= (1 << b);
            }
        }
        int full_code = (msb << ex_bits) | code_low;

        float centroid_val = __ldg(&codebook[d * entries_per_dim + full_code]);
        ip += centroid_val * resid_query[d];
    }
    return ip;
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

                float rescale = seg.factor_rescale[vec_offset];
                float ip_o_q;

                if (seg.codebook_centroids != nullptr) {
                    // ---- Codebook distance path ----
                    const float* resid_seg = smem_resid_query + seg_dim_offsets[s];
                    // Short codes in fastscan layout: num_codebooks nibbles per vector
                    const uint8_t* short_code = seg.short_codes
                        + (size_t)global_block * 32 * seg.num_codebooks
                        + lane * seg.num_codebooks;
                    const uint8_t* long_code = (seg.long_bytes_per_vec > 0)
                        ? seg.long_codes + vec_offset * seg.long_bytes_per_vec
                        : nullptr;

                    float cb_ip = gpu_codebook_ip(
                        resid_seg, seg.codebook_centroids,
                        short_code, long_code,
                        seg.D_seg, seg.num_bits,
                        seg.codebook_entries_per_dim);

                    ip_o_q = rescale * cb_ip;
                } else {
                    // ---- Uniform distance path (existing) ----
                    float sum_q_s = smem_consts_f[s * kConstsPerSeg + 2];
                    float sq_delta_s = smem_consts_f[s * kConstsPerSeg + 6];

                    float lut_sum = 0.0f;
                    {
                        const uint8_t* short_base = seg.short_codes
                            + (size_t)global_block * 32 * seg.num_codebooks;
                        const uint8_t* my_codes = short_base + lane * seg.num_codebooks;
                        for (size_t cb = 0; cb < seg.num_codebooks; ++cb) {
                            lut_sum += smem_lut_f[(seg_cb_offsets[s] + cb) * 16 + my_codes[cb]];
                        }
                    }

                    float full_ip;
                    if (seg.num_bits > 1 && seg.long_bytes_per_vec > 0) {
                        const float* resid_seg = smem_resid_query + seg_dim_offsets[s];
                        const uint8_t* long_code = seg.long_codes
                            + vec_offset * seg.long_bytes_per_vec;
                        float ext_ip = gpu_long_code_ip(resid_seg, long_code,
                                                         seg.D_seg, seg.num_bits);
                        full_ip = lut_sum + ext_ip * sq_delta_s
                                + (-1.0f + sq_delta_s / 2.0f) * sum_q_s;
                    } else {
                        full_ip = lut_sum + (-1.0f + sq_delta_s / 2.0f) * sum_q_s;
                    }
                    ip_o_q = rescale * full_ip;
                }

                float seg_dist = o_l2sqr + q_l2sqr_s - 2.0f * ip_o_q;
                acc_dist += seg_dist;
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
    float* __restrict__ d_results_dists,       // [Q * topk] output dists (nullable)
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
            if (d_results_dists) d_results_dists[q * topk + i] = all_dists[i];
        }
        for (size_t i = total_cands; i < topk; ++i) {
            d_results[q * topk + i] = 0xFFFFFFFF;
            if (d_results_dists) d_results_dists[q * topk + i] = INFINITY;
        }
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
    float* d_results_dists,
    size_t Q, size_t nprobe, size_t topk,
    size_t max_total_cands,
    cudaStream_t stream)
{
    if (Q == 0) return;

    kernel_merge_topk<<<Q, 1, 0, stream>>>(
        d_candidate_dists, d_candidate_ids, d_candidate_counts,
        d_work_dists, d_work_ids, d_results, d_results_dists,
        Q, nprobe, topk, max_total_cands);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// GPU batch centroid search: GEMM-based L2 distance + top-nprobe selection
// ============================================================================

// Per-query top-nprobe selection from K distances.
// One block per query, 256 threads cooperatively find top-nprobe smallest.
__global__ void kernel_topk_centroids(
    const float* __restrict__ d_dists,  // [Q * K] pairwise distances
    uint32_t* __restrict__ d_ids,       // [Q * nprobe] output
    size_t Q, size_t K, size_t nprobe)
{
    size_t q = blockIdx.x;
    if (q >= Q) return;

    const float* dists = d_dists + q * K;
    uint32_t* out = d_ids + q * nprobe;

    // Thread 0 does sequential partial selection (sufficient for K=4096, nprobe=200)
    if (threadIdx.x == 0) {
        // Use a simple max-heap of size nprobe
        // For K=4096, nprobe=200: selection sort is fast enough
        // Allocate a local buffer for top-nprobe tracking
        float best_dists[512];  // max nprobe supported
        uint32_t best_ids[512];
        size_t np = min(nprobe, (size_t)512);

        // Initialize with first nprobe elements
        for (size_t i = 0; i < np && i < K; ++i) {
            best_dists[i] = dists[i];
            best_ids[i] = (uint32_t)i;
        }

        // Find the current worst in the buffer
        size_t worst_idx = 0;
        float worst_val = best_dists[0];
        for (size_t i = 1; i < np; ++i) {
            if (best_dists[i] > worst_val) {
                worst_val = best_dists[i];
                worst_idx = i;
            }
        }

        // Scan remaining elements
        for (size_t i = np; i < K; ++i) {
            float d = dists[i];
            if (d < worst_val) {
                best_dists[worst_idx] = d;
                best_ids[worst_idx] = (uint32_t)i;
                // Find new worst
                worst_val = best_dists[0];
                worst_idx = 0;
                for (size_t j = 1; j < np; ++j) {
                    if (best_dists[j] > worst_val) {
                        worst_val = best_dists[j];
                        worst_idx = j;
                    }
                }
            }
        }

        // Sort the top-nprobe by distance (selection sort)
        for (size_t i = 0; i < np; ++i) {
            size_t min_idx = i;
            for (size_t j = i + 1; j < np; ++j) {
                if (best_dists[j] < best_dists[min_idx])
                    min_idx = j;
            }
            if (min_idx != i) {
                float td = best_dists[i]; best_dists[i] = best_dists[min_idx]; best_dists[min_idx] = td;
                uint32_t ti = best_ids[i]; best_ids[i] = best_ids[min_idx]; best_ids[min_idx] = ti;
            }
            out[i] = best_ids[i];
        }
    }
}

// Compute squared row norms: norms[i] = sum(data[i*D + d]^2 for d in 0..D-1)
// One block per row, 256 threads cooperatively reduce.
__global__ void kernel_row_norms(
    const float* __restrict__ data,
    float* __restrict__ norms,
    size_t N, size_t D)
{
    size_t row = blockIdx.x;
    if (row >= N) return;

    const float* r = data + row * D;
    float partial = 0.0f;
    for (size_t d = threadIdx.x; d < D; d += blockDim.x)
        partial += r[d] * r[d];

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);

    // Block reduce via shared memory
    __shared__ float warp_sums[8]; // max 256 threads = 8 warps
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0;
        for (int w = 0; w < (int)(blockDim.x + 31) / 32; ++w)
            total += warp_sums[w];
        norms[row] = total;
    }
}

// Add squared norms to distance matrix: dists[q*K+k] += q_norms[q] + c_norms[k]
// One thread per element.
__global__ void kernel_add_norms(
    float* __restrict__ dists,
    const float* __restrict__ q_norms,
    const float* __restrict__ c_norms,
    size_t Q, size_t K)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Q * K) return;
    size_t q = idx / K;
    size_t k = idx % K;
    dists[idx] += q_norms[q] + c_norms[k];
}

void launch_batch_centroid_search(
    const float* d_queries,
    const float* d_centroids,
    uint32_t* d_centroid_ids,
    size_t Q, size_t K, size_t D, size_t nprobe,
    cudaStream_t stream)
{
    if (Q == 0) return;

    // Step 1: Compute -2 * query · centroid via cuBLAS GEMM
    auto d_dists = device_alloc<float>(Q * K);

    CublasHandle cublas;
    float alpha = -2.0f, beta = 0.0f;
    // Row-major: C[Q×K] = A[Q×D] × B[K×D]^T
    // Column-major: C[K×Q] = B[D×K]^T × A[D×Q]
    SAQ_CUBLAS_CHECK(cublasSgemm(cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)K, (int)Q, (int)D,
        &alpha,
        d_centroids, (int)D,
        d_queries, (int)D,
        &beta,
        d_dists.get(), (int)K));

    // Step 2: Compute norms and add to distance matrix
    auto d_q_norms = device_alloc<float>(Q);
    auto d_c_norms = device_alloc<float>(K);

    kernel_row_norms<<<Q, 256, 0, stream>>>(d_queries, d_q_norms.get(), Q, D);
    kernel_row_norms<<<K, 256, 0, stream>>>(d_centroids, d_c_norms.get(), K, D);

    {
        size_t total = Q * K;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        kernel_add_norms<<<blocks, threads, 0, stream>>>(
            d_dists.get(), d_q_norms.get(), d_c_norms.get(), Q, K);
    }

    // Step 3: Top-nprobe selection per query
    kernel_topk_centroids<<<Q, 1, 0, stream>>>(
        d_dists.get(), d_centroid_ids, Q, K, nprobe);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace saq::gpu
