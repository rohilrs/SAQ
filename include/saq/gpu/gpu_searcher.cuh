#pragma once

#ifdef SAQ_USE_CUDA

#include <cstddef>
#include <cstdint>
#include <cfloat>

#include <cuda_runtime.h>

namespace saq::gpu {

/// Device-side per-segment descriptor for search kernel.
/// Uploaded once after construct; contains pool pointers and segment metadata.
struct GpuSegmentDescriptor {
    uint8_t* short_codes;          // base pointer in pool
    uint8_t* long_codes;           // base pointer in pool
    float* factor_o_l2norm;        // base pointer in pool (blocked layout)
    float* factor_ip_cent_oa;      // base pointer in pool (blocked layout)
    float* factor_rescale;         // base pointer in pool (per-vector)
    float* factor_error;           // base pointer in pool (per-vector)
    float* centroids;              // base pointer in pool
    size_t num_codebooks;          // D_seg / 4
    size_t D_seg;
    size_t num_bits;
    size_t long_bytes_per_vec;     // D_seg * (bits-1) / 8, or 0 for bits <= 1
    float* codebook_centroids;     // nullptr = uniform mode; else [D_seg * entries_per_dim]
    size_t codebook_entries_per_dim; // 1 << num_bits, or 0
};

/// Device-side per-cluster descriptor for search kernel.
struct GpuClusterDescriptor {
    size_t num_vec;
    size_t num_blocks;
    uint32_t* ids;                 // pointer into pool.ids for this cluster
};

/// Per-segment query constants computed on the host.
struct QuerySegmentConstants {
    float delta;                   // LUT quantization step (unused for float LUT)
    float sum_vl_lut;              // (vl + 0.5*delta) * num_codebooks (unused for float LUT)
    float sum_q;                   // sum of rotated query segment (raw, before centroid sub)
    float q_l2sqr;                 // squared norm (raw, before centroid sub)
    float q_l2norm;                // norm (raw, before centroid sub)
    float one_over_sqrtD;          // 1 / sqrt(D_seg)
    float sq_delta;                // 2.0 / (1 << num_bits), for accurate distance
};

/// Maximum candidates a single (query, cluster) block can output.
constexpr size_t kMaxCandidatesPerBlock = 256;

/// Launch the main search kernel: build LUT + 3-stage search.
/// Grid: dim3(Q, nprobe), Block: 128 threads (4 warps).
void launch_search(
    const GpuSegmentDescriptor* d_seg_descs,   // [num_segments] on device
    const GpuClusterDescriptor* d_clu_descs,   // [K] on device
    const uint32_t* d_block_offsets,            // [K+1] from pool
    const uint32_t* d_cluster_offsets,          // [K+1] from pool
    const float* d_rotated_queries,             // [Q * total_D_seg]
    const QuerySegmentConstants* d_query_consts,// [Q * num_segments]
    const uint32_t* d_centroid_ids,             // [Q * nprobe]
    size_t Q, size_t nprobe, size_t topk,
    size_t num_segments, size_t total_D_seg,
    float* d_candidate_dists,                   // [Q * nprobe * kMaxCandidatesPerBlock]
    uint32_t* d_candidate_ids,                  // [Q * nprobe * kMaxCandidatesPerBlock]
    uint32_t* d_candidate_counts,               // [Q * nprobe]
    cudaStream_t stream = 0);

/// Merge per-block candidates into final top-K results.
/// Grid: Q blocks, Block: 1 thread (sequential selection sort).
void launch_merge_topk(
    const float* d_candidate_dists,
    const uint32_t* d_candidate_ids,
    const uint32_t* d_candidate_counts,
    float* d_work_dists,                        // [Q * max_total_cands] workspace
    uint32_t* d_work_ids,                       // [Q * max_total_cands] workspace
    uint32_t* d_results,                        // [Q * topk] output
    float* d_results_dists,                     // [Q * topk] output dists (nullable)
    size_t Q, size_t nprobe, size_t topk,
    size_t max_total_cands,
    cudaStream_t stream = 0);

/// GPU batch centroid search: compute L2 distances from Q queries to K centroids
/// via cuBLAS GEMM, then find top-nprobe per query.
/// Replaces the CPU-side FlatInitializer loop.
void launch_batch_centroid_search(
    const float* d_queries,            // [Q * D] queries on device
    const float* d_centroids,          // [K * D] centroids on device
    uint32_t* d_centroid_ids,          // [Q * nprobe] output: top-nprobe cluster IDs per query
    size_t Q, size_t K, size_t D, size_t nprobe,
    cudaStream_t stream = 0);

} // namespace saq::gpu

#endif // SAQ_USE_CUDA
