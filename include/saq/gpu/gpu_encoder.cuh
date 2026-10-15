#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace saq::gpu {

/// Fused encode: subtract rotated centroid + CAQ encode + pack short/long codes.
/// Input: GEMM output (raw vectors rotated, NOT residuals) + rotated centroids.
/// Eliminates d_residuals and d_codes intermediate buffers.
void launch_fused_caq_encode(
    const float* d_vectors_rotated,    // [N * D_seg] GEMM output on raw vectors
    const float* d_rotated_centroids,  // [K * D_seg] precomputed on CPU
    const uint32_t* d_cluster_ids,     // [N]
    float* d_o_l2norm,                 // [N] output
    float* d_fac_rescale,              // [N] output
    float* d_fac_error,                // [N] output
    float* d_ip_cent_oa,               // [N] output
    uint8_t* d_short_raw,              // [N * D_seg/8] output (1-bit packed, descending)
    uint8_t* d_long_raw,               // [N * long_bytes] output
    size_t D_seg, size_t N, size_t K,
    size_t num_bits, uint16_t code_max,
    int caq_adj_rd_lmt, float caq_adj_eps, int caq_ori_qB,
    int caq_sequential = 0,
    cudaStream_t stream = 0);

/// No-rotation variant: reads raw vectors + centroids with segment offset.
void launch_fused_caq_encode_no_rotation(
    const float* d_vectors,            // [N * D_total] full vectors
    const float* d_centroids,          // [K * D_total] full centroids
    const uint32_t* d_cluster_ids,
    size_t seg_offset, size_t D_seg, size_t D_total,
    float* d_o_l2norm, float* d_fac_rescale,
    float* d_fac_error, float* d_ip_cent_oa,
    uint8_t* d_short_raw, uint8_t* d_long_raw,
    size_t N, size_t K,
    size_t num_bits, uint16_t code_max,
    int caq_adj_rd_lmt, float caq_adj_eps, int caq_ori_qB,
    cudaStream_t stream = 0);

/// Codebook-aware encode: binary search quantization with per-dimension codebooks.
/// codebook_centroids: [D_seg * entries_per_dim], dim-major layout.
/// Output codes go into d_short_raw/d_long_raw as packed indices (same layout as uniform).
/// Heterogeneous codebook encode: per-dim widths (0..8), codebooks padded to
/// cb_pad entries per dim. Writes per-dim codes to d_codes_out [N x D_seg]
/// plus the usual per-vector factors; byte packing happens separately
/// (launch_het_pack) from the layout.
void launch_het_codebook_encode(
    const float* d_vectors_rotated,
    const float* d_rotated_centroids,
    const uint32_t* d_cluster_ids,
    const float* d_codebook_centroids,   // [D_seg * cb_pad]
    const uint8_t* d_widths,             // [D_seg]
    size_t cb_pad,
    float* d_o_l2norm, float* d_fac_rescale,
    float* d_fac_error, float* d_ip_cent_oa,
    uint8_t* d_codes_out,
    size_t D_seg, size_t N, size_t K,
    int caq_adj_rd_lmt, float caq_adj_eps,
    cudaStream_t stream = 0);

void launch_fused_codebook_encode(
    const float* d_vectors_rotated,
    const float* d_rotated_centroids,
    const uint32_t* d_cluster_ids,
    const float* d_codebook_centroids,   // [D_seg * entries_per_dim], dim-major
    size_t entries_per_dim,              // 1 << seg_bits
    float* d_o_l2norm,
    float* d_fac_rescale,
    float* d_fac_error,
    float* d_ip_cent_oa,
    uint8_t* d_short_raw,
    uint8_t* d_long_raw,
    size_t D_seg, size_t N, size_t K,
    size_t num_bits, uint16_t code_max,
    int caq_adj_rd_lmt, float caq_adj_eps,
    cudaStream_t stream = 0);

} // namespace saq::gpu
