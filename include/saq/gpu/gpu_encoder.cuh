#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace saq::gpu {

/// Subtract each vector's cluster centroid from its segment slice.
void launch_subtract_centroid(
    const float* d_vectors,
    const float* d_centroids,
    const uint32_t* d_cluster_ids,
    float* d_residuals,
    size_t seg_offset,
    size_t D_seg,
    size_t D_total,
    size_t N,
    cudaStream_t stream = 0);

/// Warp-cooperative CAQ encode kernel.
void launch_caq_encode(
    const float* d_rotated,
    int* d_codes,
    float* d_o_l2norm,
    float* d_fac_rescale,
    float* d_fac_error,
    float* d_ip_cent_oa,
    const float* d_centroids_seg,
    const uint32_t* d_cluster_ids,
    size_t D_seg,
    size_t N,
    size_t K,
    size_t num_bits,
    uint16_t code_max,
    int caq_adj_rd_lmt,
    float caq_adj_eps,
    int caq_ori_qB,
    cudaStream_t stream = 0);

} // namespace saq::gpu
