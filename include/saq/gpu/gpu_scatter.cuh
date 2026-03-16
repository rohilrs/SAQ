#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace saq::gpu {

/// Scatter short codes: linear 1-bit packed → GPU blocked layout (fused reorder).
/// Precondition: vectors sorted by cluster ID.
void launch_scatter_short_codes(
    const uint8_t* flat_short,
    uint8_t* pool_short,
    const uint32_t* d_cluster_offsets,
    const uint32_t* d_block_offsets,
    const uint32_t* d_cluster_ids,
    size_t D_seg, size_t N, size_t num_bits,
    cudaStream_t stream = 0);

/// Scatter long codes: flat → per-cluster contiguous in pool.
/// Precondition: vectors sorted by cluster ID.
void launch_scatter_long_codes(
    const uint8_t* flat_long,
    uint8_t* pool_long,
    const uint32_t* d_cluster_offsets,
    const uint32_t* d_cluster_ids,
    size_t long_bytes_per_vec, size_t N,
    cudaStream_t stream = 0);

/// Scatter factors: flat → per-cluster blocked/per-vector layout in pool.
/// o_l2norm and ip_cent_oa use blocked layout (blocks of 32).
/// rescale and error use per-vector layout indexed by cluster_offsets.
/// Precondition: vectors sorted by cluster ID.
void launch_scatter_factors(
    const float* flat_o_l2norm,
    const float* flat_ip_cent_oa,
    const float* flat_rescale,
    const float* flat_error,
    float* pool_o_l2norm,
    float* pool_ip_cent_oa,
    float* pool_rescale,
    float* pool_error,
    const uint32_t* d_cluster_offsets,
    const uint32_t* d_block_offsets,
    const uint32_t* d_cluster_ids,
    size_t N,
    cudaStream_t stream = 0);

/// Copy centroids into pool (simple D2D memcpy, same contiguous layout).
void copy_centroids_to_pool(
    const float* d_centroids_seg,
    float* pool_centroids,
    size_t D_seg, size_t K,
    cudaStream_t stream = 0);

} // namespace saq::gpu
