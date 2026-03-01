#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace saq::gpu {

/// Pack short codes: extract MSB of each code value, pack 8 per byte.
void launch_pack_short_codes(
    const int* d_codes,
    uint8_t* d_short_codes,
    size_t D_seg,
    size_t N,
    size_t num_bits,
    cudaStream_t stream = 0);

/// Pack long codes: mask off short bit, compact remaining bits.
void launch_pack_long_codes(
    const int* d_codes,
    uint8_t* d_long_codes,
    size_t D_seg,
    size_t N,
    size_t num_bits,
    cudaStream_t stream = 0);

/// Copy per-vector factors into blocked cluster layout.
void launch_store_factors(
    const float* d_o_l2norm_flat,
    const float* d_ip_cent_oa_flat,
    const float* d_fac_rescale_flat,
    const float* d_fac_error_flat,
    float* d_factor_o_l2norm_blocked,
    float* d_factor_ip_cent_oa_blocked,
    float* d_long_factor_rescale,
    float* d_long_factor_error,
    size_t num_vec_in_cluster,
    size_t global_offset,
    cudaStream_t stream = 0);

} // namespace saq::gpu
