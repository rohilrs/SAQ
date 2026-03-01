#include "saq/gpu/gpu_packer.cuh"
#include "saq/gpu/gpu_utils.cuh"

namespace saq::gpu {

// ============================================================================
// Short code packing: extract MSB, pack 8 per byte
// ============================================================================

__global__ void kernel_pack_short_codes(
    const int* __restrict__ d_codes,
    uint8_t* __restrict__ d_short_codes,
    size_t D_seg,
    size_t N,
    uint16_t short_bit) {

    size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= N) return;

    const int* codes = d_codes + vec_idx * D_seg;
    size_t code_bytes = D_seg / 8;
    uint8_t* out = d_short_codes + vec_idx * code_bytes;

    for (size_t j = 0; j < code_bytes; ++j) {
        size_t base = j * 8;
        uint8_t byte = 0;
        byte |= (codes[base + 0] & short_bit) ? 0x80 : 0;
        byte |= (codes[base + 1] & short_bit) ? 0x40 : 0;
        byte |= (codes[base + 2] & short_bit) ? 0x20 : 0;
        byte |= (codes[base + 3] & short_bit) ? 0x10 : 0;
        byte |= (codes[base + 4] & short_bit) ? 0x08 : 0;
        byte |= (codes[base + 5] & short_bit) ? 0x04 : 0;
        byte |= (codes[base + 6] & short_bit) ? 0x02 : 0;
        byte |= (codes[base + 7] & short_bit) ? 0x01 : 0;
        out[j] = byte;
    }
}

void launch_pack_short_codes(
    const int* d_codes,
    uint8_t* d_short_codes,
    size_t D_seg,
    size_t N,
    size_t num_bits,
    cudaStream_t stream) {

    if (num_bits == 0 || N == 0) return;
    uint16_t short_bit = 1 << (num_bits - 1);

    constexpr int kBlockSize = 256;
    int grid = (N + kBlockSize - 1) / kBlockSize;
    kernel_pack_short_codes<<<grid, kBlockSize, 0, stream>>>(
        d_codes, d_short_codes, D_seg, N, short_bit);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Long code packing: bit-compact the lower (num_bits-1) bits
// ============================================================================

__global__ void kernel_pack_long_codes(
    const int* __restrict__ d_codes,
    uint8_t* __restrict__ d_long_codes,
    size_t D_seg,
    size_t N,
    size_t num_bits,
    uint16_t short_bit) {

    size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= N) return;

    size_t ex_bits = num_bits - 1;
    size_t long_code_bytes = D_seg * ex_bits / 8;
    const int* codes = d_codes + vec_idx * D_seg;
    uint8_t* out = d_long_codes + vec_idx * long_code_bytes;

    // Zero output
    for (size_t i = 0; i < long_code_bytes; ++i)
        out[i] = 0;

    // Bit-compact: for each dimension, extract lower (num_bits-1) bits
    // Layout: groups of 8 dimensions, each producing ex_bits bytes
    size_t shift = 0;
    uint8_t* optr = out;
    for (size_t d = 0; d < D_seg; ++d) {
        uint16_t val = codes[d] & (short_bit - 1); // lower bits
        for (size_t b = 0; b < ex_bits; ++b) {
            optr[b] |= ((val >> b) & 1) << shift;
        }
        ++shift;
        if (shift == 8) {
            shift = 0;
            optr += ex_bits;
        }
    }
}

void launch_pack_long_codes(
    const int* d_codes,
    uint8_t* d_long_codes,
    size_t D_seg,
    size_t N,
    size_t num_bits,
    cudaStream_t stream) {

    if (num_bits <= 1 || N == 0) return;
    uint16_t short_bit = 1 << (num_bits - 1);

    constexpr int kBlockSize = 256;
    int grid = (N + kBlockSize - 1) / kBlockSize;
    kernel_pack_long_codes<<<grid, kBlockSize, 0, stream>>>(
        d_codes, d_long_codes, D_seg, N, num_bits, short_bit);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Factor storage: copy flat arrays into blocked cluster layout
// ============================================================================

__global__ void kernel_store_factors(
    const float* __restrict__ d_o_l2norm_flat,
    const float* __restrict__ d_ip_cent_oa_flat,
    const float* __restrict__ d_fac_rescale_flat,
    const float* __restrict__ d_fac_error_flat,
    float* __restrict__ d_factor_o_l2norm_blocked,
    float* __restrict__ d_factor_ip_cent_oa_blocked,
    float* __restrict__ d_long_factor_rescale,
    float* __restrict__ d_long_factor_error,
    size_t num_vec,
    size_t global_offset) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vec) return;

    size_t src = global_offset + idx;
    d_factor_o_l2norm_blocked[idx] = d_o_l2norm_flat[src];
    d_factor_ip_cent_oa_blocked[idx] = d_ip_cent_oa_flat[src];
    d_long_factor_rescale[idx] = d_fac_rescale_flat[src];
    d_long_factor_error[idx] = d_fac_error_flat[src];
}

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
    cudaStream_t stream) {

    if (num_vec_in_cluster == 0) return;

    constexpr int kBlockSize = 256;
    int grid = (num_vec_in_cluster + kBlockSize - 1) / kBlockSize;
    kernel_store_factors<<<grid, kBlockSize, 0, stream>>>(
        d_o_l2norm_flat, d_ip_cent_oa_flat,
        d_fac_rescale_flat, d_fac_error_flat,
        d_factor_o_l2norm_blocked, d_factor_ip_cent_oa_blocked,
        d_long_factor_rescale, d_long_factor_error,
        num_vec_in_cluster, global_offset);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace saq::gpu
