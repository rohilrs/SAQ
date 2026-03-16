#include "saq/gpu/gpu_scatter.cuh"
#include "saq/gpu/gpu_utils.cuh"

namespace saq::gpu {

// ---------------------------------------------------------------------------
// Short code scatter + fastscan reorder
// Input:  flat_short[i] has D_seg/8 bytes per vector, 1-bit-per-dim, descending
//         bit order (dim 0 → bit 7 of byte 0).
// Output: GPU blocked layout — 1 byte per codebook (4 dims) per vector,
//         organized as blocks of 32 vectors.
// ---------------------------------------------------------------------------
__global__ void kernel_scatter_short_codes(
    const uint8_t* __restrict__ flat_short,
    uint8_t* __restrict__ pool_short,
    const uint32_t* __restrict__ d_cluster_offsets,
    const uint32_t* __restrict__ d_block_offsets,
    const uint32_t* __restrict__ d_cluster_ids,
    size_t D_seg, size_t N, size_t num_bits)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (num_bits == 0) return;

    uint32_t c = d_cluster_ids[i];
    uint32_t pos = static_cast<uint32_t>(i) - d_cluster_offsets[c];
    uint32_t block_in_clu = pos / 32;
    uint32_t vec_in_block = pos % 32;
    uint32_t global_block = d_block_offsets[c] + block_in_clu;
    size_t num_codebooks = D_seg / 4;

    const uint8_t* src = flat_short + i * (D_seg / 8);

    for (size_t cb = 0; cb < num_codebooks; ++cb) {
        size_t dim_base = cb * 4;
        uint8_t code4 = 0;
        // LUT convention (from kPos): bit 3 = dim 0, bit 2 = dim 1, bit 1 = dim 2, bit 0 = dim 3
        // So dim j maps to bit (3 - j)
        for (int j = 0; j < 4; ++j) {
            size_t dim = dim_base + j;
            size_t byte_idx = dim / 8;
            size_t bit_pos = 7 - (dim % 8);  // descending bit order in packed bytes
            uint8_t bit = (src[byte_idx] >> bit_pos) & 1;
            code4 |= (bit << (3 - j));  // dim 0 → bit 3, dim 3 → bit 0
        }
        size_t dst_idx = (size_t)global_block * 32 * num_codebooks
                       + (size_t)vec_in_block * num_codebooks + cb;
        pool_short[dst_idx] = code4;
    }
}

void launch_scatter_short_codes(
    const uint8_t* flat_short, uint8_t* pool_short,
    const uint32_t* d_cluster_offsets, const uint32_t* d_block_offsets,
    const uint32_t* d_cluster_ids,
    size_t D_seg, size_t N, size_t num_bits,
    cudaStream_t stream)
{
    if (N == 0 || num_bits == 0) return;
    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    kernel_scatter_short_codes<<<blocks, threads, 0, stream>>>(
        flat_short, pool_short, d_cluster_offsets, d_block_offsets,
        d_cluster_ids, D_seg, N, num_bits);
}

// ---------------------------------------------------------------------------
// Long code scatter: flat → per-cluster contiguous in pool.
// ---------------------------------------------------------------------------
__global__ void kernel_scatter_long_codes(
    const uint8_t* __restrict__ flat_long,
    uint8_t* __restrict__ pool_long,
    const uint32_t* __restrict__ d_cluster_offsets,
    const uint32_t* __restrict__ d_cluster_ids,
    size_t long_bytes_per_vec, size_t N)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint32_t c = d_cluster_ids[i];
    uint32_t pos = static_cast<uint32_t>(i) - d_cluster_offsets[c];

    const uint8_t* src = flat_long + i * long_bytes_per_vec;
    uint8_t* dst = pool_long + ((size_t)d_cluster_offsets[c] + pos) * long_bytes_per_vec;

    for (size_t b = 0; b < long_bytes_per_vec; ++b) {
        dst[b] = src[b];
    }
}

void launch_scatter_long_codes(
    const uint8_t* flat_long, uint8_t* pool_long,
    const uint32_t* d_cluster_offsets, const uint32_t* d_cluster_ids,
    size_t long_bytes_per_vec, size_t N,
    cudaStream_t stream)
{
    if (N == 0 || long_bytes_per_vec == 0) return;
    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    kernel_scatter_long_codes<<<blocks, threads, 0, stream>>>(
        flat_long, pool_long, d_cluster_offsets, d_cluster_ids,
        long_bytes_per_vec, N);
}

// ---------------------------------------------------------------------------
// Factor scatter: flat → blocked + per-vector layout in pool.
// ---------------------------------------------------------------------------
__global__ void kernel_scatter_factors(
    const float* __restrict__ flat_o_l2norm,
    const float* __restrict__ flat_ip_cent_oa,
    const float* __restrict__ flat_rescale,
    const float* __restrict__ flat_error,
    float* __restrict__ pool_o_l2norm,
    float* __restrict__ pool_ip_cent_oa,
    float* __restrict__ pool_rescale,
    float* __restrict__ pool_error,
    const uint32_t* __restrict__ d_cluster_offsets,
    const uint32_t* __restrict__ d_block_offsets,
    const uint32_t* __restrict__ d_cluster_ids,
    size_t N)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint32_t c = d_cluster_ids[i];
    uint32_t pos = static_cast<uint32_t>(i) - d_cluster_offsets[c];
    uint32_t block_in_clu = pos / 32;
    uint32_t vec_in_block = pos % 32;
    uint32_t global_block = d_block_offsets[c] + block_in_clu;

    // o_l2norm and ip_cent_oa: blocked layout (blocks of 32)
    size_t blocked_idx = (size_t)global_block * 32 + vec_in_block;
    pool_o_l2norm[blocked_idx]   = flat_o_l2norm[i];
    pool_ip_cent_oa[blocked_idx] = flat_ip_cent_oa[i];

    // rescale and error: per-vector layout
    size_t vec_idx = (size_t)d_cluster_offsets[c] + pos;
    pool_rescale[vec_idx] = flat_rescale[i];
    pool_error[vec_idx]   = flat_error[i];
}

void launch_scatter_factors(
    const float* flat_o_l2norm, const float* flat_ip_cent_oa,
    const float* flat_rescale, const float* flat_error,
    float* pool_o_l2norm, float* pool_ip_cent_oa,
    float* pool_rescale, float* pool_error,
    const uint32_t* d_cluster_offsets, const uint32_t* d_block_offsets,
    const uint32_t* d_cluster_ids, size_t N,
    cudaStream_t stream)
{
    if (N == 0) return;
    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    kernel_scatter_factors<<<blocks, threads, 0, stream>>>(
        flat_o_l2norm, flat_ip_cent_oa, flat_rescale, flat_error,
        pool_o_l2norm, pool_ip_cent_oa, pool_rescale, pool_error,
        d_cluster_offsets, d_block_offsets, d_cluster_ids, N);
}

// ---------------------------------------------------------------------------
// Centroid copy: simple D2D memcpy (same contiguous layout).
// ---------------------------------------------------------------------------
void copy_centroids_to_pool(
    const float* d_centroids_seg, float* pool_centroids,
    size_t D_seg, size_t K, cudaStream_t stream)
{
    SAQ_CUDA_CHECK(cudaMemcpyAsync(
        pool_centroids, d_centroids_seg,
        K * D_seg * sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

} // namespace saq::gpu
