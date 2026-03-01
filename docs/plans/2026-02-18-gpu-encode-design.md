# GPU Encode Pipeline Design

**Date**: 2026-02-28
**Scope**: CUDA acceleration for SAQ encode (IVF::construct)
**Target hardware**: RTX 5090 (`gpu-local` branch), A100/H100 (`gpu-pace` branch)
**Approach**: Full GPU encode — all steps on GPU, encoded data stays in GPU memory

## Overview

Port the SAQ encode pipeline to CUDA so that rotation, CAQ quantization, code adjustment, factor computation, and code packing all run on GPU. The input vectors, centroids, and rotation matrices ("codebook") are uploaded once and stay resident. Encoded cluster data remains in GPU memory for future GPU search.

## Current CPU Encode Flow

```
IVF::construct (ivf_index.cpp)
  For each cluster (OpenMP parallel):
    SAQuantizer::quantize_cluster
      For each segment:
        1. Slice vectors to segment dims
        2. Rotate: vecs * rotator.P  (optional per-segment random rotation)
        3. Subtract centroid
        4. QuantizerCluster::quantize
           CAQEncoder::encode_and_fac (per vector, sequential):
             a. Compute v_max = max(|o_i|)
             b. Initial quantize: code[j] = floor((o[j]+v_max) / delta)
             c. Code adjustment: 6 rounds x D_seg dims (cosine optimization)
             d. Compute factors: fac_rescale, fac_error
             e. ip_cent_oa = centroid . get_oa()
           ClusterPacker::store_and_pack (per vector):
             f. Pack short codes (MSB extraction)
             g. Pack long codes (remaining bits)
        5. ClusterPacker::finalize_and_store (fastscan reorder)
```

Bottleneck: Step 4 is sequential per vector. With 100K vectors, this is the dominant cost.

## GPU Memory Layout

| Data | Size (100K, 1536D, K=4096) | Lifetime |
|------|----------------------------|----------|
| Input vectors (PCA-rotated) | 100K x 1536 x 4B = 586 MB | Entire construct |
| Centroids | 4096 x 1536 x 4B = 24 MB | Entire construct |
| Cluster IDs | 100K x 4B = 400 KB | Entire construct |
| Per-segment rotation matrices | ~5-15 x D_seg^2 x 4B | Entire construct |
| Output encoded data | Variable per cluster | Permanent (GPU-resident) |

Total persistent: ~700 MB. Fits on RTX 5090 (32 GB), A100 (40/80 GB), H100 (80 GB).

The rotation matrices are the "codebook" stored on GPU. Loaded once, reused for every cluster's encode.

## Key Design Insight

**We don't loop over clusters on GPU.** Instead:
1. Sort all vectors by cluster ID (thrust)
2. Rotate ALL vectors at once per segment (one cuBLAS GEMM)
3. Encode ALL vectors at once (one kernel launch)
4. Pack codes with cluster-aware indexing

This eliminates 4096 small kernel launches and maximizes GPU occupancy.

## CUDA Kernels

### Kernel 1: Rotation + Residual (`gpu_rotate_and_residual`)

**Purpose**: For each segment, compute `residual = (vec_segment - centroid_segment) * P_segment`

**Implementation**: cuBLAS `cublasSgemm` for the matrix multiply. Centroid subtraction fused as a pre-processing elementwise kernel or handled via cuBLAS bias.

**Dimensions**: GEMM of shape `(N_total, D_seg) x (D_seg, D_seg)` where N_total = 100K, D_seg varies per segment.

**Note**: Vectors must be gathered by cluster to subtract the correct centroid per vector. Two sub-steps:
- `gpu_subtract_centroid<<<>>>`: each thread subtracts its cluster's centroid (looked up via sorted cluster IDs + offsets)
- `cublasSgemm`: multiply residuals by rotation matrix

### Kernel 2: Warp-Cooperative CAQ Encode (`gpu_caq_encode`)

**Purpose**: Full CAQ encode including code adjustment.

**Launch config**:
- Grid: `(ceil(N_vectors / warps_per_block), 1)`, warps_per_block = 4-8
- Block: `(32 * warps_per_block, 1)`
- Each warp (32 threads) cooperates on one vector

**Per-warp flow** (32 lanes, each handles `chunk = ceil(D_seg/32)` dimensions):

```
1. v_max reduction
   - Each lane: local_max = max(|o[j]|) for j in my chunk
   - Warp reduce via __shfl_down_sync -> global v_max
   - __shfl_sync broadcast to all lanes

2. Initial quantization
   - Each lane: code[j] = floor((o[j] + v_max) / delta) for j in my chunk
   - Each lane: partial ip_o_code, code_l2sqr, code_sum
   - Warp reduce + broadcast -> global ip_o_oa, oa_l2sqr

3. Code adjustment (up to cfg.caq_adj_rd_lmt rounds)
   For each round:
     - Each lane adjusts codes in its chunk (try ++/--, compare cosine)
       Dimensions are INDEPENDENT within a round -> no cross-lane sync needed
     - After adjustment: warp reduce adj_cnt to check convergence
     - Correction pass: each lane recomputes partial sums, warp reduce
     - Break if adj_cnt == 0 or round >= limit

4. Factor computation
   - Lane 0 computes fac_rescale = o_l2sqr / ip_o_oa
   - Lane 0 computes fac_error
   - rescale_vmx_to1() adjustments (lane 0, broadcast)

5. ip_cent_oa (if num_bits > 1)
   - Each lane computes partial dot product of centroid . get_oa()
   - Warp reduce to get full dot product
```

**Register budget per lane**:
- `ceil(1536/32) = 48` int codes = 48 registers
- `48` float vector values = 48 registers
- Accumulators + temporaries = ~20 registers
- Total: ~116 registers per thread (within 255 limit)

**No shared memory needed** — all cross-lane communication via warp shuffles.

### Kernel 3: Pack Short Codes (`gpu_pack_short_codes`)

**Purpose**: Extract MSB of each code, pack into bytes.

**Launch**: One thread per vector.

**Per-thread**:
```
for j in 0..shortcode_byte_num:
  byte = 0
  for bit in 0..7:
    byte |= (code[j*8 + bit] & short_bit) ? (0x80 >> bit) : 0
  short_code[j] = byte
```

### Kernel 4: Pack Long Codes (`gpu_pack_long_codes`)

**Purpose**: Mask off short bit, pack remaining bits into compact format.

**Launch**: One thread per vector.

**Per-thread**: Applies the `compacted_code_func` logic — bit extraction and packing for `num_bits - 1` lower bits per dimension.

### Kernel 5: Fastscan Reorder (`gpu_fastscan_reorder`)

**Purpose**: Reorder short codes from per-vector layout into the blocked fastscan layout (32 vectors interleaved per block).

**Launch**: One thread per block-of-32 per cluster.

**Per-thread**: Reads 32 vectors' short codes, writes them in the transposed fastscan format that the SIMD search expects.

### Kernel 6: Factor Copy (`gpu_store_factors`)

**Purpose**: Copy o_l2norm and ip_cent_oa factors into the blocked cluster layout (KFastScanSize-aligned).

**Launch**: One thread per block-of-32 per cluster.

## GPU Data Structures

### `GpuSaqCluData`

GPU-resident equivalent of `SaqCluData`. Stores per-cluster encoded data:

```cpp
struct GpuSaqCluData {
    size_t num_vec;
    size_t num_segments;
    size_t num_blocks;       // ceil(num_vec / 32)
    uint32_t* d_ids;         // [num_vec] vector IDs

    struct GpuSegmentData {
        size_t num_dim_pad;
        size_t num_bits;
        float* d_centroid;           // [num_dim_pad]
        float* d_factor_o_l2norm;    // [num_blocks * 32]
        float* d_factor_ip_cent_oa;  // [num_blocks * 32] (optional)
        uint8_t* d_short_codes;      // fastscan-packed layout
        uint8_t* d_long_codes;       // compacted per-vector
        ExFactor* d_long_factors;    // [num_vec] rescale + error
    };

    std::vector<GpuSegmentData> segments;
};
```

### `GpuIVF`

Extends or wraps IVF with GPU-resident data:

```cpp
class GpuIVF {
    // CPU-side metadata
    std::unique_ptr<SaqData> saq_data_;
    std::unique_ptr<Initializer> initer_;

    // GPU-resident encoded clusters
    std::vector<GpuSaqCluData> gpu_clusters_;

    // GPU-resident codebook (rotation matrices)
    std::vector<float*> d_rotation_matrices_;  // per-segment

    void construct_gpu(const FloatRowMat& data,
                       const FloatRowMat& centroids,
                       const PID* cluster_ids);
};
```

## Execution Flow

```
GpuIVF::construct_gpu(data, centroids, cluster_ids):

  1. UPLOAD
     cudaMemcpy: d_vectors, d_centroids, d_cluster_ids
     cudaMemcpy: d_rotation_matrices (per segment)

  2. PREPARE CLUSTERS (GPU-side)
     thrust::sort_by_key(d_cluster_ids, d_vector_indices)
     thrust::exclusive_scan(cluster_counts) -> d_cluster_offsets
     Allocate GpuSaqCluData for each cluster

  3. FOR EACH SEGMENT s:
     a. gpu_subtract_centroid<<<>>>
        Each thread: d_residual[i] = d_vectors[i][s_offset:s_end] - d_centroids[cluster_of[i]][s_offset:s_end]

     b. cublasSgemm(d_residual, d_rotation_matrices[s], d_rotated)
        GEMM: (N_total, D_seg) x (D_seg, D_seg) -> (N_total, D_seg)

     c. gpu_caq_encode<<<>>> (warp-cooperative)
        Input: d_rotated (N_total x D_seg)
        Output: d_codes (N_total x D_seg int), d_factors (N_total)

     d. gpu_pack_short_codes<<<>>>
        Input: d_codes
        Output: d_short_codes_raw (per-vector byte layout)

     e. gpu_pack_long_codes<<<>>>
        Input: d_codes
        Output: d_long_codes (compacted per-vector)

     f. gpu_store_factors<<<>>>
        Copy factors into per-cluster blocked layout

  4. gpu_fastscan_reorder<<<>>>
     Reorder short codes into fastscan layout per cluster

  5. DATA STAYS ON GPU in GpuSaqCluData structures
```

## Branch Strategy

- `gpu` branch: Base GPU design + shared infrastructure (memory management, error handling, CMake CUDA setup)
- `gpu-local` branch (from `gpu`): RTX 5090 tuning — block sizes, shared memory config for compute capability 10.0
- `gpu-pace` branch (from `gpu`): A100/H100 tuning — larger shared memory (164KB+), Tensor Core usage for rotation GEMM if beneficial

## CMake Integration

```cmake
option(SAQ_BUILD_CUDA "Build CUDA GPU acceleration" OFF)

if(SAQ_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100")  # A100, RTX30xx, RTX40xx, H100, RTX50xx

    target_sources(saq PRIVATE
        src/gpu/gpu_encoder.cu
        src/gpu/gpu_packer.cu
        src/gpu/gpu_ivf_construct.cu
    )
    target_include_directories(saq PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(saq PRIVATE cublas)
    target_compile_definitions(saq PRIVATE SAQ_USE_CUDA)
endif()
```

## Performance Expectations

**Rotation GEMM** (cuBLAS): (100K, 1536) x (1536, 1536) = ~450 GFLOP. At RTX 5090 ~80 TFLOPS FP32: ~5.6 ms. At A100 ~19.5 TFLOPS: ~23 ms.

**CAQ Encode**: 100K vectors, 8 warps/block = 4000 vectors/SM. RTX 5090 (170 SMs): ~24 vectors/SM. A100 (108 SMs): ~37 vectors/SM. Each vector: ~6 rounds x 1536 dims x ~10 ops = ~90K ops. Highly compute-bound, expect < 10 ms.

**Total GPU encode** (vs CPU OpenMP): Expected 10-50x speedup over 8-thread CPU for the full construct.

## Future Work (GPU Search)

The `GpuSaqCluData` stored on GPU enables a future GPU search path:
- GPU centroid distances (replace FlatInitializer)
- GPU 3-stage search (variance pruning, fastscan LUT, accurate distance)
- Requires porting the Lut class and fastscan SIMD to CUDA equivalents

This is out of scope for the current design but the data layout is designed to support it.
