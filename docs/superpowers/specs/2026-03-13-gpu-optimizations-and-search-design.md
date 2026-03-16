# GPU Encode Optimizations and Batch Search — Design Spec

**Date:** 2026-03-13
**Branch:** `gpu`
**Status:** Draft

## 1. Overview

This spec covers six components that optimize the GPU encode pipeline and add GPU-accelerated batch search to the SAQ inverted file index. The work targets the RTX 5090 (SM 120) with portability across SM 80+ GPUs.

### Motivation

The current GPU encode pipeline spends only 7.5% of wall time in actual compute kernels. The rest is CUDA API overhead: per-cluster `cudaMalloc` (57%), per-cluster D2D scatter (22%), and H2D upload (8%). Additionally, there is no GPU search path — search runs single-threaded on CPU at ~40ms/query.

### Goals

1. Reduce GPU encode wall time from ~1,550ms to ~220ms (2.3x faster than 8-thread CPU)
2. Add batch GPU search achieving ~250-500x throughput improvement over 8-thread CPU
3. Maintain correctness: GPU-encoded indices must produce equivalent recall to CPU

### Non-Goals

- GPU single-query search (poor utilization at <2% of GPU capacity)
- Full Level 3 kernel fusion replacing cuBLAS GEMM (deferred to N > 1M scale)
- Multi-GPU support
- CPU multi-threaded search (orthogonal improvement)

## 2. Component Dependencies

```
1. GpuMemoryPool          — no dependencies, unblocks everything
2. GPU Scatter Kernel      — depends on pool
3. Fused Encode (L1+L2)   — independent of pool/scatter (kernel-internal)
4. Fastscan Reorder        — fused into scatter kernel, depends on pool
5. GPU Batch Search        — depends on pool + reorder
```

Items 2 and 3 are independent after item 1 completes. Item 4 is implemented inside item 2. Item 5 depends on items 1 and 4.

## 3. Component 1: Pooled Memory Allocator

### Problem

`GpuSaqCluData::allocate()` calls `cudaMalloc` 7 times per segment per cluster. For K=4,096 clusters and 4 segments: ~115,000 `cudaMalloc` calls costing ~900ms.

### Design

A new `GpuMemoryPool` performs one `cudaMalloc` per array type per segment (28 total), then assigns raw pointers into the pool for each cluster.

### Interface

```cpp
// New file: include/saq/gpu/gpu_memory_pool.h

namespace saq::gpu {

struct GpuMemoryPool {
    struct SegmentPool {
        DevicePtr<float>   centroids;          // K * D_seg contiguous
        DevicePtr<float>   factor_o_l2norm;    // total_blocks * 32
        DevicePtr<float>   factor_ip_cent_oa;  // total_blocks * 32
        DevicePtr<uint8_t> short_codes;        // total_blocks * num_codebooks * 32 (GPU layout)
        DevicePtr<uint8_t> long_codes;         // N * long_bytes_per_vec
        DevicePtr<float>   factor_rescale;     // N
        DevicePtr<float>   factor_error;       // N
    };

    DevicePtr<uint32_t> ids;                   // N total
    std::vector<SegmentPool> segments;

    // Host-side offset tables (used by scatter and pointer assignment)
    std::vector<size_t> cluster_offsets;        // [K+1] prefix sum of cluster_sizes
    std::vector<size_t> block_offsets;           // [K+1] prefix sum of blocks per cluster

    // Device-side offset tables (used by scatter kernels)
    DevicePtr<uint32_t> d_cluster_offsets;      // [K+1]
    DevicePtr<uint32_t> d_block_offsets;         // [K+1]

    void allocate(size_t K, const std::vector<size_t>& cluster_sizes,
                  const std::vector<std::pair<size_t,size_t>>& quant_plan);

    void assign_pointers(GpuSaqCluData& clu, size_t c) const;
};

} // namespace saq::gpu
```

### Allocation Logic

```
allocate():
    compute cluster_offsets via prefix sum of cluster_sizes
    compute block_offsets via prefix sum of ceil(cluster_sizes[c] / 32)
    total_vecs = cluster_offsets[K]
    total_blocks = block_offsets[K]

    ids = device_alloc<uint32_t>(total_vecs)

    for each segment s:
        D_seg = quant_plan[s].first
        bits = quant_plan[s].second
        num_codebooks = D_seg / 4

        segments[s].centroids        = device_alloc<float>(K * D_seg)
        segments[s].factor_o_l2norm  = device_alloc<float>(total_blocks * 32)
        segments[s].factor_ip_cent_oa = device_alloc<float>(total_blocks * 32)
        segments[s].short_codes      = device_alloc<uint8_t>(bits ? total_blocks * num_codebooks * 32 : 1)
        segments[s].long_codes       = device_alloc<uint8_t>(long_bytes > 0 ? total_vecs * long_bytes : 1)
        segments[s].factor_rescale   = device_alloc<float>(total_vecs)
        segments[s].factor_error     = device_alloc<float>(total_vecs)

    upload cluster_offsets and block_offsets to d_cluster_offsets, d_block_offsets
```

### Changes to Existing Code

- **`GpuSaqCluData`**: Remove all `owned_*` DevicePtrs and `allocate()` method. Keep `segments` vector with raw pointers, `num_vec`, `num_segments`, `num_blocks`. The pool owns all memory.
- **`GpuIVF`**: Add `GpuMemoryPool pool_` member. Remove `gpu_clusters_` ownership semantics (clusters hold views into pool).
- **`gpu_ivf_construct.cpp`**: Replace per-cluster `allocate()` loop (lines 101-116) with `pool_.allocate()` + per-cluster `pool_.assign_pointers()`.

### Expected Impact

~900ms reduced to <1ms (28 `cudaMalloc` calls instead of ~115,000).

## 4. Component 2: GPU Scatter Kernel

### Problem

The per-cluster `cudaMemcpy` D2D loop (lines 222-259 of `gpu_ivf_construct.cpp`) makes ~16K API calls per segment, costing ~340ms total.

### Design

Replace with one kernel launch per array type per segment. Each kernel processes all N vectors in parallel, writing from flat encode output to pool-allocated per-cluster storage. Fastscan reorder (Component 4) is fused into the short-code scatter.

### Interface

```cpp
// New file: include/saq/gpu/gpu_scatter.cuh

namespace saq::gpu {

// Scatter short codes: linear packed → GPU blocked layout (fused reorder)
void launch_scatter_short_codes(
    const uint8_t* flat_short,         // [N * D_seg/8] linear 1-bit packed
    uint8_t* pool_short,               // pool allocation (GPU blocked layout)
    const uint32_t* d_cluster_offsets,
    const uint32_t* d_block_offsets,
    const uint32_t* d_cluster_ids,
    size_t D_seg, size_t N, size_t num_bits,
    cudaStream_t stream = 0);

// Scatter long codes: flat → per-cluster contiguous in pool
void launch_scatter_long_codes(
    const uint8_t* flat_long,
    uint8_t* pool_long,
    const uint32_t* d_cluster_offsets,
    const uint32_t* d_cluster_ids,
    size_t long_bytes_per_vec, size_t N,
    cudaStream_t stream = 0);

// Scatter factors: flat → per-cluster blocked layout in pool
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

// Copy centroids into pool: simple cudaMemcpy D2D (K * D_seg floats).
// Both source and pool are [K * D_seg] contiguous, so no scatter needed.
void copy_centroids_to_pool(
    const float* d_centroids_seg,      // [K * D_seg]
    float* pool_centroids,             // [K * D_seg] in pool
    size_t D_seg, size_t K,
    cudaStream_t stream = 0);

} // namespace saq::gpu
```

### Preconditions

All scatter kernels require that vectors are **sorted by cluster ID**, with `d_cluster_offsets` being the prefix sum of cluster sizes. This is guaranteed by the sort step in `gpu_ivf_construct.cpp` (lines 70-83). The formula `pos_in_cluster = i - d_cluster_offsets[c]` only yields correct intra-cluster positions when vectors of cluster `c` occupy contiguous positions `[d_cluster_offsets[c], d_cluster_offsets[c+1])` in the flat arrays.

### Kernel Logic (Short Code Scatter + Reorder)

The fused encode produces short codes in **1-bit-per-dimension linear packed** format: the MSB of each dimension's quantization code, packed 8 dimensions per byte, giving `D_seg/8` bytes per vector. The scatter kernel must regroup these into **4-bit codebook entries** for the GPU blocked layout.

```
kernel_scatter_short_codes:
    thread i handles vector i (i < N)
    c = d_cluster_ids[i]
    pos_in_cluster = i - d_cluster_offsets[c]
    block_in_cluster = pos_in_cluster / 32
    vec_in_block = pos_in_cluster % 32
    global_block = d_block_offsets[c] + block_in_cluster
    num_codebooks = D_seg / 4

    // Read D_seg/8 bytes of linear 1-bit-per-dim packed short codes for vector i
    src = flat_short + i * (D_seg / 8)

    // Bit order convention: the fused encode packs in DESCENDING order within each byte
    // (matching the existing kernel_pack_short_codes): dim 0 → bit 7, dim 7 → bit 0.
    // This matches the CPU pack convention.

    // For each codebook cb (group of 4 consecutive dimensions):
    //   dim_base = cb * 4
    //   Extract 4 individual bits from the linear packed bytes (descending order):
    //     bit_j = (src[(dim_base + j) / 8] >> (7 - (dim_base + j) % 8)) & 1   for j in 0..3
    //   Form 4-bit codebook index:
    //     code4 = bit_0 | (bit_1 << 1) | (bit_2 << 2) | (bit_3 << 3)
    //   Write as 1 byte to GPU blocked layout:
    //     pool_short[global_block * 32 * num_codebooks + vec_in_block * num_codebooks + cb] = code4
```

### Kernel Logic (Factor Scatter)

```
kernel_scatter_factors:
    // Precondition: vectors sorted by cluster ID (see above)
    thread i handles vector i (i < N)
    c = d_cluster_ids[i]
    pos_in_cluster = i - d_cluster_offsets[c]
    block_in_cluster = pos_in_cluster / 32
    vec_in_block = pos_in_cluster % 32
    global_block = d_block_offsets[c] + block_in_cluster

    // Factor o_l2norm and ip_cent_oa use blocked layout (blocks of 32)
    blocked_idx = global_block * 32 + vec_in_block
    pool_o_l2norm[blocked_idx]  = flat_o_l2norm[i]
    pool_ip_cent_oa[blocked_idx] = flat_ip_cent_oa[i]

    // Rescale and error use per-vector layout
    pool_rescale[d_cluster_offsets[c] + pos_in_cluster] = flat_rescale[i]
    pool_error[d_cluster_offsets[c] + pos_in_cluster]   = flat_error[i]
```

### Changes to Existing Code

- **`gpu_ivf_construct.cpp`**: Replace per-cluster scatter loop (lines 222-259) with 4 kernel launches per segment.
- **`gpu_packer.cuh/cu`**: Remove `launch_store_factors` (superseded by scatter kernel).

### Expected Impact

~340ms reduced to ~3-5ms (4 kernel launches per segment instead of ~4K API calls).

## 5. Component 3: Fused Encode Kernel (L1 + L2)

### Problem

The per-segment pipeline launches 5 kernels with 3 intermediate global memory buffers (`d_residuals`, `d_rotated`, `d_codes`), producing ~670MB of redundant memory traffic per segment.

### Design

Two fusions that eliminate `d_residuals` and `d_codes`, reducing to 1 cuBLAS GEMM + 1 fused encode kernel with 1 intermediate buffer (`d_rotated`).

### Fusion L1: Eliminate d_residuals

**Observation:** `rotated = (vec_seg - cent_seg) * P = vec_seg * P - cent_seg * P = vec_seg * P - rotated_centroid`.

The rotated centroids are already computed on CPU (lines 169-184). So:
1. Run cuBLAS GEMM directly on raw vector segments (skip `subtract_centroid`)
2. Subtract `rotated_centroid[cluster_id]` inside the encode kernel

For the no-rotation case: subtract centroid directly inside encode from raw vectors (no separate kernel needed).

### Fusion L2: Eliminate d_codes

After CAQ adjustment converges, integer codes are in registers. Instead of writing to `d_codes` then re-reading in pack kernels, pack short and long codes inline.

**Important: DownUpSample ordering.** When `caq_ori_qB > 0`, codes must be right-shifted (`code >>= (caq_ori_qB - num_bits)`) after adjustment but BEFORE packing. In the current separate-kernel flow, DownUpSample writes to `d_codes` and pack reads from it. In the fused version, the shift must happen explicitly between the adjustment loop exit and the packing logic:

```
// Inside encode kernel, after adjustment loop:
// codes[] array is in registers (local per lane)

// DownUpSample: if caq_ori_qB > 0, right-shift codes to target bit width
if (caq_ori_qB > 0) {
    int shift = caq_ori_qB - num_bits;
    for (d = 0; d < dims_per_lane; ++d)
        codes[d] >>= shift;
}

// Pack short: extract MSB, pack 8 per byte (DESCENDING bit order: dim 0 → bit 7)
for (g = 0; g < dims_per_lane / 8; ++g) {
    uint8_t byte = 0;
    for (b = 0; b < 8; ++b)
        byte |= ((codes[g*8 + b] >> (num_bits - 1)) & 1) << (7 - b);
    d_short_raw[vec_idx * short_bytes + lane_offset + g] = byte;
}

// Pack long: extract lower (num_bits-1) bits, bit-compact
// ... similar bit manipulation from register-resident codes ...
```

### Updated Encode Signature

```cpp
void launch_fused_caq_encode(
    const float* d_vectors_rotated,    // GEMM output on raw vectors [N * D_seg]
    const float* d_rotated_centroids,  // [K * D_seg] precomputed
    const uint32_t* d_cluster_ids,
    // Factor outputs (unchanged):
    float* d_o_l2norm, float* d_fac_rescale,
    float* d_fac_error, float* d_ip_cent_oa,
    // Packed code outputs (new — replaces d_codes):
    uint8_t* d_short_raw,
    uint8_t* d_long_raw,
    // Parameters:
    size_t D_seg, size_t N, size_t K,
    size_t num_bits, uint16_t code_max,
    int caq_adj_rd_lmt, float caq_adj_eps, int caq_ori_qB,
    cudaStream_t stream = 0);
```

### No-Rotation Path

When `bdata.rotator` is null, there is no GEMM. The fused encode kernel takes the raw vector slice and centroids directly:

```cpp
void launch_fused_caq_encode_no_rotation(
    const float* d_vectors,            // [N * D] full vectors
    const float* d_centroids,          // [K * D] full centroids
    const uint32_t* d_cluster_ids,
    size_t seg_offset, size_t D_seg, size_t D_total,
    // ... same factor and code outputs ...
    cudaStream_t stream = 0);
```

This kernel subtracts the unrotated centroid segment inline. Note the asymmetry: the rotation variant takes pre-sliced `[K * D_seg]` centroid segments, while the no-rotation variant takes full `[K * D]` centroids with an offset — matching how centroid data is available at each call site.

### Edge Cases

- **`num_bits == 0` segments**: No short or long codes to pack. The fused encode skips packing and only writes factors (`o_l2norm` specifically, since rescale/error/ip_cent_oa are not meaningful for 0-bit segments).
- **`num_bits == 1` segments**: Short codes only (MSB = the only bit). Long codes are empty (0 bytes per vec). The long packing section is skipped.
- **`caq_ori_qB > 0`**: DownUpSample right-shift must happen after adjustment, before packing (see Fusion L2 above).

### Changes to Existing Code

- **`gpu_encoder.cu`**: Replace `kernel_caq_encode` with `kernel_fused_caq_encode` that reads rotated vectors, subtracts rotated centroid, encodes, and packs in one kernel. Add `kernel_fused_caq_encode_no_rotation` variant.
- **`gpu_encoder.cuh`**: Updated signatures.
- **`gpu_packer.cu`**: Remove `kernel_pack_short_codes` and `kernel_pack_long_codes` (folded into encode).
- **`gpu_packer.cuh`**: Remove `launch_pack_short_codes` and `launch_pack_long_codes`.
- **`gpu_ivf_construct.cpp`**: Remove `d_residuals`, `d_codes` allocations. Remove separate subtract/pack kernel calls. Call fused encode + cuBLAS GEMM only.

### Net Result Per Segment

```
Before: 5 kernel launches, 3 intermediate buffers, ~670MB traffic
After:  1 cuBLAS GEMM + 1 fused kernel, 1 intermediate buffer (d_rotated), ~420MB traffic
```

## 6. Component 4: Fastscan Reorder (Fused into Scatter)

### Problem

CPU search uses `vpshufb`-optimized short code layout (nibble-packed, `kPerm0` interleaved). GPU search needs a different layout for shared-memory LUT lookups.

### Design

The GPU short code layout stores one byte per vector per codebook in warp-linear order:

```
For a block of 32 vectors, num_codebooks codebooks:
  offset = block_idx * 32 * num_codebooks + vec_in_block * num_codebooks + cb

Each byte holds the 4-bit code for that vector's codebook (zero-extended to uint8).
32 consecutive threads reading their codebook entries = 1 coalesced transaction.
```

This is 2x the memory of nibble-packed layout (1 byte vs 4 bits per entry). For D_seg=576: 144 codebooks * 32 * sizeof(uint8) = 4,608 bytes per block vs 2,304 packed. Negligible overhead at the index level.

### Implementation

Fused into `kernel_scatter_short_codes` (Component 2). During the scatter copy, each thread reads its linear-packed short codes, extracts 4-bit codebook entries, and writes them in GPU blocked layout. No separate kernel needed.

### Pool Allocation Size

Short code allocation in the pool accounts for the GPU layout:

```
num_codebooks = D_seg / 4
short_code_bytes = bits ? total_blocks * num_codebooks * 32 : 0
```

## 7. Component 5: GPU Batch Search

### Problem

CPU search is single-threaded per query (~40ms at nprobe=200). No GPU search path exists.

### Design

A batch search kernel processes Q queries simultaneously, with one thread block per (query, cluster) pair. Four warps per block process 32-vector blocks using a 3-stage pipeline (variance pruning, shared-memory LUT fastscan, accurate distance).

### Architecture

```
Grid:   dim3(Q, nprobe)         one block per (query, cluster) pair
Block:  128 threads (4 warps)   warps process 32-vec blocks via work-stealing
```

For Q=1000, nprobe=200: 200,000 blocks fully saturating the RTX 5090.

### Interface

```cpp
// In include/saq/gpu/gpu_ivf.h:
void GpuIVF::search_batch(
    const FloatRowMat& queries,        // [Q x D]
    size_t topk,
    size_t nprobe,
    SearcherConfig cfg,
    PID* results);                     // [Q x topk] output, host memory
```

### Host-Side Orchestration

```
search_batch():
    1. For each query: find nprobe nearest centroids via FlatInitializer (CPU, OpenMP)
    2. For each query, each segment: compute rotated_query = query * P_s (CPU, Eigen)
    3. Upload to GPU:
       - d_rotated_queries [Q * sum(D_seg)]
       - d_centroid_ids [Q * nprobe]  (which clusters to search)
       - d_query_constants [Q * num_segments * ...] (delta, sum_vl_lut, q_l2sqr, etc.)
    4. Launch kernel_build_lut_and_search<<<dim3(Q, nprobe), 128>>>(...)
    5. Launch kernel_merge_topk<<<Q, 256>>>(...)
    6. Download results [Q * topk] PID array
```

### Device-Side Descriptor Tables

The search kernel cannot dereference host pointers (`pool.segments[s].short_codes`, `clusters[c].num_blocks`). All per-segment and per-cluster metadata must be marshaled into device-accessible structures, uploaded once before the search loop.

```cpp
// Device-side per-segment descriptor (uploaded as array of num_segments)
struct GpuSegmentDescriptor {
    uint8_t* short_codes;          // base pointer in pool for this segment
    uint8_t* long_codes;           // base pointer in pool
    float* factor_o_l2norm;        // base pointer in pool
    float* factor_ip_cent_oa;      // base pointer in pool
    float* factor_rescale;         // base pointer in pool
    float* factor_error;           // base pointer in pool
    float* centroids;              // base pointer in pool
    size_t num_codebooks;          // D_seg / 4
    size_t D_seg;
    size_t num_bits;
    size_t long_bytes_per_vec;     // D_seg * (bits-1) / 8, or 0 for bits <= 1
};
// Precondition: D_seg is always a multiple of 8 (guaranteed by the DP optimizer
// which allocates in 64-dim blocks, and the padding logic in quantization_plan.h).
// This ensures D_seg/8, D_seg/4, and D_seg*(bits-1)/8 are always integers.

// Device-side per-cluster descriptor (uploaded as array of K)
struct GpuClusterDescriptor {
    size_t num_vec;
    size_t num_blocks;
    uint32_t* ids;                 // pointer into pool.ids
};
```

The `launch_search` wrapper constructs these arrays on the host from the pool and cluster metadata, uploads them once via `cudaMemcpy`, and passes device pointers to the kernel.

### Kernel 1: Build LUT and Search

```cpp
// In include/saq/gpu/gpu_searcher.cuh:

void launch_search(
    // Device-side descriptors (uploaded once):
    const GpuSegmentDescriptor* d_seg_descs,  // [num_segments]
    const GpuClusterDescriptor* d_clu_descs,  // [K]
    const uint32_t* d_block_offsets,           // [K+1] from pool
    const uint32_t* d_cluster_offsets,         // [K+1] from pool
    // Query data:
    const float* d_rotated_queries,    // [Q * sum(D_seg)]
    const float* d_query_constants,    // [Q * num_segments * num_constants]
    const uint32_t* d_centroid_ids,    // [Q * nprobe] which clusters to visit
    // Variance data for stage 1:
    const float* d_data_variance,      // per-segment variance stats
    // Search parameters:
    size_t Q, size_t nprobe, size_t topk,
    size_t num_segments,
    // Output:
    float* d_candidate_dists,          // [Q * nprobe * max_candidates_per_block]
    uint32_t* d_candidate_ids,         // same layout
    uint32_t* d_candidate_counts,      // [Q * nprobe] how many candidates per block
    cudaStream_t stream = 0);
```

### Per-Block Execution

```
Phase 0 — Build LUT in shared memory:
    __shared__ int16_t lut[num_segments][max_codebooks][16];  // ~12KB for D=1536
    __shared__ float q_consts[num_segments][4];  // delta, sum_vl_lut, sum_q, q_l2sqr

    q_idx = blockIdx.x
    cluster_rank = blockIdx.y
    c = d_centroid_ids[q_idx * nprobe + cluster_rank]

    All 128 threads cooperate to fill LUT:
      For each segment s:
        query_seg = d_rotated_queries[q_idx * sum_D_seg + seg_offset_s]
        For each codebook cb (thread cooperatively):
          Compute 16 subset sums of 4 query dims
          (same formula as CPU pack_lut: LUT[j] = LUT[j - lowbit(j)] + query[kPos[j]])
          Quantize to int16 using same scheme as CPU Lut::prepare():
            delta = (max_lut - min_lut) / (2^16 - 0.01)
            lut_u16[j] = (lut_float[j] - min_lut) / delta
          Write to lut[s][cb][0..15]
        Store q_consts[s] = {delta, sum_vl_lut, sum_q, q_l2sqr}
    __syncthreads()

Phase 1+2 — Warp-level 32-vector block processing:
    __shared__ int next_block;  // work-stealing counter
    if (threadIdx.x == 0) next_block = 0;  // initialize before use
    __syncthreads();

    warp_id = threadIdx.x / 32
    lane = threadIdx.x % 32
    distk = FLT_MAX   // per-warp distance threshold, updated as candidates found

    while (true):
        // Claim next block
        if (lane == 0) block_idx = atomicAdd(&next_block, 1)
        block_idx = __shfl_sync(0xFFFFFFFF, block_idx, 0)
        if (block_idx >= d_clu_descs[c].num_blocks) break

        // STAGE 1: Variance pruning
        // One lane computes lower bound for block
        // All 32 lanes take same branch (no warp divergence)
        if (lower_bound > distk) continue

        // STAGE 2: Shared-memory LUT fastscan
        // lane = vector index within 32-vec block
        // global_block = d_block_offsets[c] + block_idx (used for short codes AND factors)
        global_block = d_block_offsets[c] + block_idx

        float approx_dist = 0
        for each segment s:
            seg = d_seg_descs[s]
            short_code_base = seg.short_codes + global_block * 32 * seg.num_codebooks
            for each codebook cb:
                code4 = short_code_base[lane * seg.num_codebooks + cb]
                approx_dist += lut[s][cb][code4]  // shared memory read

            // Read o_l2norm from blocked factor layout (same global_block index):
            o_l2norm = seg.factor_o_l2norm[global_block * 32 + lane]
            // Apply scaling: delta, sum_vl_lut, o_l2norm
            // Convert to distance estimate

        // Warp ballot: which vectors beat threshold?
        uint32_t candidates = __ballot_sync(0xFFFFFFFF, approx_dist < distk)
        if (candidates == 0) continue

        // STAGE 3: Accurate distance (candidates only)
        // Skip for segments with num_bits <= 1 (no long codes)
        if (candidates & (1 << lane)):
            // Decode long codes for my vector across all segments with bits > 1
            // Compute full-precision IP using all bits
            // Formula: ip = fast_ip + long_ip * delta + (vl + delta/2) * sum_q
            //          dist_L2 = o_l2sqr + q_l2sqr - 2 * ip * rescale * o_l2norm
            my_dist = computed_distance
        else:
            my_dist = FLT_MAX

        // Update warp-local distk via warp reduce:
        float warp_best = warp_reduce_min(my_dist)  // min across 32 lanes
        if (warp_best < distk):
            distk = warp_best  // all lanes update (broadcast via reduce)
            // Insert candidates with dist < distk into warp-local buffer
            // Buffer is bounded at max_candidates_per_warp (e.g., topk)
            // When full, evict worst candidate and tighten distk further

Phase 3 — Output:
    Merge 4 warps' candidates into per-block output buffer
    Write to d_candidate_dists/ids at [q_idx * nprobe + cluster_rank] slot
    Write count to d_candidate_counts
```

### Kernel 2: Top-K Merge

```cpp
void launch_merge_topk(
    const float* d_candidate_dists,
    const uint32_t* d_candidate_ids,
    const uint32_t* d_candidate_counts,
    PID* d_results,                    // [Q * topk] output
    float* d_result_dists,             // [Q * topk] output (optional)
    size_t Q, size_t nprobe, size_t topk,
    size_t max_candidates_per_block,
    cudaStream_t stream = 0);
```

One block per query. Loads all candidates from nprobe clusters (~200 × ~5 candidates = ~1,000 entries typical, ~4,800 worst case), performs partial sort (radix select or bitonic sort) to find top-K, writes final results.

Each (query, cluster) search block writes at most `max_candidates_per_block` results (e.g., `topk` or a fixed cap like 64). Enforcement: the per-warp candidate buffer has fixed capacity; when full, the worst candidate is evicted and `distk` tightened. The output buffer is pre-allocated as `Q * nprobe * max_candidates_per_block` entries. The merge kernel reads `d_candidate_counts[q * nprobe + cluster_rank]` to know how many valid entries exist per slot.

### Accurate Distance Device Function

```cpp
__device__ float gpu_long_code_ip(
    const float* query_seg,           // query segment in registers/shared
    const uint8_t* long_code,         // packed (bits-1) per dim
    size_t D_seg, size_t num_bits);
```

Unpacks variable-bit long codes (switch on `num_bits` for 1-8 bit paths), multiplies by query, accumulates. Full distance:

```
ip_full = ip_fast + gpu_long_code_ip(...) * delta + (vl + delta/2) * sum_q
dist_L2 = o_l2sqr + q_l2sqr - 2 * ip_full * rescale * o_l2norm
```

### Shared Memory Budget

Per block (D=1536, 4 segments). Total codebooks = sum(D_seg_i) / 4 = 1536/4 = 384 regardless of segment split:

| Data | Size |
|------|------|
| LUT: 384 codebooks × 16 entries × 2 bytes | 12 KB |
| Query constants: 4 segs × 4 floats | 64 B |
| Work counter + warp buffers | ~1 KB |
| **Total** | **~13 KB** |

RTX 5090: 128KB shared memory per SM. At 13KB/block, up to 9 blocks/SM — excellent occupancy.

### Search Edge Cases

- **0-bit segments**: No short codes, no LUT entries. Skip this segment in stage 2. Stage 3 long decode also skipped. Only factor `o_l2norm` contributes to distance.
- **1-bit segments**: Short codes only, no long codes. Stage 2 processes normally. Stage 3 skips long code decode for this segment.
- **Empty clusters** (size=0): `d_clu_descs[c].num_blocks == 0`, so the work-stealing loop exits immediately. No wasted compute.
- **`distk` scope**: `distk` is per-warp, initialized to `FLT_MAX`. It is NOT shared across warps within a block or across blocks. Cross-warp sharing would require shared-memory atomics with potential serialization overhead that outweighs the pruning benefit — at ~24 vectors per cluster (~1 block), most blocks are processed by a single warp anyway. Cross-block sharing (across the nprobe=200 clusters for a query) would require global atomics and is intentionally omitted: the merge kernel handles cross-cluster top-K selection.
- **Pool `allocate()` idempotency**: `allocate()` is single-use. The method should assert `segments.empty()` at entry to fail loudly on accidental re-allocation. For index rebuild, destroy and recreate the `GpuMemoryPool`.

### Precomputed Data Uploaded Once

Before the search loop, upload and keep resident:
- `d_data_variance`: per-segment variance for stage 1 pruning
- Segment metadata (dims, bits, offsets): small constant buffer

Per batch upload:
- `d_rotated_queries`: Q × sum(D_seg) floats. For Q=1000: ~9MB
- `d_centroid_ids`: Q × nprobe uint32s. ~800KB
- `d_query_constants`: Q × num_segments × 4 floats. ~64KB

## 8. File Changes Summary

### New Files

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `include/saq/gpu/gpu_memory_pool.h` | Pool struct + allocate/assign | ~150 |
| `include/saq/gpu/gpu_scatter.cuh` | Scatter + reorder kernel declarations | ~60 |
| `src/gpu/gpu_scatter.cu` | Scatter + reorder kernel implementations | ~250 |
| `include/saq/gpu/gpu_searcher.cuh` | Search device functions + launch wrappers | ~200 |
| `src/gpu/gpu_search.cu` | Search kernel implementations | ~500 |

### Modified Files

| File | Change |
|------|--------|
| `include/saq/gpu/gpu_cluster_data.cuh` | Remove `owned_*` DevicePtrs and `allocate()`. Keep segment struct with raw pointers. |
| `include/saq/gpu/gpu_ivf.h` | Add `GpuMemoryPool pool_` member. Add `search_batch()` method. |
| `src/gpu/gpu_ivf_construct.cpp` | Use pool, call fused encode, call scatter kernels. |
| `src/gpu/gpu_encoder.cu` | Replace encode kernel with fused version (subtract + encode + pack). |
| `include/saq/gpu/gpu_encoder.cuh` | Updated `launch_fused_caq_encode` signature. |
| `src/gpu/gpu_packer.cu` | Delete entirely. Pack kernels folded into fused encode; `launch_store_factors` replaced by scatter kernel. |
| `include/saq/gpu/gpu_packer.cuh` | Delete entirely. All declarations moved to `gpu_encoder.cuh` (fused pack) or `gpu_scatter.cuh`. |
| `src/CMakeLists.txt` | Add `gpu_scatter.cu`, `gpu_search.cu`. |
| `samples/gpu_benchmark_sample.cpp` | Add search benchmark comparing GPU batch vs CPU. |

## 9. Testing Strategy

### Encode Correctness

Compare GPU-encoded index against CPU-encoded index by downloading GPU pool data and comparing codes/factors element-wise. Allow for floating-point rounding differences (~1 ULP for factors, exact match for integer codes in most cases).

### Search Correctness

Run identical queries through CPU `IVF::search` and GPU `GpuIVF::search_batch`. Compare recall@K — should be within 0.1% (minor differences from float rounding in distance computation are acceptable).

### Performance Benchmarks

Extend `gpu_benchmark_sample.cpp`:
- Encode: report pool alloc time, scatter time, kernel time, total (compare to CPU 8T)
- Search: report batch search time for Q=1, 10, 100, 1000 at nprobe=200 (compare to CPU 1T and 8T)

## 10. Expected Performance

### Encode (N=99K, D=1536, K=4096)

| Phase | Current (ms) | Optimized (ms) |
|-------|-------------|---------------|
| CPU prep | 77 | 77 |
| H2D upload | 120 | 120 |
| GPU alloc (pool) | 900 | <1 |
| GPU kernels (fused) | 118 | ~100 |
| GPU scatter | 340 | ~3 |
| **Total** | **1,555** | **~301** |
| **CPU 8T** | **~500** | — |
| **Speedup** | 0.32x | **~1.7x** |

### Search (DBpedia 100K, nprobe=200)

| Configuration | Time | Throughput |
|--------------|------|-----------|
| CPU 1T, Q=1 | 40ms | 25 QPS |
| CPU 8T, Q=1 (not yet impl) | ~5ms | ~200 QPS |
| GPU, Q=1 | ~0.3ms | ~3,300 QPS |
| GPU, Q=100 | ~3ms | ~33,000 QPS |
| GPU, Q=1000 | ~15ms | ~67,000 QPS |

GPU search estimates assume data resident on GPU and exclude host-side centroid search + query rotation prep time.
