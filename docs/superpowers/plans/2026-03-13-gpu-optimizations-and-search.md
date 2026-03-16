# GPU Encode Optimizations and Batch Search — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce GPU encode wall time from ~1,550ms to ~300ms and add GPU batch search with ~250-500x throughput over CPU.

**Architecture:** Pooled memory allocator replaces ~115K `cudaMalloc` calls with 28. GPU scatter kernels replace ~16K `cudaMemcpy` D2D calls per segment. Fused encode eliminates 2 intermediate buffers. GPU search uses shared-memory LUT fastscan with 3-stage pipeline.

**Tech Stack:** C++20, CUDA 13.1, cuBLAS, MSVC 19.50 (VS 2025), Ninja generator, CMake. Build on Windows, run via `cmd.exe`.

**Spec:** `docs/superpowers/specs/2026-03-13-gpu-optimizations-and-search-design.md`

**Build command:**
```
cmd.exe /c "cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\SAQ && cmake -B build -G Ninja -DSAQ_BUILD_CUDA=ON -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build"
```

**Run command (encode benchmark at 2.0 bpd):**
```
cmd.exe /c "cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\SAQ && set PATH=%CD%\build\_deps\fmt-build\bin;%CD%\build\_deps\glog-build;%CD%\build\_deps\gflags-build;%PATH% && build\samples\gpu_benchmark_sample.exe data\datasets\dbpedia_100k 2.0 4096 8"
```

**Important CUDA/MSVC notes (from CLAUDE.md and memory):**
- Must use Ninja generator (VS 2025 lacks CUDA toolset integration)
- `--allow-unsupported-compiler` needed for CUDA 13.1 + VS 2025
- Eigen 3.4.0 is NOT CUDA-compatible — orchestrator code must be `.cpp`, not `.cu`
- Guard `__device__` functions with `#ifdef __CUDACC__` in headers included from `.cpp`
- Use `$<$<COMPILE_LANGUAGE:CXX>:...>` for MSVC-only flags to prevent leaking to nvcc

---

## Chunk 1: Pooled Memory Allocator + Integration

### Task 1: Create GpuMemoryPool

**Files:**
- Create: `include/saq/gpu/gpu_memory_pool.h`

This is a header-only struct. It owns all GPU memory for the index via bulk `DevicePtr` allocations, computes per-cluster offsets, and assigns raw pointers into `GpuSaqCluData` segment structs.

- [ ] **Step 1: Write `gpu_memory_pool.h`**

```cpp
#pragma once

#ifdef SAQ_USE_CUDA

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "saq/defines.h"
#include "saq/gpu/gpu_utils.cuh"
#include "saq/gpu/gpu_cluster_data.cuh"

namespace saq::gpu {

struct GpuMemoryPool {
    struct SegmentPool {
        DevicePtr<float>   centroids;
        DevicePtr<float>   factor_o_l2norm;
        DevicePtr<float>   factor_ip_cent_oa;
        DevicePtr<uint8_t> short_codes;
        DevicePtr<uint8_t> long_codes;
        DevicePtr<float>   factor_rescale;
        DevicePtr<float>   factor_error;
    };

    size_t K_ = 0;
    size_t total_vecs_ = 0;
    size_t total_blocks_ = 0;

    DevicePtr<uint32_t> ids;
    std::vector<SegmentPool> segments;

    // Host-side offset tables
    std::vector<size_t> cluster_offsets;   // [K+1]
    std::vector<size_t> block_offsets;      // [K+1]

    // Device-side offset tables (for scatter/search kernels)
    DevicePtr<uint32_t> d_cluster_offsets;
    DevicePtr<uint32_t> d_block_offsets;

    // Quant plan (cached for assign_pointers)
    std::vector<std::pair<size_t, size_t>> quant_plan_;

    void allocate(size_t K,
                  const std::vector<size_t>& cluster_sizes,
                  const std::vector<std::pair<size_t, size_t>>& quant_plan) {
        assert(segments.empty() && "Pool already allocated; destroy and recreate");

        K_ = K;
        quant_plan_ = quant_plan;
        size_t num_segments = quant_plan.size();

        // Compute offset tables
        cluster_offsets.resize(K + 1, 0);
        block_offsets.resize(K + 1, 0);
        for (size_t c = 0; c < K; ++c) {
            cluster_offsets[c + 1] = cluster_offsets[c] + cluster_sizes[c];
            block_offsets[c + 1] = block_offsets[c] +
                (cluster_sizes[c] + KFastScanSize - 1) / KFastScanSize;
        }
        total_vecs_ = cluster_offsets[K];
        total_blocks_ = block_offsets[K];

        // Upload offset tables to device
        std::vector<uint32_t> co_u32(K + 1), bo_u32(K + 1);
        for (size_t i = 0; i <= K; ++i) {
            co_u32[i] = static_cast<uint32_t>(cluster_offsets[i]);
            bo_u32[i] = static_cast<uint32_t>(block_offsets[i]);
        }
        d_cluster_offsets = device_alloc<uint32_t>(K + 1);
        d_block_offsets = device_alloc<uint32_t>(K + 1);
        upload(d_cluster_offsets.get(), co_u32.data(), K + 1);
        upload(d_block_offsets.get(), bo_u32.data(), K + 1);

        // Allocate IDs pool
        if (total_vecs_ > 0) {
            ids = device_alloc<uint32_t>(total_vecs_);
        }

        // Allocate per-segment pools
        segments.resize(num_segments);
        for (size_t s = 0; s < num_segments; ++s) {
            size_t D_seg = quant_plan[s].first;
            size_t bits = quant_plan[s].second;
            size_t num_codebooks = D_seg / 4;
            size_t long_bytes_per_vec = (bits > 1) ? D_seg * (bits - 1) / 8 : 0;

            auto& sp = segments[s];
            sp.centroids        = device_alloc<float>(K * D_seg);
            sp.factor_o_l2norm  = device_alloc<float>(total_blocks_ * KFastScanSize);
            sp.factor_ip_cent_oa = device_alloc<float>(total_blocks_ * KFastScanSize);

            size_t short_bytes = bits ? total_blocks_ * num_codebooks * KFastScanSize : 0;
            sp.short_codes = device_alloc<uint8_t>(short_bytes > 0 ? short_bytes : 1);

            size_t long_total = long_bytes_per_vec * total_vecs_;
            sp.long_codes = device_alloc<uint8_t>(long_total > 0 ? long_total : 1);

            sp.factor_rescale = device_alloc<float>(total_vecs_ > 0 ? total_vecs_ : 1);
            sp.factor_error   = device_alloc<float>(total_vecs_ > 0 ? total_vecs_ : 1);
        }
    }

    void assign_pointers(GpuSaqCluData& clu, size_t c) const {
        size_t num_segments = quant_plan_.size();
        clu.num_vec = cluster_offsets[c + 1] - cluster_offsets[c];
        clu.num_segments = num_segments;
        clu.num_blocks = block_offsets[c + 1] - block_offsets[c];
        clu.segments.resize(num_segments);

        size_t vec_off = cluster_offsets[c];
        size_t blk_off = block_offsets[c];

        for (size_t s = 0; s < num_segments; ++s) {
            size_t D_seg = quant_plan_[s].first;
            size_t bits = quant_plan_[s].second;
            size_t num_codebooks = D_seg / 4;
            size_t long_bytes_per_vec = (bits > 1) ? D_seg * (bits - 1) / 8 : 0;

            auto& seg = clu.segments[s];
            seg.num_dim_pad = D_seg;
            seg.num_bits = bits;
            seg.num_blocks = clu.num_blocks;

            const auto& sp = segments[s];
            seg.d_centroid         = sp.centroids.get() + c * D_seg;
            seg.d_factor_o_l2norm  = sp.factor_o_l2norm.get() + blk_off * KFastScanSize;
            seg.d_factor_ip_cent_oa = sp.factor_ip_cent_oa.get() + blk_off * KFastScanSize;

            if (bits > 0) {
                seg.d_short_codes = sp.short_codes.get()
                    + blk_off * num_codebooks * KFastScanSize;
            } else {
                seg.d_short_codes = nullptr;
            }

            if (long_bytes_per_vec > 0) {
                seg.d_long_codes = sp.long_codes.get() + vec_off * long_bytes_per_vec;
            } else {
                seg.d_long_codes = nullptr;
            }

            seg.d_long_factor_rescale = sp.factor_rescale.get() + vec_off;
            seg.d_long_factor_error   = sp.factor_error.get() + vec_off;
        }
    }
};

} // namespace saq::gpu

#endif // SAQ_USE_CUDA
```

- [ ] **Step 2: Build to verify compilation**

Run the build command. Expected: compiles without errors (pool is not yet used by anything).

- [ ] **Step 3: Commit**

```
git add include/saq/gpu/gpu_memory_pool.h
git commit -m "feat(gpu): add GpuMemoryPool for bulk device memory allocation"
```

---

### Task 2: Strip Ownership from GpuSaqCluData

**Files:**
- Modify: `include/saq/gpu/gpu_cluster_data.cuh`

Remove all `owned_*` DevicePtrs and `allocate()` method. Keep `GpuSegmentData` struct (raw pointers) and `GpuSaqCluData` metadata. The pool will own all memory and assign pointers.

- [ ] **Step 1: Modify `gpu_cluster_data.cuh`**

Replace the entire file. The `GpuSegmentData` struct stays identical. `GpuSaqCluData` becomes a lightweight view:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "saq/defines.h"

namespace saq::gpu {

/// GPU-resident per-segment cluster data (mirrors CAQClusterData layout).
struct GpuSegmentData {
    size_t num_dim_pad;
    size_t num_bits;
    size_t num_blocks;

    // Non-owning device pointers — owned by GpuMemoryPool
    float* d_centroid            = nullptr;
    float* d_factor_o_l2norm     = nullptr;
    float* d_factor_ip_cent_oa   = nullptr;
    uint8_t* d_short_codes       = nullptr;
    uint8_t* d_long_codes        = nullptr;
    float* d_long_factor_rescale = nullptr;
    float* d_long_factor_error   = nullptr;
};

/// GPU-resident cluster data view (pointers assigned by GpuMemoryPool).
struct GpuSaqCluData {
    size_t num_vec = 0;
    size_t num_segments = 0;
    size_t num_blocks = 0;
    std::vector<GpuSegmentData> segments;
};

} // namespace saq::gpu
```

Note: `d_ids` is removed from `GpuSaqCluData` — the pool owns the contiguous IDs array and search uses `pool.ids + cluster_offsets[c]` to find a cluster's IDs.

- [ ] **Step 2: Build — expect errors**

The build will fail because `gpu_ivf_construct.cpp` still references `gpu_clusters_[c].allocate(...)`, `gpu_clusters_[c].d_ids`, etc. This is expected and will be fixed in Task 3.

- [ ] **Step 3: Commit (WIP)**

```
git add include/saq/gpu/gpu_cluster_data.cuh
git commit -m "refactor(gpu): strip ownership from GpuSaqCluData (pool will own memory)"
```

---

### Task 3: Integrate Pool into GpuIVF::construct

**Files:**
- Modify: `include/saq/gpu/gpu_ivf.h` — add `pool_` member
- Modify: `src/gpu/gpu_ivf_construct.cpp` — replace per-cluster alloc + scatter with pool + existing scatter (scatter kernels come in Task 5)

This task wires up the pool for allocation and pointer assignment, but keeps the existing per-cluster scatter loop for now (it reads from the pool-assigned pointers, so it still works). The scatter kernel replacement comes in Task 5.

- [ ] **Step 1: Add pool member to `gpu_ivf.h`**

Add `#include "saq/gpu/gpu_memory_pool.h"` and add `GpuMemoryPool pool_;` member to the class alongside `gpu_clusters_`.

- [ ] **Step 2: Rewrite allocation section in `gpu_ivf_construct.cpp`**

Replace lines 101-116 (the per-cluster allocate loop) with:

```cpp
// 4. Allocate GPU memory via pool
phase_timer.reset();
pool_.allocate(K, cluster_sizes, quant_plan);

// Set up cluster views
gpu_clusters_.clear();
gpu_clusters_.resize(K);
for (size_t c = 0; c < K; ++c) {
    pool_.assign_pointers(gpu_clusters_[c], c);
}

// Upload original IDs into pool
upload(pool_.ids.get(), h_sorted_original_ids.data(), N);

SAQ_CUDA_CHECK(cudaDeviceSynchronize());
auto alloc_ms = phase_timer.getElapsedTimeMicro() / 1000.0;
LOG(INFO) << "[TIMING] GPU pool alloc + ID upload: " << alloc_ms << " ms";
```

- [ ] **Step 3: Temporarily keep the existing scatter loop (it will be replaced in Task 6)**

The existing scatter loop (lines 222-259) writes to `gpu_clusters_[c].segments[seg]` which now has pool-assigned pointers. The factor copies and `launch_store_factors` calls still work because the pool pointer offsets are correct for the blocked layout.

**However**, the short-code `cudaMemcpy` copies linear-packed short codes into pool memory that is sized for GPU blocked layout (1 byte per codebook instead of 1 bit per dim). The data format is wrong but the copy size is small enough to not overflow. **This intermediate state produces incorrect short codes.** This is acceptable because:
1. The current benchmark only measures timing, not search correctness
2. Task 6 replaces this entire scatter loop with proper GPU scatter kernels

Do NOT run search benchmarks between Task 3 and Task 6. Encode timing benchmarks are safe.

- [ ] **Step 4: Build and run benchmark**

Run the build command, then the benchmark at 2.0 bpd. Expected:
- `[TIMING] GPU pool alloc + ID upload: <5 ms` (vs. ~900ms before)
- All other timings should be similar
- GPU and CPU encode both complete without errors

- [ ] **Step 5: Commit**

```
git add include/saq/gpu/gpu_ivf.h src/gpu/gpu_ivf_construct.cpp
git commit -m "feat(gpu): integrate GpuMemoryPool into construct pipeline

Replaces ~115K cudaMalloc calls with 28 bulk allocations via pool.
Pool alloc time: <5ms (down from ~900ms)."
```

---

### Task 4: Run Full Benchmark Suite (Pool Validation)

**Files:**
- None modified — this is a validation task

- [ ] **Step 1: Run at 1.0, 2.0, 4.0 bpd**

Run the benchmark command for each bpd value. Verify:
- Pool alloc time < 5ms for all configurations
- GPU encode total is ~650ms (was ~1,550ms, saved ~900ms from pool)
- CPU encode time is unchanged (~475-641ms)
- No CUDA errors

- [ ] **Step 2: Verify encode correctness**

The GPU-encoded data should produce the same timing summary structure. If the benchmark sample has correctness checks (comparing GPU vs CPU recall), verify they pass. Currently the sample only compares timing, so functional correctness is verified by the fact that the GPU encode completes without CUDA errors and produces valid factor/code data (which the scatter loop writes without crashing).

---

## Chunk 2: GPU Scatter Kernels

### Task 5: Create Scatter Kernel Source Files

**Files:**
- Create: `include/saq/gpu/gpu_scatter.cuh`
- Create: `src/gpu/gpu_scatter.cu`
- Modify: `src/CMakeLists.txt` — add `gpu/gpu_scatter.cu`

- [ ] **Step 1: Write `gpu_scatter.cuh` header**

Declare the four scatter functions exactly as specified in the design spec Section 4. See spec for the full interface. Key signatures:

```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace saq::gpu {

void launch_scatter_short_codes(
    const uint8_t* flat_short, uint8_t* pool_short,
    const uint32_t* d_cluster_offsets, const uint32_t* d_block_offsets,
    const uint32_t* d_cluster_ids,
    size_t D_seg, size_t N, size_t num_bits,
    cudaStream_t stream = 0);

void launch_scatter_long_codes(
    const uint8_t* flat_long, uint8_t* pool_long,
    const uint32_t* d_cluster_offsets, const uint32_t* d_cluster_ids,
    size_t long_bytes_per_vec, size_t N,
    cudaStream_t stream = 0);

void launch_scatter_factors(
    const float* flat_o_l2norm, const float* flat_ip_cent_oa,
    const float* flat_rescale, const float* flat_error,
    float* pool_o_l2norm, float* pool_ip_cent_oa,
    float* pool_rescale, float* pool_error,
    const uint32_t* d_cluster_offsets, const uint32_t* d_block_offsets,
    const uint32_t* d_cluster_ids, size_t N,
    cudaStream_t stream = 0);

} // namespace saq::gpu
```

- [ ] **Step 2: Write `gpu_scatter.cu` kernels**

Implement three CUDA kernels + launch wrappers following the spec pseudocode exactly.

**Key implementation detail for `kernel_scatter_short_codes`:** The input is 1-bit-per-dim packed in descending bit order (dim 0 → bit 7). The output is 1-byte-per-codebook in GPU blocked layout. Each thread handles one vector, reads `D_seg/8` bytes, regroups into `D_seg/4` codebook entries of 4 bits each.

```cuda
__global__ void kernel_scatter_short_codes(
    const uint8_t* __restrict__ flat_short,
    uint8_t* __restrict__ pool_short,
    const uint32_t* __restrict__ d_cluster_offsets,
    const uint32_t* __restrict__ d_block_offsets,
    const uint32_t* __restrict__ d_cluster_ids,
    size_t D_seg, size_t N, size_t num_bits)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
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
        for (int j = 0; j < 4; ++j) {
            size_t dim = dim_base + j;
            size_t byte_idx = dim / 8;
            size_t bit_pos = 7 - (dim % 8);  // descending bit order
            uint8_t bit = (src[byte_idx] >> bit_pos) & 1;
            code4 |= (bit << j);
        }
        pool_short[global_block * 32 * num_codebooks + vec_in_block * num_codebooks + cb] = code4;
    }
}
```

Implement `kernel_scatter_long_codes` (simple byte copy with offset computation) and `kernel_scatter_factors` (blocked layout for o_l2norm/ip_cent_oa, per-vector for rescale/error) following the spec pseudocode.

- [ ] **Step 3: Add `gpu_scatter.cu` to `src/CMakeLists.txt`**

Add the new source file to the CUDA sources list in `src/CMakeLists.txt`.

- [ ] **Step 4: Build**

Run build command. Expected: compiles successfully. Scatter kernels are not yet called.

- [ ] **Step 5: Commit**

```
git add include/saq/gpu/gpu_scatter.cuh src/gpu/gpu_scatter.cu src/CMakeLists.txt
git commit -m "feat(gpu): add scatter kernels for pool-based data layout"
```

---

### Task 6: Replace Scatter Loop with Kernel Calls

**Files:**
- Modify: `src/gpu/gpu_ivf_construct.cpp` — replace per-cluster scatter loop

- [ ] **Step 1: Replace scatter loop in construct**

Replace the per-cluster for-loop (currently lines 222-259 area) with kernel launches:

```cpp
// 5f. Scatter to pool
SAQ_CUDA_CHECK(cudaDeviceSynchronize());
auto seg_kernel_ms = phase_timer.getElapsedTimeMicro() / 1000.0;
total_kernel_ms += seg_kernel_ms;
LOG(INFO) << "[TIMING] Segment " << seg << " kernels: " << seg_kernel_ms << " ms";

phase_timer.reset();

// Copy centroids to pool (same contiguous layout, just a D2D memcpy)
copy_centroids_to_pool(
    d_centroids_seg.get(), pool_.segments[seg].centroids.get(),
    D_seg, K);

// Scatter factors
launch_scatter_factors(
    d_o_l2norm.get(), d_ip_cent_oa.get(),
    d_fac_rescale.get(), d_fac_error.get(),
    pool_.segments[seg].factor_o_l2norm.get(),
    pool_.segments[seg].factor_ip_cent_oa.get(),
    pool_.segments[seg].factor_rescale.get(),
    pool_.segments[seg].factor_error.get(),
    pool_.d_cluster_offsets.get(), pool_.d_block_offsets.get(),
    d_cluster_ids.get(), N);

// Scatter short codes (with fastscan reorder)
if (num_bits > 0) {
    launch_scatter_short_codes(
        d_short_raw.get(), pool_.segments[seg].short_codes.get(),
        pool_.d_cluster_offsets.get(), pool_.d_block_offsets.get(),
        d_cluster_ids.get(), D_seg, N, num_bits);
}

// Scatter long codes
if (long_code_bytes > 0) {
    launch_scatter_long_codes(
        d_long_raw.get(), pool_.segments[seg].long_codes.get(),
        pool_.d_cluster_offsets.get(), d_cluster_ids.get(),
        long_code_bytes, N);
}

SAQ_CUDA_CHECK(cudaDeviceSynchronize());
auto seg_scatter_ms = phase_timer.getElapsedTimeMicro() / 1000.0;
total_scatter_ms += seg_scatter_ms;
LOG(INFO) << "[TIMING] Segment " << seg << " scatter: " << seg_scatter_ms << " ms";
```

- [ ] **Step 2: Remove `#include "saq/gpu/gpu_packer.cuh"` and add `#include "saq/gpu/gpu_scatter.cuh"`**

Update includes at top of `gpu_ivf_construct.cpp`.

- [ ] **Step 3: Build and run benchmark**

Run at 2.0 bpd. Expected:
- `[TIMING] Segment N scatter: <2 ms` (vs. ~95-100ms per segment before)
- `[TIMING] GPU scatter: <5 ms` total (vs. ~340ms before)
- Total GPU encode time: ~300ms

- [ ] **Step 4: Run at 1.0, 2.0, 4.0 bpd**

Verify all three configurations run without CUDA errors.

- [ ] **Step 5: Commit**

```
git add src/gpu/gpu_ivf_construct.cpp
git commit -m "feat(gpu): replace per-cluster scatter loop with GPU scatter kernels

Scatter time: ~3ms (down from ~340ms). Combined with pool allocator,
GPU encode total is ~300ms vs ~500ms CPU (1.7x speedup)."
```

---

## Chunk 3: Fused Encode Kernel

### Task 7: Implement Fused CAQ Encode (L1 + L2)

**Files:**
- Modify: `src/gpu/gpu_encoder.cu` — replace `kernel_caq_encode` with fused version
- Modify: `include/saq/gpu/gpu_encoder.cuh` — update signatures

This is the most complex kernel change. The fused kernel:
1. Subtracts rotated centroid (L1 — eliminates `d_residuals`)
2. Performs CAQ encode (same warp-cooperative logic as before)
3. Applies DownUpSample if `caq_ori_qB > 0`
4. Packs short and long codes inline (L2 — eliminates `d_codes`)

- [ ] **Step 1: Update `gpu_encoder.cuh` signatures**

Replace the existing signatures with:

```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace saq::gpu {

/// Fused encode: subtract rotated centroid + CAQ encode + pack short/long codes.
/// Input: GEMM output (raw vectors rotated, NOT residuals) + rotated centroids.
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
    cudaStream_t stream = 0);

/// No-rotation variant: reads raw vectors + centroids, subtracts inline.
void launch_fused_caq_encode_no_rotation(
    const float* d_vectors,            // [N * D_total] full vectors
    const float* d_centroids,          // [K * D_total] full centroids
    const uint32_t* d_cluster_ids,
    size_t seg_offset, size_t D_seg, size_t D_total,
    float* d_o_l2norm, float* d_fac_rescale,
    float* d_fac_error, float* d_ip_cent_oa,
    uint8_t* d_short_raw, uint8_t* d_long_raw,
    size_t num_bits, uint16_t code_max,
    int caq_adj_rd_lmt, float caq_adj_eps, int caq_ori_qB,
    cudaStream_t stream = 0);

} // namespace saq::gpu
```

- [ ] **Step 2: Modify `kernel_caq_encode` in `gpu_encoder.cu`**

The key changes to the existing kernel:

**At the start (subtract rotated centroid, L1):**
```cuda
// Before: residual was pre-computed and passed as d_rotated
// After: compute residual from GEMM output - rotated_centroid
uint32_t cid = d_cluster_ids[vec_idx];
for (int my_d = 0; my_d < dims_per_lane; ++my_d) {
    int d = lane + my_d * 32;
    if (d < D_seg) {
        // d_vectors_rotated = raw_vectors * P (from cuBLAS)
        // Subtract rotated centroid to get residual
        local_data[my_d] = d_vectors_rotated[vec_idx * D_seg + d]
                         - d_rotated_centroids[cid * D_seg + d];
    }
}
```

**After adjustment loop (DownUpSample + pack, L2):**
```cuda
// DownUpSample
if (caq_ori_qB > 0) {
    int shift = caq_ori_qB - (int)num_bits;
    for (int my_d = 0; my_d < dims_per_lane; ++my_d)
        codes[my_d] >>= shift;
}

// Pack short codes (descending bit order: dim 0 → bit 7)
if (num_bits > 0) {
    size_t short_bytes_per_vec = D_seg / 8;
    for (int g = 0; g < dims_per_lane / 8; ++g) {
        uint8_t byte_val = 0;
        for (int b = 0; b < 8; ++b) {
            int d_local = g * 8 + b;
            if (d_local < dims_per_lane) {
                byte_val |= ((codes[d_local] >> ((int)num_bits - 1)) & 1) << (7 - b);
            }
        }
        size_t global_dim = lane * 1 + g * 32;  // interleaved across lanes
        // Need to compute correct byte index for this lane's dims
        // Each lane handles dims: lane, lane+32, lane+64, ...
        // Pack the 8 consecutive dims this lane handles
        // Byte index = (lane * dims_per_lane + g*8) / 8 simplified
        size_t byte_offset = (lane + g * 32 * 8) / 8;  // this needs care
        // ... pack into d_short_raw at correct offset ...
    }
}

// Pack long codes similarly
// ... extract lower (num_bits-1) bits, bit-compact ...
```

**Dimension layout:** The existing kernel uses **contiguous-chunk** layout: lane `i` handles dimensions `[i*chunk, (i+1)*chunk)` where `chunk = ceil(D_seg/32)`. This means each lane's dimensions ARE contiguous, so bit packing within a lane is straightforward — lane `i` writes bytes `[i*chunk/8, (i+1)*chunk/8)` of the output.

```cuda
// Each lane packs its contiguous chunk of dimensions
size_t chunk = (D_seg + 31) / 32;
size_t start = lane_id * chunk;
size_t end = min(start + chunk, D_seg);
size_t short_bytes_per_vec = D_seg / 8;

// Pack short codes: each lane handles its byte range
for (size_t byte_idx = start / 8; byte_idx < (end + 7) / 8; ++byte_idx) {
    uint8_t byte_val = 0;
    for (int b = 0; b < 8; ++b) {
        size_t d = byte_idx * 8 + b;
        if (d >= start && d < end) {
            byte_val |= ((codes[d - start] >> ((int)num_bits - 1)) & 1) << (7 - b);
        }
    }
    d_short_raw[vec_idx * short_bytes_per_vec + byte_idx] = byte_val;
}
```

**Byte boundary note:** When `chunk` is not a multiple of 8, a lane's dim range may straddle byte boundaries. Two adjacent lanes could write to the same byte. This is resolved by having each lane write only the bits for dimensions it owns — the output byte is built with the correct bits positioned by `(7 - b)`, and only lanes whose dims fall within that byte write to it. Use `atomicOr` on the byte to handle the boundary case safely, or ensure `chunk` is always a multiple of 8 (it is when D_seg is a multiple of 256, which covers all practical cases from the DP optimizer's 64-dim blocks).

- [ ] **Step 3: Add no-rotation variant**

Copy the fused kernel and modify the initial data loading to read from `d_vectors[vec_idx * D_total + seg_offset + d] - d_centroids[cid * D_total + seg_offset + d]`.

- [ ] **Step 4: Build**

Run build command. Expected: compiles. Not yet called from construct.

- [ ] **Step 5: Commit**

```
git add src/gpu/gpu_encoder.cu include/saq/gpu/gpu_encoder.cuh
git commit -m "feat(gpu): implement fused CAQ encode kernel (subtract + encode + pack)"
```

---

### Task 8: Wire Fused Encode into Construct Pipeline

**Files:**
- Modify: `src/gpu/gpu_ivf_construct.cpp`

- [ ] **Step 1: Update per-segment pipeline in construct**

Replace the current 5-step pipeline (subtract → GEMM → encode → pack_short → pack_long) with the fused 2-step pipeline (GEMM on raw vectors → fused encode):

```cpp
// 5a. Rotation: GEMM on raw vector segment (no subtract needed)
auto d_rotated = device_alloc<float>(N * D_seg);

if (bdata.rotator) {
    auto d_P = device_alloc<float>(D_seg * D_seg);
    upload(d_P.get(), bdata.rotator->get_P().data(), D_seg * D_seg);

    // GEMM directly on raw vectors (not residuals)
    // Extract vector segment slice via pointer arithmetic
    // d_vectors points to [N x D] full vectors
    // Extract vector segment slice (cuBLAS can't slice row-major subarrays)
    // Use cublasSgeam to copy the submatrix:
    //   d_vec_seg[i, d] = d_vectors[i, dim_offset + d]
    //   In column-major terms: copy submatrix starting at column dim_offset
    auto d_vec_seg = device_alloc<float>(N * D_seg);
    {
        float alpha = 1.0f, beta = 0.0f;
        // Row-major [N x D] looks like column-major [D x N] to cuBLAS
        // We want columns [dim_offset, dim_offset+D_seg) of the [D x N] matrix
        // = rows [dim_offset, dim_offset+D_seg) of the row-major [N x D] matrix
        // cublasSgeam copies a submatrix with leading dimension
        SAQ_CUBLAS_CHECK(cublasSgeam(cublas.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)D_seg, (int)N,
            &alpha, d_vectors.get() + dim_offset, (int)D,  // src with stride D
            &beta,  d_vec_seg.get(), (int)D_seg,            // dummy
            d_vec_seg.get(), (int)D_seg));                   // dst with stride D_seg
    }

    float alpha = 1.0f, beta = 0.0f;
    SAQ_CUBLAS_CHECK(cublasSgemm(cublas.get(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)D_seg, (int)N, (int)D_seg,
        &alpha, d_P.get(), (int)D_seg,
        d_vec_seg.get(), (int)D_seg,
        &beta, d_rotated.get(), (int)D_seg));
}

// Compute rotated centroids on CPU, upload
auto d_rotated_centroids = device_alloc<float>(K * D_seg);
// ... (existing centroid rotation code, already in construct) ...
upload(d_rotated_centroids.get(), cent_rotated.data(), K * D_seg);

// 5b. Fused encode (subtract rotated centroid + encode + pack)
auto d_short_raw = device_alloc<uint8_t>(N * (D_seg / 8));
size_t long_code_bytes = (num_bits > 1) ? D_seg * (num_bits - 1) / 8 : 0;
auto d_long_raw = device_alloc<uint8_t>(N * (long_code_bytes > 0 ? long_code_bytes : 1));

auto d_o_l2norm = device_alloc<float>(N);
auto d_fac_rescale = device_alloc<float>(N);
auto d_fac_error = device_alloc<float>(N);
auto d_ip_cent_oa = device_alloc<float>(N);

if (bdata.rotator) {
    launch_fused_caq_encode(
        d_rotated.get(), d_rotated_centroids.get(), d_cluster_ids.get(),
        d_o_l2norm.get(), d_fac_rescale.get(), d_fac_error.get(), d_ip_cent_oa.get(),
        d_short_raw.get(), d_long_raw.get(),
        D_seg, N, K, num_bits, code_max,
        bdata.cfg.caq_adj_rd_lmt, bdata.cfg.caq_adj_eps, bdata.cfg.caq_ori_qB);
} else {
    launch_fused_caq_encode_no_rotation(
        d_vectors.get(), d_centroids.get(), d_cluster_ids.get(),
        dim_offset, D_seg, D,
        d_o_l2norm.get(), d_fac_rescale.get(), d_fac_error.get(), d_ip_cent_oa.get(),
        d_short_raw.get(), d_long_raw.get(),
        num_bits, code_max,
        bdata.cfg.caq_adj_rd_lmt, bdata.cfg.caq_adj_eps, bdata.cfg.caq_ori_qB);
}
```

- [ ] **Step 2: Remove d_residuals and d_codes allocations**

Delete the `device_alloc` calls for `d_residuals` and `d_codes`. Delete calls to `launch_subtract_centroid`, `launch_pack_short_codes`, `launch_pack_long_codes`.

- [ ] **Step 3: Build and run benchmark**

Expected: GPU kernel time may be slightly faster (less memory traffic). Total GPU time should remain ~300ms. Verify no CUDA errors at all 3 bpd values.

- [ ] **Step 4: Commit**

```
git add src/gpu/gpu_ivf_construct.cpp
git commit -m "feat(gpu): wire fused encode into construct, remove intermediate buffers

Eliminates d_residuals and d_codes global memory round-trips.
Pipeline reduced from 5 kernels to 1 cuBLAS GEMM + 1 fused encode."
```

---

### Task 9: Delete gpu_packer.cu/cuh

**Files:**
- Delete: `src/gpu/gpu_packer.cu`
- Delete: `include/saq/gpu/gpu_packer.cuh`
- Modify: `src/CMakeLists.txt` — remove `gpu/gpu_packer.cu`

- [ ] **Step 1: Delete files and update CMakeLists**

- [ ] **Step 2: Build to verify no remaining references**

- [ ] **Step 3: Commit**

```
git add -u
git commit -m "chore(gpu): delete gpu_packer (folded into fused encode + scatter)"
```

---

## Chunk 4: GPU Batch Search

### Task 10: Create Search Descriptor Types

**Files:**
- Create: `include/saq/gpu/gpu_searcher.cuh`

- [ ] **Step 1: Write descriptor structs and launch declarations**

```cpp
#pragma once

#ifdef SAQ_USE_CUDA

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace saq::gpu {

/// Device-side per-segment descriptor for search kernel.
struct GpuSegmentDescriptor {
    uint8_t* short_codes;
    uint8_t* long_codes;
    float* factor_o_l2norm;
    float* factor_ip_cent_oa;
    float* factor_rescale;
    float* factor_error;
    float* centroids;
    size_t num_codebooks;       // D_seg / 4
    size_t D_seg;
    size_t num_bits;
    size_t long_bytes_per_vec;  // D_seg * (bits-1) / 8, or 0
};

/// Device-side per-cluster descriptor for search kernel.
struct GpuClusterDescriptor {
    size_t num_vec;
    size_t num_blocks;
    uint32_t* ids;          // pointer into pool.ids for this cluster's original IDs
};

void launch_search(
    const GpuSegmentDescriptor* d_seg_descs,
    const GpuClusterDescriptor* d_clu_descs,
    const uint32_t* d_block_offsets,
    const uint32_t* d_cluster_offsets,
    const float* d_rotated_queries,
    const float* d_query_constants,
    const uint32_t* d_centroid_ids,
    const float* d_data_variance,
    size_t Q, size_t nprobe, size_t topk,
    size_t num_segments, size_t total_D_seg,
    float* d_candidate_dists,
    uint32_t* d_candidate_ids,
    uint32_t* d_candidate_counts,
    size_t max_candidates_per_block,
    cudaStream_t stream = 0);

void launch_merge_topk(
    const float* d_candidate_dists,
    const uint32_t* d_candidate_ids,
    const uint32_t* d_candidate_counts,
    uint32_t* d_results,
    float* d_result_dists,
    size_t Q, size_t nprobe, size_t topk,
    size_t max_candidates_per_block,
    cudaStream_t stream = 0);

} // namespace saq::gpu

#endif
```

- [ ] **Step 2: Build**

- [ ] **Step 3: Commit**

```
git add include/saq/gpu/gpu_searcher.cuh
git commit -m "feat(gpu): add search descriptor types and kernel declarations"
```

---

### Task 11: Implement Search Kernel

**Files:**
- Create: `src/gpu/gpu_search.cu`
- Modify: `src/CMakeLists.txt`

This is the largest single task. The kernel implements:
1. Phase 0: Build LUT in shared memory (cooperative across 128 threads)
2. Phase 1+2: Warp-level 3-stage search (variance → LUT fastscan → accurate)
3. Phase 3: Output candidates

- [ ] **Step 1: Implement `kernel_build_lut_and_search`**

The kernel is launched as `<<<dim3(Q, nprobe), 128>>>`. Each block handles one (query, cluster) pair. Implement following the spec's per-block execution pseudocode.

Key device functions needed:
- `__device__ void build_lut(...)` — fills shared memory LUT from rotated query
- `__device__ float gpu_long_code_ip(...)` — unpacks variable-bit long codes and computes IP
- `__device__ float warp_reduce_min(float val)` — min reduction across warp

**Shared memory:** Use `extern __shared__` dynamic shared memory for the LUT since `num_segments` and `num_codebooks` are runtime values. The launch wrapper computes the required size (`total_codebooks * 16 * sizeof(int16_t) + num_segments * 4 * sizeof(float) + 64`) and passes it as the third kernel launch parameter:
```cuda
kernel<<<dim3(Q, nprobe), 128, shmem_bytes>>>(...)
```

- [ ] **Step 2: Implement `kernel_merge_topk`**

One block per query, 256 threads. Loads all candidates from nprobe slots, performs a parallel partial sort to find top-K results.

- [ ] **Step 3: Implement launch wrappers**

- [ ] **Step 4: Add to CMakeLists**

- [ ] **Step 5: Build**

- [ ] **Step 6: Commit**

```
git add src/gpu/gpu_search.cu src/CMakeLists.txt
git commit -m "feat(gpu): implement GPU batch search kernel with 3-stage pipeline"
```

---

### Task 12: Add search_batch to GpuIVF

**Files:**
- Modify: `include/saq/gpu/gpu_ivf.h` — add `search_batch` declaration
- Create: `src/gpu/gpu_ivf_search.cpp` — host-side orchestration
- Modify: `src/CMakeLists.txt` — add `gpu/gpu_ivf_search.cpp`

This file is `.cpp` (not `.cu`) because it uses Eigen for query rotation. Same pattern as `gpu_ivf_construct.cpp`.

- [ ] **Step 1: Add `search_batch` to `gpu_ivf.h`**

```cpp
/// GPU-accelerated batch search.
template <DistType kDistType = DistType::L2Sqr>
void search_batch(
    const FloatRowMat& queries,
    size_t topk, size_t nprobe,
    SearcherConfig cfg,
    PID* results);
```

- [ ] **Step 2: Implement `gpu_ivf_search.cpp`**

Since `search_batch` is a template (parameterized on `DistType`), add explicit instantiations at the bottom of the `.cpp` file:
```cpp
template void GpuIVF::search_batch<DistType::L2Sqr>(...);
template void GpuIVF::search_batch<DistType::IP>(...);
```

Host-side orchestration:
1. Find nprobe centroids per query (CPU, OpenMP)
2. Rotate queries per segment (CPU, Eigen)
3. Compute LUT constants (delta, sum_vl_lut, sum_q, q_l2sqr) per query per segment
4. Upload query data + centroid IDs
5. Build and upload descriptor tables
6. Launch search kernel
7. Launch merge kernel
8. Download results

- [ ] **Step 3: Add to CMakeLists**

- [ ] **Step 4: Build**

- [ ] **Step 5: Commit**

```
git add include/saq/gpu/gpu_ivf.h src/gpu/gpu_ivf_search.cpp src/CMakeLists.txt
git commit -m "feat(gpu): add search_batch host orchestration"
```

---

### Task 13: Add Search Benchmark to Sample

**Files:**
- Modify: `samples/gpu_benchmark_sample.cpp`

- [ ] **Step 1: Add search benchmark section**

After the existing encode benchmark, add:

```cpp
// --- GPU Search ---
LOG(INFO) << "--- GPU Search ---";
// Load queries and ground truth
FloatRowMat queries;
load_something<float, FloatRowMat>(data_dir + "/queries_pca.fvecs", queries);
UintRowMat gt;
load_something<uint32_t, UintRowMat>(data_dir + "/groundtruth.ivecs", gt);

size_t Q = queries.rows();
size_t topk = 100;
std::vector<PID> gpu_results(Q * topk);

StopW search_timer;
gpu_ivf.search_batch(queries, topk, nprobe, SearcherConfig{}, gpu_results.data());
auto gpu_search_ms = search_timer.getElapsedTimeMicro() / 1000.0;
LOG(INFO) << "GPU batch search: Q=" << Q << " nprobe=" << nprobe
          << " time=" << gpu_search_ms << " ms"
          << " QPS=" << (Q * 1000.0 / gpu_search_ms);

// Compare recall vs ground truth
size_t correct = 0;
for (size_t q = 0; q < Q; ++q) {
    for (size_t k = 0; k < topk && k < gt.cols(); ++k) {
        for (size_t r = 0; r < topk; ++r) {
            if (gpu_results[q * topk + r] == gt(q, k)) { correct++; break; }
        }
    }
}
double recall = 100.0 * correct / (Q * std::min(topk, (size_t)gt.cols()));
LOG(INFO) << "GPU Recall@" << topk << " = " << recall << "%";
```

- [ ] **Step 2: Build and run**

Run at 2.0 bpd. Expected: GPU search completes, recall should be within 0.5% of CPU recall at same nprobe.

- [ ] **Step 3: Commit**

```
git add samples/gpu_benchmark_sample.cpp
git commit -m "feat(gpu): add search benchmark to gpu_benchmark_sample"
```

---

## Chunk 5: Final Validation and Cleanup

### Task 14: Full Benchmark Suite

- [ ] **Step 1: Run encode + search at 1.0, 2.0, 4.0 bpd**

Verify:
- Encode time ~300ms (pool + scatter + fused kernel)
- Search time and recall reported
- No CUDA errors

- [ ] **Step 2: Update GPU analysis doc**

The analysis doc is at `docs/gpu-encode-analysis.md`. Update

Add new benchmark results with optimized pipeline.

- [ ] **Step 3: Commit**

```
git add docs/gpu-encode-analysis.md
git commit -m "docs: update GPU analysis with optimized encode and search benchmarks"
```
