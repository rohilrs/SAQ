# GPU Encode Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Accelerate SAQ encode (IVF::construct) using CUDA, keeping codebook and encoded data on GPU.

**Architecture:** Full GPU encode pipeline — upload vectors/centroids/rotation matrices once, run all encode steps (rotation, CAQ quantization, code adjustment, packing) as CUDA kernels, store results in GPU-resident `GpuSaqCluData` structures. Warp-cooperative encoding (32 threads per vector) for efficient register usage. cuBLAS for rotation GEMM.

**Tech Stack:** CUDA 12+, cuBLAS, Thrust, C++17 device code

---

### Task 1: CMake CUDA Infrastructure

**Files:**
- Modify: `CMakeLists.txt:12-16` (add option)
- Modify: `CMakeLists.txt:86-93` (add CUDA block)
- Modify: `src/CMakeLists.txt` (conditional CUDA sources)

**Step 1: Add SAQ_BUILD_CUDA option to root CMakeLists.txt**

After the existing options block (line 16), add:

```cmake
option(SAQ_BUILD_CUDA "Build CUDA GPU acceleration" OFF)
```

**Step 2: Add CUDA configuration block**

After the OpenMP block (after line 107), add:

```cmake
if(SAQ_BUILD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # A100=80, RTX3090=86, RTX4090=89, H100=90, RTX5090=100
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100" CACHE STRING "CUDA architectures")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(saq PUBLIC CUDA::cublas CUDA::cudart)
  target_compile_definitions(saq PUBLIC SAQ_USE_CUDA=1)
  message(STATUS "CUDA enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
endif()
```

**Step 3: Add conditional CUDA sources to src/CMakeLists.txt**

Append to `src/CMakeLists.txt`:

```cmake
if(SAQ_BUILD_CUDA)
  target_sources(saq
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_utils.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_encoder.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_packer.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_ivf_construct.cu
  )
  target_include_directories(saq PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_encoder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_packer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_ivf_construct.cu
    PROPERTIES LANGUAGE CUDA
  )
endif()
```

**Step 4: Create directory structure**

```bash
mkdir -p include/saq/gpu src/gpu
```

**Step 5: Verify CMake configure succeeds**

```bash
cmake -B build -DSAQ_BUILD_CUDA=ON -DSAQ_BUILD_SAMPLES=ON
```

Expected: CMake configures successfully, finds CUDA toolkit.

**Step 6: Commit**

```bash
git add CMakeLists.txt src/CMakeLists.txt
git commit -m "build: add SAQ_BUILD_CUDA cmake option with cuBLAS linkage"
```

---

### Task 2: GPU Utilities — Error Checking and Memory Helpers

**Files:**
- Create: `include/saq/gpu/gpu_utils.cuh`
- Create: `src/gpu/gpu_utils.cu`

**Step 1: Create gpu_utils.cuh with error macros and memory helpers**

Create `include/saq/gpu/gpu_utils.cuh`:

```cpp
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace saq::gpu {

// Error checking macros
#define SAQ_CUDA_CHECK(call)                                                      \
    do {                                                                           \
        cudaError_t err = (call);                                                  \
        if (err != cudaSuccess) {                                                  \
            throw std::runtime_error(                                              \
                std::string("CUDA error at ") + __FILE__ + ":" +                   \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));         \
        }                                                                          \
    } while (0)

#define SAQ_CUBLAS_CHECK(call)                                                     \
    do {                                                                           \
        cublasStatus_t status = (call);                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                     \
            throw std::runtime_error(                                              \
                std::string("cuBLAS error at ") + __FILE__ + ":" +                 \
                std::to_string(__LINE__) + ": status=" + std::to_string(status));   \
        }                                                                          \
    } while (0)

/// RAII wrapper for device memory.
template <typename T>
struct DeviceDeleter {
    void operator()(T* p) const {
        if (p) cudaFree(p);
    }
};

template <typename T>
using DevicePtr = std::unique_ptr<T[], DeviceDeleter<T>>;

/// Allocate device memory and return a managed pointer.
template <typename T>
DevicePtr<T> device_alloc(size_t count) {
    T* p = nullptr;
    SAQ_CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    SAQ_CUDA_CHECK(cudaMemset(p, 0, count * sizeof(T)));
    return DevicePtr<T>(p);
}

/// Upload host data to device.
template <typename T>
void upload(T* d_dst, const T* h_src, size_t count) {
    SAQ_CUDA_CHECK(cudaMemcpy(d_dst, h_src, count * sizeof(T), cudaMemcpyHostToDevice));
}

/// Download device data to host.
template <typename T>
void download(T* h_dst, const T* d_src, size_t count) {
    SAQ_CUDA_CHECK(cudaMemcpy(h_dst, d_src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

/// RAII cuBLAS handle.
class CublasHandle {
    cublasHandle_t handle_;
public:
    CublasHandle() { SAQ_CUBLAS_CHECK(cublasCreate(&handle_)); }
    ~CublasHandle() { cublasDestroy(handle_); }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    cublasHandle_t get() const { return handle_; }
};

/// Warp-level reduce: sum across all 32 lanes.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

/// Warp-level reduce: max across all 32 lanes.
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

/// Warp-level reduce: sum of ints across all 32 lanes.
__device__ __forceinline__ int warp_reduce_sum_int(int val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

/// Broadcast value from lane 0 to all lanes.
__device__ __forceinline__ float warp_broadcast(float val) {
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

__device__ __forceinline__ double warp_broadcast_double(double val) {
    // __shfl_sync doesn't support double; transfer as two 32-bit ints
    int lo = __shfl_sync(0xFFFFFFFF, __double2loint(val), 0);
    int hi = __shfl_sync(0xFFFFFFFF, __double2hiint(val), 0);
    return __hiloint2double(hi, lo);
}

__device__ __forceinline__ double warp_reduce_sum_double(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int lo = __shfl_down_sync(0xFFFFFFFF, __double2loint(val), offset);
        int hi = __shfl_down_sync(0xFFFFFFFF, __double2hiint(val), offset);
        val += __hiloint2double(hi, lo);
    }
    return val;
}

__device__ __forceinline__ int warp_broadcast_int(int val) {
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

} // namespace saq::gpu
```

**Step 2: Create minimal gpu_utils.cu**

Create `src/gpu/gpu_utils.cu`:

```cpp
#include "saq/gpu/gpu_utils.cuh"

// Translation unit for gpu_utils — keeps linker happy.
// Device functions are __forceinline__ in the header.
```

**Step 3: Build to verify compilation**

```bash
cmake -B build -DSAQ_BUILD_CUDA=ON -DSAQ_BUILD_SAMPLES=ON && cmake --build build --target saq
```

Expected: compiles without errors.

**Step 4: Commit**

```bash
git add include/saq/gpu/gpu_utils.cuh src/gpu/gpu_utils.cu
git commit -m "feat(gpu): add CUDA error checking and warp reduction utilities"
```

---

### Task 3: GPU Cluster Data Structure

**Files:**
- Create: `include/saq/gpu/gpu_cluster_data.cuh`

**Step 1: Create GpuSaqCluData — GPU-resident cluster storage**

Create `include/saq/gpu/gpu_cluster_data.cuh`:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "saq/gpu/gpu_utils.cuh"
#include "saq/cluster_data.h"
#include "saq/defines.h"

namespace saq::gpu {

/// GPU-resident per-segment cluster data (mirrors CAQClusterData layout).
struct GpuSegmentData {
    size_t num_dim_pad;
    size_t num_bits;
    size_t num_blocks;

    // Device pointers — owned by GpuSaqCluData
    float* d_centroid;            // [num_dim_pad]
    float* d_factor_o_l2norm;     // [num_blocks * KFastScanSize]
    float* d_factor_ip_cent_oa;   // [num_blocks * KFastScanSize]
    uint8_t* d_short_codes;       // fastscan-packed layout
    uint8_t* d_long_codes;        // compacted per-vector
    float* d_long_factor_rescale; // [num_vec]
    float* d_long_factor_error;   // [num_vec]
};

/// GPU-resident cluster data (mirrors SaqCluData).
struct GpuSaqCluData {
    size_t num_vec = 0;
    size_t num_segments = 0;
    size_t num_blocks = 0;

    DevicePtr<uint32_t> d_ids;
    std::vector<GpuSegmentData> segments;

    // Owning device memory — segments point into these
    std::vector<DevicePtr<float>> owned_centroids;
    std::vector<DevicePtr<float>> owned_factor_o_l2norm;
    std::vector<DevicePtr<float>> owned_factor_ip_cent_oa;
    std::vector<DevicePtr<uint8_t>> owned_short_codes;
    std::vector<DevicePtr<uint8_t>> owned_long_codes;
    std::vector<DevicePtr<float>> owned_long_factor_rescale;
    std::vector<DevicePtr<float>> owned_long_factor_error;

    /// Allocate GPU memory matching a quantization plan.
    void allocate(size_t n_vec,
                  const std::vector<std::pair<size_t, size_t>>& quant_plan) {
        num_vec = n_vec;
        num_segments = quant_plan.size();
        num_blocks = (num_vec + KFastScanSize - 1) / KFastScanSize;

        d_ids = device_alloc<uint32_t>(num_vec);
        segments.resize(num_segments);

        owned_centroids.resize(num_segments);
        owned_factor_o_l2norm.resize(num_segments);
        owned_factor_ip_cent_oa.resize(num_segments);
        owned_short_codes.resize(num_segments);
        owned_long_codes.resize(num_segments);
        owned_long_factor_rescale.resize(num_segments);
        owned_long_factor_error.resize(num_segments);

        for (size_t s = 0; s < num_segments; ++s) {
            auto dim_pad = quant_plan[s].first;
            auto bits = quant_plan[s].second;
            auto& seg = segments[s];

            seg.num_dim_pad = dim_pad;
            seg.num_bits = bits;
            seg.num_blocks = num_blocks;

            owned_centroids[s] = device_alloc<float>(dim_pad);
            seg.d_centroid = owned_centroids[s].get();

            owned_factor_o_l2norm[s] = device_alloc<float>(num_blocks * KFastScanSize);
            seg.d_factor_o_l2norm = owned_factor_o_l2norm[s].get();

            owned_factor_ip_cent_oa[s] = device_alloc<float>(num_blocks * KFastScanSize);
            seg.d_factor_ip_cent_oa = owned_factor_ip_cent_oa[s].get();

            size_t short_code_bytes = bits ? dim_pad * KFastScanSize / 8 * num_blocks : 0;
            owned_short_codes[s] = device_alloc<uint8_t>(short_code_bytes > 0 ? short_code_bytes : 1);
            seg.d_short_codes = short_code_bytes > 0 ? owned_short_codes[s].get() : nullptr;

            size_t long_code_bytes_per_vec = bits > 1 ? dim_pad * (bits - 1) / 8 : 0;
            size_t total_long = long_code_bytes_per_vec * num_vec;
            owned_long_codes[s] = device_alloc<uint8_t>(total_long > 0 ? total_long : 1);
            seg.d_long_codes = total_long > 0 ? owned_long_codes[s].get() : nullptr;

            owned_long_factor_rescale[s] = device_alloc<float>(num_vec);
            seg.d_long_factor_rescale = owned_long_factor_rescale[s].get();

            owned_long_factor_error[s] = device_alloc<float>(num_vec);
            seg.d_long_factor_error = owned_long_factor_error[s].get();
        }
    }

    /// Download GPU cluster data to a CPU SaqCluData.
    /// Needed for CPU-side search or save/load.
    void download_to_cpu(SaqCluData& cpu_clu) const;
};

} // namespace saq::gpu
```

**Step 2: Build to verify**

```bash
cmake --build build --target saq
```

**Step 3: Commit**

```bash
git add include/saq/gpu/gpu_cluster_data.cuh
git commit -m "feat(gpu): add GpuSaqCluData GPU-resident cluster data structure"
```

---

### Task 4: Centroid Subtraction Kernel

**Files:**
- Create: `include/saq/gpu/gpu_encoder.cuh`
- Create: `src/gpu/gpu_encoder.cu`

**Step 1: Write gpu_subtract_centroid kernel**

Create `include/saq/gpu/gpu_encoder.cuh`:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace saq::gpu {

/// Subtract each vector's cluster centroid from its segment slice.
/// d_vectors: [N x D_total] row-major input vectors
/// d_centroids: [K x D_total] row-major centroids
/// d_cluster_ids: [N] cluster assignment per vector (sorted order)
/// d_residuals: [N x D_seg] output residuals
/// seg_offset: dimension offset for this segment
/// D_seg: padded segment dimension
/// D_total: total padded dimension
/// N: number of vectors
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
/// d_rotated: [N x D_seg] rotated residual vectors
/// d_codes: [N x D_seg] output integer codes (as int32)
/// d_o_l2norm: [N] output L2 norms
/// d_fac_rescale: [N] output rescale factors
/// d_fac_error: [N] output error factors
/// d_ip_cent_oa: [N] output centroid-oa inner products (0 if num_bits <= 1)
/// d_centroids_seg: [K x D_seg] per-segment rotated centroids
/// d_cluster_ids: [N] cluster assignment per vector
/// D_seg: padded segment dimension
/// N: number of vectors
/// K: number of clusters
/// num_bits: quantization bits for this segment
/// code_max: (1 << num_bits) - 1 or (1 << caq_ori_qB) - 1
/// caq_adj_rd_lmt: max code adjustment rounds (0 = disabled)
/// caq_adj_eps: code adjustment epsilon
/// caq_ori_qB: original quantization bits (0 = disabled)
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
```

**Step 2: Implement subtract_centroid kernel**

Create `src/gpu/gpu_encoder.cu`:

```cpp
#include "saq/gpu/gpu_encoder.cuh"
#include "saq/gpu/gpu_utils.cuh"

namespace saq::gpu {

__global__ void kernel_subtract_centroid(
    const float* __restrict__ d_vectors,
    const float* __restrict__ d_centroids,
    const uint32_t* __restrict__ d_cluster_ids,
    float* __restrict__ d_residuals,
    size_t seg_offset,
    size_t D_seg,
    size_t D_total,
    size_t N) {

    size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= N) return;

    uint32_t cid = d_cluster_ids[vec_idx];
    const float* vec = d_vectors + vec_idx * D_total + seg_offset;
    const float* cen = d_centroids + cid * D_total + seg_offset;
    float* out = d_residuals + vec_idx * D_seg;

    for (size_t d = 0; d < D_seg; ++d) {
        out[d] = vec[d] - cen[d];
    }
}

void launch_subtract_centroid(
    const float* d_vectors,
    const float* d_centroids,
    const uint32_t* d_cluster_ids,
    float* d_residuals,
    size_t seg_offset,
    size_t D_seg,
    size_t D_total,
    size_t N,
    cudaStream_t stream) {

    constexpr int kBlockSize = 256;
    int grid = (N + kBlockSize - 1) / kBlockSize;
    kernel_subtract_centroid<<<grid, kBlockSize, 0, stream>>>(
        d_vectors, d_centroids, d_cluster_ids, d_residuals,
        seg_offset, D_seg, D_total, N);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Warp-cooperative CAQ Encode
// ============================================================================

// Each warp (32 threads) processes one vector cooperatively.
// Lane i handles dimensions [i*chunk, (i+1)*chunk) where chunk = ceil(D_seg/32).

__global__ void kernel_caq_encode(
    const float* __restrict__ d_rotated,  // [N x D_seg]
    int* __restrict__ d_codes,            // [N x D_seg]
    float* __restrict__ d_o_l2norm,       // [N]
    float* __restrict__ d_fac_rescale,    // [N]
    float* __restrict__ d_fac_error,      // [N]
    float* __restrict__ d_ip_cent_oa,     // [N]
    const float* __restrict__ d_centroids_seg, // [K x D_seg]
    const uint32_t* __restrict__ d_cluster_ids, // [N]
    size_t D_seg,
    size_t N,
    size_t K,
    size_t num_bits,
    uint16_t code_max,
    int caq_adj_rd_lmt,
    float caq_adj_eps,
    int caq_ori_qB) {

    // One warp per vector
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if ((size_t)warp_id >= N) return;
    if (num_bits == 0) {
        // Zero-bit segment: just compute o_l2norm
        const float* vec = d_rotated + (size_t)warp_id * D_seg;
        size_t chunk = (D_seg + 31) / 32;
        size_t start = lane_id * chunk;
        size_t end = min(start + chunk, D_seg);
        double partial_l2 = 0.0;
        for (size_t d = start; d < end; ++d)
            partial_l2 += (double)vec[d] * vec[d];
        double total_l2 = warp_reduce_sum_double(partial_l2);
        if (lane_id == 0) {
            d_o_l2norm[warp_id] = sqrtf((float)total_l2);
            d_fac_rescale[warp_id] = 0.0f;
            d_fac_error[warp_id] = 0.0f;
            d_ip_cent_oa[warp_id] = 0.0f;
        }
        return;
    }

    const float* vec = d_rotated + (size_t)warp_id * D_seg;
    int* codes = d_codes + (size_t)warp_id * D_seg;

    // Each lane handles a chunk of dimensions
    size_t chunk = (D_seg + 31) / 32;
    size_t start = lane_id * chunk;
    size_t end = min(start + chunk, D_seg);

    // ---- Step 1: Compute v_max ----
    float local_max = 0.0f;
    for (size_t d = start; d < end; ++d) {
        float v = vec[d];
        local_max = fmaxf(local_max, fabsf(v));
    }
    float v_mx = warp_reduce_max(local_max);
    v_mx = warp_broadcast(v_mx);
    float v_mi = -v_mx;

    float delta = (v_mx - v_mi) / (code_max + 1);

    // ---- Step 2: Initial quantization ----
    double partial_ip_o_code = 0.0;
    uint64_t partial_code_l2sqr = 0;
    int partial_code_sum = 0;
    double partial_vec_sum = 0.0;
    double partial_o_l2sqr = 0.0;

    for (size_t d = start; d < end; ++d) {
        float o = vec[d];
        partial_o_l2sqr += (double)o * o;
        partial_vec_sum += o;

        int c;
        if (delta > 0.0f) {
            c = (int)floorf((o - v_mi) / delta);
            c = min(c, (int)code_max);
            c = max(c, 0);
        } else {
            c = 0;
        }
        codes[d] = c;
        partial_ip_o_code += (double)c * o;
        partial_code_l2sqr += (uint64_t)c * c;
        partial_code_sum += c;
    }

    // Warp reduce to get global sums
    double ip_o_code = warp_reduce_sum_double(partial_ip_o_code);
    double vec_sum = warp_reduce_sum_double(partial_vec_sum);
    double o_l2sqr = warp_reduce_sum_double(partial_o_l2sqr);

    // For code_l2sqr and code_sum, reduce as doubles to avoid overflow
    double code_l2sqr_d = warp_reduce_sum_double((double)partial_code_l2sqr);
    double code_sum_d = warp_reduce_sum_double((double)partial_code_sum);

    // Broadcast to all lanes
    ip_o_code = warp_broadcast_double(ip_o_code);
    vec_sum = warp_broadcast_double(vec_sum);
    o_l2sqr = warp_broadcast_double(o_l2sqr);
    code_l2sqr_d = warp_broadcast_double(code_l2sqr_d);
    code_sum_d = warp_broadcast_double(code_sum_d);

    double ip_o_oa = ip_o_code * delta + (v_mi + 0.5 * delta) * vec_sum;
    double oa_l2sqr = delta * delta * code_l2sqr_d
                    + (delta * delta + 2.0 * delta * v_mi) * code_sum_d
                    + (0.25 * delta * delta + delta * v_mi + (double)v_mi * v_mi) * D_seg;

    // ---- Step 3: Code adjustment ----
    if (caq_adj_rd_lmt && oa_l2sqr > 0.0 && delta > 0.0f) {
        double re_eps = (double)caq_adj_eps * oa_l2sqr;

        for (int round = 1; round <= caq_adj_rd_lmt || caq_adj_rd_lmt == 0; ++round) {
            int local_adj_cnt = 0;

            for (size_t d = start; d < end; ++d) {
                float o = vec[d];
                int c = codes[d];
                double oa = (c + 0.5) * delta + v_mi;
                double oa_l2sqr_tmp = oa_l2sqr - oa * oa;
                double ip_delta = delta * o;

                // Try ++
                while (c < (int)code_max) {
                    double new_q = oa + delta;
                    double new_length = oa_l2sqr_tmp + new_q * new_q;
                    double new_ip = ip_o_oa + ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c++;
                    ip_o_oa = new_ip;
                    oa = new_q;
                    oa_l2sqr = new_length;
                    local_adj_cnt++;
                }
                // Try --
                while (c > 0) {
                    double new_q = oa - delta;
                    double new_length = oa_l2sqr_tmp + new_q * new_q;
                    double new_ip = ip_o_oa - ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c--;
                    ip_o_oa = new_ip;
                    oa = new_q;
                    oa_l2sqr = new_length;
                    local_adj_cnt++;
                }
                codes[d] = c;
            }

            // Check convergence across warp
            int total_adj = warp_reduce_sum_int(local_adj_cnt);
            total_adj = warp_broadcast_int(total_adj);
            if (total_adj == 0) break;

            // Correction pass: recompute sums from codes
            double corr_oa_l2 = 0.0, corr_ip = 0.0;
            for (size_t d = start; d < end; ++d) {
                float o = vec[d];
                double q = (codes[d] + 0.5) * delta + v_mi;
                corr_ip += q * o;
                corr_oa_l2 += q * q;
            }
            oa_l2sqr = warp_reduce_sum_double(corr_oa_l2);
            ip_o_oa = warp_reduce_sum_double(corr_ip);
            oa_l2sqr = warp_broadcast_double(oa_l2sqr);
            ip_o_oa = warp_broadcast_double(ip_o_oa);
            re_eps = (double)caq_adj_eps * oa_l2sqr;
        }
    }

    // ---- Step 3b: DownUpSample (if caq_ori_qB > 0) ----
    if (caq_ori_qB > 0) {
        int sampled_rshift = caq_ori_qB - (int)num_bits;
        delta *= (float)(1 << sampled_rshift);
        // Reset and recompute
        double new_ip = 0.0, new_oa_l2 = 0.0;
        for (size_t d = start; d < end; ++d) {
            codes[d] >>= sampled_rshift;
            float o = vec[d];
            double q = (codes[d] + 0.5) * delta + v_mi;
            new_ip += q * o;
            new_oa_l2 += q * q;
        }
        ip_o_oa = warp_reduce_sum_double(new_ip);
        oa_l2sqr = warp_reduce_sum_double(new_oa_l2);
        ip_o_oa = warp_broadcast_double(ip_o_oa);
        oa_l2sqr = warp_broadcast_double(oa_l2sqr);
    }

    // ---- Step 4: Compute factors ----
    // rescale_vmx_to1: scale so v_mx = 1
    double scale_rate = (v_mx > 0.0f) ? 1.0 / v_mx : 0.0;
    double scaled_ip = ip_o_oa * scale_rate;
    double scaled_oa_l2 = oa_l2sqr * scale_rate * scale_rate;

    float fac_rescale = (ip_o_oa != 0.0) ? (float)(o_l2sqr / ip_o_oa) : 0.0f;
    float o_l2norm = sqrtf((float)o_l2sqr);

    constexpr float kConstEpsilon = 1.9f;
    float fac_error = 0.0f;
    if (ip_o_oa > 0.0 && D_seg > 1) {
        fac_error = (float)(o_l2sqr * kConstEpsilon *
            sqrt(((o_l2sqr * oa_l2sqr) / (ip_o_oa * ip_o_oa) - 1.0) / (D_seg - 1)));
    }

    // ---- Step 5: ip_cent_oa (if num_bits > 1) ----
    float ip_c_oa = 0.0f;
    if (num_bits > 1) {
        uint32_t cid = d_cluster_ids[warp_id];
        const float* cent = d_centroids_seg + (size_t)cid * D_seg;
        // After rescale_vmx_to1: oa = (code + 0.5) * (delta * scale_rate) + (v_mi * scale_rate)
        double scaled_delta = delta * scale_rate;
        double scaled_vmi = v_mi * scale_rate;
        double partial_ip = 0.0;
        for (size_t d = start; d < end; ++d) {
            double oa_d = (codes[d] + 0.5) * scaled_delta + scaled_vmi;
            partial_ip += cent[d] * oa_d;
        }
        double total_ip = warp_reduce_sum_double(partial_ip);
        ip_c_oa = (float)warp_broadcast_double(total_ip);
    }

    // ---- Write outputs (lane 0 only for scalars) ----
    if (lane_id == 0) {
        d_o_l2norm[warp_id] = o_l2norm;
        d_fac_rescale[warp_id] = fac_rescale;
        d_fac_error[warp_id] = fac_error;
        d_ip_cent_oa[warp_id] = ip_c_oa;
    }
    // All lanes write their chunk of codes (already written in-place above)
}

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
    cudaStream_t stream) {

    constexpr int kWarpsPerBlock = 4;
    constexpr int kBlockSize = 32 * kWarpsPerBlock;
    int grid = ((int)N + kWarpsPerBlock - 1) / kWarpsPerBlock;

    kernel_caq_encode<<<grid, kBlockSize, 0, stream>>>(
        d_rotated, d_codes, d_o_l2norm, d_fac_rescale, d_fac_error, d_ip_cent_oa,
        d_centroids_seg, d_cluster_ids,
        D_seg, N, K, num_bits, code_max, caq_adj_rd_lmt, caq_adj_eps, caq_ori_qB);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace saq::gpu
```

**Step 3: Build to verify**

```bash
cmake --build build --target saq
```

**Step 4: Commit**

```bash
git add include/saq/gpu/gpu_encoder.cuh src/gpu/gpu_encoder.cu
git commit -m "feat(gpu): add centroid subtraction and warp-cooperative CAQ encode kernels"
```

---

### Task 5: GPU Code Packing Kernels

**Files:**
- Create: `include/saq/gpu/gpu_packer.cuh`
- Create: `src/gpu/gpu_packer.cu`

**Step 1: Create gpu_packer.cuh declarations**

Create `include/saq/gpu/gpu_packer.cuh`:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace saq::gpu {

/// Pack short codes: extract MSB of each code value, pack 8 per byte.
/// d_codes: [N x D_seg] integer codes
/// d_short_codes: [N x (D_seg/8)] output packed bytes
/// D_seg: padded segment dimension
/// N: number of vectors
/// num_bits: quantization bits
void launch_pack_short_codes(
    const int* d_codes,
    uint8_t* d_short_codes,
    size_t D_seg,
    size_t N,
    size_t num_bits,
    cudaStream_t stream = 0);

/// Pack long codes: mask off short bit, compact remaining bits.
/// d_codes: [N x D_seg] integer codes
/// d_long_codes: [N x long_code_bytes] output compacted codes
/// D_seg: padded segment dimension
/// N: number of vectors
/// num_bits: quantization bits (long code uses num_bits - 1 lower bits)
void launch_pack_long_codes(
    const int* d_codes,
    uint8_t* d_long_codes,
    size_t D_seg,
    size_t N,
    size_t num_bits,
    cudaStream_t stream = 0);

/// Reorder short codes into fastscan layout: 32 vectors interleaved per block.
/// d_short_codes_raw: [N x code_bytes] per-vector short codes
/// d_short_codes_packed: output in fastscan block layout
/// d_cluster_ids: [N] sorted cluster assignments
/// d_cluster_offsets: [K+1] prefix sum of cluster sizes
/// code_bytes: D_seg / 8 bytes per vector
/// N: total vectors
/// K: number of clusters
void launch_fastscan_reorder(
    const uint8_t* d_short_codes_raw,
    uint8_t* d_short_codes_packed,
    const uint32_t* d_cluster_ids,
    const size_t* d_cluster_offsets,
    size_t code_bytes,
    size_t N,
    size_t K,
    cudaStream_t stream = 0);

/// Copy per-vector factors into blocked cluster layout.
/// Rearranges linear [N] arrays into per-cluster blocks of KFastScanSize.
void launch_store_factors(
    const float* d_o_l2norm_flat,
    const float* d_ip_cent_oa_flat,
    const float* d_fac_rescale_flat,
    const float* d_fac_error_flat,
    float* d_factor_o_l2norm_blocked,
    float* d_factor_ip_cent_oa_blocked,
    float* d_long_factor_rescale,
    float* d_long_factor_error,
    const size_t* d_cluster_offsets,
    size_t cluster_idx,
    size_t num_vec_in_cluster,
    size_t global_offset,
    cudaStream_t stream = 0);

} // namespace saq::gpu
```

**Step 2: Implement packing kernels**

Create `src/gpu/gpu_packer.cu`:

```cpp
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
// Fastscan reorder: transpose per-vector codes into 32-vector blocks
// ============================================================================

// The fastscan layout interleaves 32 vectors' codes in a specific pattern
// matching the CPU fastscan::pack_codes layout used by the SIMD search.
// This kernel handles one block (32 vectors) per thread.

__global__ void kernel_fastscan_reorder(
    const uint8_t* __restrict__ d_short_codes_raw,
    uint8_t* __restrict__ d_short_codes_packed,
    size_t code_bytes_per_vec,
    size_t num_vecs_in_cluster,
    size_t num_blocks,
    size_t output_block_stride) {

    size_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;

    size_t blk_start = blk * 32;
    const uint8_t* src_base = d_short_codes_raw;
    uint8_t* dst = d_short_codes_packed + blk * output_block_stride;

    // For each column (byte position in the code)
    for (size_t col = 0; col < code_bytes_per_vec; ++col) {
        // Gather the col-th byte from each of the 32 vectors
        uint8_t vals[32];
        for (size_t v = 0; v < 32; ++v) {
            size_t global_v = blk_start + v;
            if (global_v < num_vecs_in_cluster) {
                vals[v] = src_base[global_v * code_bytes_per_vec + col];
            } else {
                vals[v] = 0;
            }
        }
        // Write in fastscan interleaved order: groups of 2, permuted
        // Follow the same pattern as fastscan::pack_codes
        for (size_t v = 0; v < 32; ++v) {
            dst[col * 32 + v] = vals[v];
        }
    }
}

void launch_fastscan_reorder(
    const uint8_t* d_short_codes_raw,
    uint8_t* d_short_codes_packed,
    const uint32_t* d_cluster_ids,
    const size_t* d_cluster_offsets,
    size_t code_bytes,
    size_t N,
    size_t K,
    cudaStream_t stream) {

    // Note: this is a simplified version. The actual fastscan layout
    // follows the specific permutation pattern from fastscan::pack_codes.
    // This will be refined in Task 7 to match exactly.
    (void)d_cluster_ids;
    (void)d_cluster_offsets;
    (void)N;
    (void)K;
    (void)d_short_codes_raw;
    (void)d_short_codes_packed;
    (void)code_bytes;
    (void)stream;
    // TODO: implement per-cluster fastscan reorder in Task 7
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
    const size_t* d_cluster_offsets,
    size_t cluster_idx,
    size_t num_vec_in_cluster,
    size_t global_offset,
    cudaStream_t stream) {

    if (num_vec_in_cluster == 0) return;
    (void)d_cluster_offsets;
    (void)cluster_idx;

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
```

**Step 3: Build to verify**

```bash
cmake --build build --target saq
```

**Step 4: Commit**

```bash
git add include/saq/gpu/gpu_packer.cuh src/gpu/gpu_packer.cu
git commit -m "feat(gpu): add short/long code packing and factor storage kernels"
```

---

### Task 6: GpuIVF Orchestrator — construct_gpu

**Files:**
- Create: `include/saq/gpu/gpu_ivf.h`
- Create: `src/gpu/gpu_ivf_construct.cu`

**Step 1: Create GpuIVF class declaration**

Create `include/saq/gpu/gpu_ivf.h`:

```cpp
#pragma once

#ifdef SAQ_USE_CUDA

#include <memory>
#include <vector>

#include "saq/defines.h"
#include "saq/config.h"
#include "saq/quantization_plan.h"
#include "saq/initializer.h"
#include "saq/gpu/gpu_cluster_data.cuh"

namespace saq::gpu {

/// GPU-accelerated IVF index.
/// Performs encode on GPU, stores encoded data in GPU memory.
/// Currently delegates search to CPU (downloads cluster data as needed).
class GpuIVF {
    size_t num_data_ = 0;
    size_t num_dim_ = 0;
    size_t num_cen_ = 0;
    QuantizeConfig cfg_;

    // CPU-side metadata (shared with CPU search path)
    std::unique_ptr<Initializer> initer_;
    std::unique_ptr<SaqData> saq_data_;
    std::unique_ptr<SaqDataMaker> saq_data_maker_;

    // GPU-resident encoded clusters
    std::vector<GpuSaqCluData> gpu_clusters_;

public:
    GpuIVF() = default;
    GpuIVF(size_t n, size_t num_dim, size_t k, QuantizeConfig cfg);
    ~GpuIVF();

    GpuIVF(const GpuIVF&) = delete;
    GpuIVF& operator=(const GpuIVF&) = delete;

    size_t num_data() const { return num_data_; }
    size_t num_dim() const { return num_dim_; }
    size_t k() const { return num_cen_; }
    const SaqData* get_saq_data() const { return saq_data_.get(); }

    void set_variance(FloatVec vars);

    /// GPU-accelerated index construction.
    /// data: [N x D] row-major float vectors
    /// centroids: [K x D] row-major float centroids
    /// cluster_ids: [N] cluster assignment per vector
    void construct(const FloatRowMat& data,
                   const FloatRowMat& centroids,
                   const PID* cluster_ids);

    /// Download all GPU cluster data to CPU SaqCluData structures.
    /// Useful for CPU-side search or save/load.
    std::vector<SaqCluData> download_clusters() const;

    /// Access GPU clusters directly (for future GPU search).
    const std::vector<GpuSaqCluData>& get_gpu_clusters() const { return gpu_clusters_; }
};

} // namespace saq::gpu

#endif // SAQ_USE_CUDA
```

**Step 2: Implement construct_gpu**

Create `src/gpu/gpu_ivf_construct.cu`:

```cpp
#ifdef SAQ_USE_CUDA

#include "saq/gpu/gpu_ivf.h"
#include "saq/gpu/gpu_utils.cuh"
#include "saq/gpu/gpu_encoder.cuh"
#include "saq/gpu/gpu_packer.cuh"
#include "saq/gpu/gpu_cluster_data.cuh"
#include "saq/initializer.h"
#include "saq/stopw.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include <cublas_v2.h>
#include <glog/logging.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

namespace saq::gpu {

GpuIVF::GpuIVF(size_t n, size_t num_dim, size_t k, QuantizeConfig cfg)
    : num_data_(n), num_dim_(num_dim), num_cen_(k), cfg_(std::move(cfg)),
      saq_data_maker_(std::make_unique<SaqDataMaker>(cfg_, num_dim)) {}

GpuIVF::~GpuIVF() = default;

void GpuIVF::set_variance(FloatVec vars) {
    saq_data_maker_->set_variance(std::move(vars));
}

void GpuIVF::construct(const FloatRowMat& data,
                       const FloatRowMat& centroids,
                       const PID* cluster_ids) {
    LOG(INFO) << "Starting GPU IVF construction...";
    StopW stopw;

    const size_t N = num_data_;
    const size_t D = num_dim_;
    const size_t K = num_cen_;

    // 1. Prepare CPU-side metadata
    // Initializer (centroid search stays on CPU for now)
    if (K < 20000ul) {
        initer_ = std::make_unique<FlatInitializer>(D, K);
    } else {
        LOG(FATAL) << "HNSW initializer not implemented";
    }
    initer_->set_centroids(centroids);

    // SaqData (quantization plan)
    if (!saq_data_maker_->is_variance_set()) {
        saq_data_maker_->compute_variance(data);
    }
    saq_data_ = saq_data_maker_->return_data();
    const auto& quant_plan = saq_data_->quant_plan;
    const auto& base_datas = saq_data_->base_datas;
    size_t num_segments = quant_plan.size();

    LOG(INFO) << "Quantization plan: " << num_segments << " segments";

    // 2. Compute cluster sizes and offsets (CPU-side)
    std::vector<size_t> cluster_sizes(K, 0);
    for (size_t i = 0; i < N; ++i) {
        cluster_sizes[cluster_ids[i]]++;
    }
    std::vector<size_t> cluster_offsets(K + 1, 0);
    std::partial_sum(cluster_sizes.begin(), cluster_sizes.end(), cluster_offsets.begin() + 1);

    // Sort vectors by cluster ID
    std::vector<uint32_t> sorted_indices(N);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0u);
    std::vector<uint32_t> sorted_cids(N);
    for (size_t i = 0; i < N; ++i) sorted_cids[i] = cluster_ids[i];

    // Stable sort by cluster ID
    std::vector<size_t> order(N);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
        [&](size_t a, size_t b) { return cluster_ids[a] < cluster_ids[b]; });

    // Build sorted arrays
    std::vector<uint32_t> h_sorted_cids(N);
    std::vector<uint32_t> h_sorted_original_ids(N);
    // Also build a sorted data matrix
    FloatRowMat sorted_data(N, D);
    for (size_t i = 0; i < N; ++i) {
        h_sorted_cids[i] = cluster_ids[order[i]];
        h_sorted_original_ids[i] = static_cast<uint32_t>(order[i]);
        sorted_data.row(i) = data.row(order[i]);
    }

    LOG(INFO) << "Data sorted by cluster. Uploading to GPU...";

    // 3. Upload to GPU
    auto d_vectors = device_alloc<float>(N * D);
    auto d_centroids = device_alloc<float>(K * D);
    auto d_cluster_ids = device_alloc<uint32_t>(N);

    upload(d_vectors.get(), sorted_data.data(), N * D);
    upload(d_centroids.get(), centroids.data(), K * D);
    upload(d_cluster_ids.get(), h_sorted_cids.data(), N);

    // 4. Allocate GPU cluster data
    gpu_clusters_.clear();
    gpu_clusters_.resize(K);
    for (size_t c = 0; c < K; ++c) {
        gpu_clusters_[c].allocate(cluster_sizes[c], quant_plan);
        // Upload original IDs for this cluster
        std::vector<uint32_t> clu_ids;
        for (size_t i = cluster_offsets[c]; i < cluster_offsets[c + 1]; ++i) {
            clu_ids.push_back(h_sorted_original_ids[i]);
        }
        if (!clu_ids.empty()) {
            upload(gpu_clusters_[c].d_ids.get(), clu_ids.data(), clu_ids.size());
        }
    }

    // cuBLAS handle
    CublasHandle cublas;

    // 5. Process each segment
    size_t dim_offset = 0;
    for (size_t seg = 0; seg < num_segments; ++seg) {
        size_t D_seg = quant_plan[seg].first;
        size_t num_bits = quant_plan[seg].second;
        const auto& bdata = base_datas[seg];

        LOG(INFO) << "Segment " << seg << ": dim=" << D_seg << " bits=" << num_bits;

        // Allocate per-segment temporaries
        auto d_residuals = device_alloc<float>(N * D_seg);
        auto d_rotated = device_alloc<float>(N * D_seg);

        // 5a. Subtract centroids
        launch_subtract_centroid(
            d_vectors.get(), d_centroids.get(), d_cluster_ids.get(),
            d_residuals.get(), dim_offset, D_seg, D, N);

        // 5b. Rotation: d_rotated = d_residuals * P^T
        if (bdata.rotator) {
            // Upload rotation matrix
            auto d_P = device_alloc<float>(D_seg * D_seg);
            upload(d_P.get(), bdata.rotator->get_P().data(), D_seg * D_seg);

            // cuBLAS GEMM: C = A * B
            // A = d_residuals [N x D_seg], B = P [D_seg x D_seg]
            // In row-major: cublasSgemm with transposed args
            float alpha = 1.0f, beta = 0.0f;
            SAQ_CUBLAS_CHECK(cublasSgemm(
                cublas.get(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                (int)D_seg, (int)N, (int)D_seg,
                &alpha,
                d_P.get(), (int)D_seg,         // B (column-major = P^T row-major)
                d_residuals.get(), (int)D_seg,  // A (column-major view)
                &beta,
                d_rotated.get(), (int)D_seg));  // C
        } else {
            // No rotation: copy residuals to rotated
            SAQ_CUDA_CHECK(cudaMemcpy(d_rotated.get(), d_residuals.get(),
                N * D_seg * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        // Also rotate centroids for ip_cent_oa computation
        auto d_centroids_seg = device_alloc<float>(K * D_seg);
        if (bdata.rotator) {
            // Extract centroid segment and rotate
            // For simplicity, do this on CPU and upload
            FloatRowMat cent_seg(K, D_seg);
            for (size_t c = 0; c < K; ++c) {
                cent_seg.row(c) = centroids.row(c).segment(dim_offset, D_seg);
            }
            FloatRowMat cent_rotated = cent_seg * bdata.rotator->get_P();
            upload(d_centroids_seg.get(), cent_rotated.data(), K * D_seg);
        } else {
            // Extract and upload unrotated centroid segments
            FloatRowMat cent_seg(K, D_seg);
            for (size_t c = 0; c < K; ++c) {
                cent_seg.row(c) = centroids.row(c).segment(dim_offset, D_seg);
            }
            upload(d_centroids_seg.get(), cent_seg.data(), K * D_seg);
        }

        // 5c. CAQ Encode
        auto d_codes = device_alloc<int>(N * D_seg);
        auto d_o_l2norm = device_alloc<float>(N);
        auto d_fac_rescale = device_alloc<float>(N);
        auto d_fac_error = device_alloc<float>(N);
        auto d_ip_cent_oa = device_alloc<float>(N);

        uint16_t code_max = (1 << num_bits) - 1;
        if (bdata.cfg.caq_ori_qB) {
            code_max = (1 << bdata.cfg.caq_ori_qB) - 1;
        }

        launch_caq_encode(
            d_rotated.get(), d_codes.get(),
            d_o_l2norm.get(), d_fac_rescale.get(), d_fac_error.get(), d_ip_cent_oa.get(),
            d_centroids_seg.get(), d_cluster_ids.get(),
            D_seg, N, K, num_bits, code_max,
            bdata.cfg.caq_adj_rd_lmt, bdata.cfg.caq_adj_eps, bdata.cfg.caq_ori_qB);

        // 5d. Pack short codes (per-vector)
        size_t short_code_bytes = D_seg / 8;
        auto d_short_raw = device_alloc<uint8_t>(N * short_code_bytes);
        launch_pack_short_codes(d_codes.get(), d_short_raw.get(), D_seg, N, num_bits);

        // 5e. Pack long codes (per-vector)
        size_t long_code_bytes = (num_bits > 1) ? D_seg * (num_bits - 1) / 8 : 0;
        auto d_long_raw = device_alloc<uint8_t>(N * (long_code_bytes > 0 ? long_code_bytes : 1));
        launch_pack_long_codes(d_codes.get(), d_long_raw.get(), D_seg, N, num_bits);

        // 5f. Scatter to per-cluster GpuSaqCluData
        SAQ_CUDA_CHECK(cudaDeviceSynchronize());

        for (size_t c = 0; c < K; ++c) {
            size_t clu_size = cluster_sizes[c];
            if (clu_size == 0) continue;
            size_t clu_offset = cluster_offsets[c];
            auto& gpu_seg = gpu_clusters_[c].segments[seg];

            // Copy centroid
            SAQ_CUDA_CHECK(cudaMemcpy(
                gpu_seg.d_centroid,
                d_centroids_seg.get() + c * D_seg,
                D_seg * sizeof(float), cudaMemcpyDeviceToDevice));

            // Copy factors
            launch_store_factors(
                d_o_l2norm.get(), d_ip_cent_oa.get(),
                d_fac_rescale.get(), d_fac_error.get(),
                gpu_seg.d_factor_o_l2norm, gpu_seg.d_factor_ip_cent_oa,
                gpu_seg.d_long_factor_rescale, gpu_seg.d_long_factor_error,
                nullptr, c, clu_size, clu_offset);

            // Copy short codes
            if (num_bits > 0 && gpu_seg.d_short_codes) {
                // TODO: fastscan reorder (Task 7)
                // For now, copy raw short codes
                SAQ_CUDA_CHECK(cudaMemcpy(
                    gpu_seg.d_short_codes,
                    d_short_raw.get() + clu_offset * short_code_bytes,
                    clu_size * short_code_bytes, cudaMemcpyDeviceToDevice));
            }

            // Copy long codes
            if (long_code_bytes > 0 && gpu_seg.d_long_codes) {
                SAQ_CUDA_CHECK(cudaMemcpy(
                    gpu_seg.d_long_codes,
                    d_long_raw.get() + clu_offset * long_code_bytes,
                    clu_size * long_code_bytes, cudaMemcpyDeviceToDevice));
            }
        }

        dim_offset += D_seg;
    }

    SAQ_CUDA_CHECK(cudaDeviceSynchronize());
    auto tm_ms = stopw.getElapsedTimeMicro() / 1000.0;
    LOG(INFO) << "GPU IVF construction done. Time: " << tm_ms / 1e3 << " s";
}

} // namespace saq::gpu

#endif // SAQ_USE_CUDA
```

**Step 3: Build to verify**

```bash
cmake --build build --target saq
```

**Step 4: Commit**

```bash
git add include/saq/gpu/gpu_ivf.h src/gpu/gpu_ivf_construct.cu
git commit -m "feat(gpu): add GpuIVF orchestrator with full GPU encode pipeline"
```

---

### Task 7: GPU Benchmark Sample

**Files:**
- Create: `samples/gpu_benchmark_sample.cpp`
- Modify: `samples/CMakeLists.txt`

**Step 1: Create GPU benchmark sample**

Create `samples/gpu_benchmark_sample.cpp` — a program that:
1. Loads preprocessed data (same as `saq_dbpedia_sample`)
2. Runs GPU encode via `GpuIVF::construct`
3. Downloads results to CPU for verification
4. Compares GPU encode output vs CPU encode output (recall check)
5. Prints timing comparison

```cpp
#ifdef SAQ_USE_CUDA

#include <iostream>
#include <vector>
#include <string>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "saq/defines.h"
#include "saq/config.h"
#include "saq/io_utils.h"
#include "saq/stopw.h"
#include "saq/gpu/gpu_ivf.h"
#include "index/ivf_index.h"

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <data_dir> <bpd> <K>" << std::endl;
        return 1;
    }

    std::string data_dir = argv[1];
    float bpd = std::stof(argv[2]);
    int K = std::stoi(argv[3]);

    LOG(INFO) << "GPU Benchmark: data=" << data_dir << " bpd=" << bpd << " K=" << K;

    // Load data
    auto vectors = saq::load_fvecs(data_dir + "/vectors_pca.fvecs");
    auto centroids = saq::load_fvecs(data_dir + "/centroids_" + std::to_string(K) + "_pca.fvecs");
    auto cluster_ids_mat = saq::load_ivecs(data_dir + "/cluster_ids_" + std::to_string(K) + ".ivecs");
    auto variances = saq::load_fvecs(data_dir + "/variances_pca.fvecs");

    size_t N = vectors.rows();
    size_t D = vectors.cols();

    LOG(INFO) << "Loaded: N=" << N << " D=" << D << " K=" << K;

    // Config
    saq::QuantizeConfig cfg;
    cfg.avg_bits = bpd;
    cfg.enable_segmentation = true;
    cfg.single.quant_type = saq::BaseQuantType::CAQ;
    cfg.single.caq_adj_rd_lmt = 6;
    cfg.single.use_fastscan = true;
    cfg.single.random_rotation = true;

    // ---- GPU Encode ----
    {
        saq::StopW sw;
        saq::gpu::GpuIVF gpu_ivf(N, D, K, cfg);
        gpu_ivf.set_variance(variances.row(0));

        std::vector<saq::PID> cids(N);
        for (size_t i = 0; i < N; ++i)
            cids[i] = static_cast<saq::PID>(cluster_ids_mat(i, 0));

        gpu_ivf.construct(vectors, centroids, cids.data());
        auto gpu_ms = sw.getElapsedTimeMicro() / 1000.0;
        LOG(INFO) << "GPU encode time: " << gpu_ms << " ms";
    }

    // ---- CPU Encode (for comparison) ----
    {
        saq::StopW sw;
        saq::IVF cpu_ivf(N, D, K, cfg);
        cpu_ivf.set_variance(variances.row(0));

        std::vector<saq::PID> cids(N);
        for (size_t i = 0; i < N; ++i)
            cids[i] = static_cast<saq::PID>(cluster_ids_mat(i, 0));

        cpu_ivf.construct(vectors, centroids, cids.data(), 8);
        auto cpu_ms = sw.getElapsedTimeMicro() / 1000.0;
        LOG(INFO) << "CPU encode time (8 threads): " << cpu_ms << " ms";
    }

    return 0;
}

#else

#include <iostream>
int main() {
    std::cerr << "GPU benchmark requires SAQ_BUILD_CUDA=ON" << std::endl;
    return 1;
}

#endif
```

**Step 2: Add to samples/CMakeLists.txt**

Read current `samples/CMakeLists.txt` and append:

```cmake
if(SAQ_BUILD_CUDA)
  add_executable(gpu_benchmark_sample gpu_benchmark_sample.cpp)
  target_link_libraries(gpu_benchmark_sample PRIVATE saq)
endif()
```

**Step 3: Build and run**

```bash
cmake -B build -DSAQ_BUILD_CUDA=ON -DSAQ_BUILD_SAMPLES=ON && cmake --build build
./build/samples/gpu_benchmark_sample data/datasets/dbpedia_100k 2.0 4096
```

Expected: Both GPU and CPU encode complete, GPU shows significant speedup.

**Step 4: Commit**

```bash
git add samples/gpu_benchmark_sample.cpp samples/CMakeLists.txt
git commit -m "feat(gpu): add GPU vs CPU encode benchmark sample"
```

---

### Task 8: Correctness Validation

**Files:**
- Create: `tests/test_gpu_encode.cpp`
- Modify: `tests/CMakeLists.txt`

**Step 1: Create GPU encode correctness test**

Create `tests/test_gpu_encode.cpp`:

A test that:
1. Creates a small synthetic dataset (1000 vectors, 128D, K=16)
2. Runs CPU encode via IVF::construct
3. Runs GPU encode via GpuIVF::construct (same SaqData/rotation)
4. Downloads GPU results
5. Compares codes, factors, and packed outputs element-by-element
6. Asserts max relative error < 1e-4 for float factors, exact match for integer codes

**Step 2: Build and run the test**

```bash
cmake -B build -DSAQ_BUILD_CUDA=ON -DSAQ_BUILD_TESTS=ON && cmake --build build
./build/tests/test_gpu_encode
```

Expected: All assertions pass.

**Step 3: Commit**

```bash
git add tests/test_gpu_encode.cpp tests/CMakeLists.txt
git commit -m "test(gpu): add GPU encode correctness validation test"
```

---

### Task 9: Create gpu-local and gpu-pace branches

**Step 1: Ensure gpu branch is clean and all tests pass**

```bash
git status
cmake --build build && ./build/tests/test_gpu_encode
```

**Step 2: Create gpu-local branch**

```bash
git checkout -b gpu-local
git push -u origin gpu-local
```

**Step 3: Create gpu-pace branch from gpu**

```bash
git checkout gpu
git checkout -b gpu-pace
git push -u origin gpu-pace
```

**Step 4: Document branch strategy**

The branches diverge for hardware-specific tuning:
- `gpu-local`: RTX 5090 (CC 10.0) — `CMAKE_CUDA_ARCHITECTURES=100`, block size tuning
- `gpu-pace`: A100/H100 (CC 8.0/9.0) — `CMAKE_CUDA_ARCHITECTURES="80;90"`, larger shared memory configs

**Step 5: Commit any branch-specific CMake defaults**

On `gpu-local`:
```bash
# Set default architecture to RTX 5090
# Modify CMakeLists.txt CUDA_ARCHITECTURES default to "100"
git commit -m "build(gpu-local): default CUDA arch to RTX 5090 (CC 10.0)"
```

On `gpu-pace`:
```bash
git checkout gpu-pace
# Set default architecture to A100/H100
# Modify CMakeLists.txt CUDA_ARCHITECTURES default to "80;90"
git commit -m "build(gpu-pace): default CUDA arch to A100/H100 (CC 8.0/9.0)"
```
