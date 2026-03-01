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

// Device-only warp intrinsics — only available when compiled by nvcc/nvrtc
#ifdef __CUDACC__

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

#endif // __CUDACC__

} // namespace saq::gpu
