#pragma once

#ifdef SAQ_USE_CUDA

#include <memory>
#include <vector>

#include "saq/defines.h"
#include "saq/config.h"
#include "saq/quantization_plan.h"
#include "saq/initializer.h"
#include "saq/gpu/gpu_cluster_data.cuh"
#include "saq/gpu/gpu_memory_pool.h"
#include "saq/config.h"

namespace saq::gpu {

/// GPU-accelerated IVF index.
/// Performs encode on GPU, stores encoded data in GPU memory.
class GpuIVF {
    size_t num_data_ = 0;
    size_t num_dim_ = 0;
    size_t num_cen_ = 0;
    QuantizeConfig cfg_;

    std::unique_ptr<Initializer> initer_;
    std::unique_ptr<SaqData> saq_data_;
    std::unique_ptr<SaqDataMaker> saq_data_maker_;

    GpuMemoryPool pool_;
    std::vector<GpuSaqCluData> gpu_clusters_;
    DevicePtr<float> d_centroids_raw_;  // [K × D] raw centroids for GPU centroid search

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
    void construct(const FloatRowMat& data,
                   const FloatRowMat& centroids,
                   const PID* cluster_ids);

    /// Access GPU clusters directly.
    const std::vector<GpuSaqCluData>& get_gpu_clusters() const { return gpu_clusters_; }
    const GpuMemoryPool& get_pool() const { return pool_; }

    /// GPU-accelerated batch search.
    void search_batch(const FloatRowMat& queries,
                      size_t topk, size_t nprobe,
                      SearcherConfig cfg,
                      PID* results);
};

} // namespace saq::gpu

#endif // SAQ_USE_CUDA
