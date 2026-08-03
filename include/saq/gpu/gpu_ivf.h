#pragma once

#ifdef SAQ_USE_CUDA

#include <memory>
#include <vector>

#include "saq/codebook_encoder.h"
#include "saq/defines.h"
#include "saq/config.h"
#include "saq/preprocessing/codebook_builder.h"
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

    // Codebook support
    bool has_codebooks_ = false;
    std::vector<std::vector<DimensionCodebook>> codebooks_explicit_;
    std::vector<std::vector<float>> gaussian_codebook_centroids_;
    std::vector<float> residual_stds_;
    bool      derive_codebooks_ = false;  // derive natively from data in construct()
    LloydOpts lloyd_opts_{};


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

    bool has_codebooks() const { return has_codebooks_; }

    void set_gaussian_codebooks(
        std::vector<std::vector<float>> base_centroids,
        std::vector<float> residual_stds) {
        gaussian_codebook_centroids_ = std::move(base_centroids);
        residual_stds_ = std::move(residual_stds);
        has_codebooks_ = true;
    }

    void set_codebooks(std::vector<std::vector<DimensionCodebook>> cbs) {
        codebooks_explicit_ = std::move(cbs);
        has_codebooks_ = true;
    }

    /// Enable native, data-driven codebook derivation during construct().
    /// Mutually exclusive with set_codebooks()/set_gaussian_codebooks() (those
    /// inject precomputed codebooks instead). After calling, construct() will
    /// build per-dim Lloyd codebooks from the data matrix at the bit-counts
    /// allocated by quant_plan.
    void set_derive_codebooks(LloydOpts opts = {}) {
        lloyd_opts_ = opts;
        derive_codebooks_ = true;
        has_codebooks_ = true;
    }

    void set_variance(FloatVec vars);

    /// GPU-accelerated index construction.
    void construct(const FloatRowMat& data,
                   const FloatRowMat& centroids,
                   const PID* cluster_ids);

    /// Access GPU clusters directly.
    const std::vector<GpuSaqCluData>& get_gpu_clusters() const { return gpu_clusters_; }
    const GpuMemoryPool& get_pool() const { return pool_; }

    /// GPU-accelerated batch search.
    /// If out_dists != nullptr it receives the final top-k ADC distances
    /// (row-major [Q*topk], aligned with results) — needed to merge results
    /// across DB shards by distance.
    void search_batch(const FloatRowMat& queries,
                      size_t topk, size_t nprobe,
                      SearcherConfig cfg,
                      PID* results,
                      float* out_dists = nullptr);
};

} // namespace saq::gpu

#endif // SAQ_USE_CUDA
