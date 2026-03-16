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
            sp.centroids         = device_alloc<float>(K * D_seg);
            sp.factor_o_l2norm   = device_alloc<float>(total_blocks_ * KFastScanSize);
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
        size_t num_segs = quant_plan_.size();
        clu.num_vec = cluster_offsets[c + 1] - cluster_offsets[c];
        clu.num_segments = num_segs;
        clu.num_blocks = block_offsets[c + 1] - block_offsets[c];
        clu.segments.resize(num_segs);

        size_t vec_off = cluster_offsets[c];
        size_t blk_off = block_offsets[c];

        for (size_t s = 0; s < num_segs; ++s) {
            size_t D_seg = quant_plan_[s].first;
            size_t bits = quant_plan_[s].second;
            size_t num_codebooks = D_seg / 4;
            size_t long_bytes_per_vec = (bits > 1) ? D_seg * (bits - 1) / 8 : 0;

            auto& seg = clu.segments[s];
            seg.num_dim_pad = D_seg;
            seg.num_bits = bits;
            seg.num_blocks = clu.num_blocks;

            const auto& sp = segments[s];
            seg.d_centroid          = sp.centroids.get() + c * D_seg;
            seg.d_factor_o_l2norm   = sp.factor_o_l2norm.get() + blk_off * KFastScanSize;
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
