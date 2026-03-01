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

        if (num_vec == 0) {
            segments.resize(num_segments);
            return;
        }

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
};

} // namespace saq::gpu
