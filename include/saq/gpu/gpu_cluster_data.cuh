#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "saq/defines.h"

namespace saq::gpu {

/// GPU-resident per-segment cluster data (mirrors CAQClusterData layout).
struct GpuSegmentData {
    size_t num_dim_pad = 0;
    size_t num_bits = 0;
    size_t num_blocks = 0;

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
