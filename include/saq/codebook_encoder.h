#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <glog/logging.h>

namespace saq {

/// Per-dimension codebook: sorted centroids for DP-optimal quantization.
struct DimensionCodebook {
    size_t num_entries = 0;
    std::vector<float> centroids;  // sorted ascending

    /// Binary search: find nearest centroid index for a given value.
    int nearest(float value) const {
        int lo = 0, hi = static_cast<int>(num_entries) - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            float boundary = (centroids[mid] + centroids[mid + 1]) * 0.5f;
            if (value <= boundary) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    float centroid_value(int idx) const { return centroids[idx]; }
};

/// Build per-segment, per-dimension codebooks from either explicit codebooks or
/// Gaussian base codebooks scaled by per-dimension standard deviations.
///
/// @param quant_plan  Quantization plan: (dim_length, bits) per segment
/// @param codebooks_explicit  Explicit codebooks [seg][dim] (empty = use Gaussian)
/// @param gaussian_centroids  Gaussian base centroids [bits][entry] (empty = use explicit)
/// @param residual_stds  Per-dimension standard deviations for Gaussian scaling
/// @return  segment_codebooks[seg][dim_within_seg]
inline std::vector<std::vector<DimensionCodebook>> build_segment_codebooks(
    const std::vector<std::pair<size_t, size_t>>& quant_plan,
    const std::vector<std::vector<DimensionCodebook>>& codebooks_explicit,
    const std::vector<std::vector<float>>& gaussian_centroids,
    const std::vector<float>& residual_stds) {

    std::vector<std::vector<DimensionCodebook>> result;
    size_t dim_offset = 0;
    for (size_t s = 0; s < quant_plan.size(); ++s) {
        auto [seg_dims, seg_bits] = quant_plan[s];
        std::vector<DimensionCodebook> seg_cbs;
        if (seg_bits > 0) {
            bool have_cb = false;
            if (!codebooks_explicit.empty()) {
                have_cb = (s < codebooks_explicit.size());
            } else {
                have_cb = (seg_bits < gaussian_centroids.size()
                           && !gaussian_centroids[seg_bits].empty());
            }

            if (have_cb) {
                size_t k = 1u << seg_bits;
                if (!codebooks_explicit.empty()) {
                    CHECK(codebooks_explicit[s].size() >= seg_dims)
                        << "Explicit codebook for segment " << s
                        << " has " << codebooks_explicit[s].size()
                        << " dims, need " << seg_dims;
                } else {
                    CHECK(gaussian_centroids[seg_bits].size() >= k)
                        << "Gaussian codebook for " << seg_bits << " bits has "
                        << gaussian_centroids[seg_bits].size()
                        << " entries, need " << k;
                }
                for (size_t d = 0; d < seg_dims; d++) {
                    DimensionCodebook cb;
                    if (!codebooks_explicit.empty()) {
                        cb = codebooks_explicit[s][d];
                    } else {
                        size_t global_dim = dim_offset + d;
                        float sigma = residual_stds[global_dim];
                        cb.num_entries = k;
                        cb.centroids.resize(k);
                        const auto& base = gaussian_centroids[seg_bits];
                        for (size_t c = 0; c < k; c++) {
                            cb.centroids[c] = base[c] * sigma;
                        }
                        std::sort(cb.centroids.begin(), cb.centroids.end());
                    }
                    seg_cbs.push_back(std::move(cb));
                }
            }
        }
        result.push_back(std::move(seg_cbs));
        dim_offset += seg_dims;
    }
    return result;
}

}  // namespace saq
