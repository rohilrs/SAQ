#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

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

}  // namespace saq
