/// @file bit_allocator.h
/// @brief Shared types and abstract base for bit allocators (DP, Greedy).
#pragma once

#include "saq/defines.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace saq {

enum class AllocatorKind : uint8_t { DP = 0, Greedy = 1 };

/// Configuration for a joint segmentation + bit-allocation problem.
struct JointAllocationConfig {
    size_t total_bits;             ///< Bit budget (D * avg_bits + factor overhead)
    size_t max_bits_per_dim;       ///< Per-dim bit cap (typically kMaxQuantBit = 13)
    size_t dim_padding_size;       ///< Block size; segments are multiples (typically 64)
    size_t num_dim_padded;         ///< Padded total dimension (multiple of dim_padding_size)
    size_t num_bit_factors;        ///< Fixed factor overhead added to each segment's bits
};

/// Result of a joint allocation: per-segment plan + summary stats.
struct BitAllocationResult {
    using QuantPlanT = std::vector<std::pair<size_t, size_t>>;
    QuantPlanT  quant_plan;        ///< (dim_length, bits) per segment, contiguous
    size_t      total_bits_used;   ///< Sum over segments of dim_length * bits + num_bit_factors per segment
    double      total_distortion;  ///< Sum of segment costs from the allocator's cost model
    std::string error;             ///< Empty on success; populated message on failure

    bool ok() const { return error.empty(); }
};

/// Abstract base — concrete allocators expose their own AllocateJoint(...) overloads
/// that differ in cost-source type (variance for DP, MSE table for Greedy).
class BitAllocator {
public:
    virtual ~BitAllocator() = default;
    virtual AllocatorKind kind() const = 0;
};

}  // namespace saq
