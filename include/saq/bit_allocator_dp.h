#pragma once

#include "saq/bit_allocator.h"
#include "saq/defines.h"

namespace saq {

/// Joint DP allocator — co-optimizes segmentation and bit allocation using
/// the closed-form Bennett high-resolution cost model `D(seg) = var_sum(seg) / 2^bits`.
/// Algorithmically identical to the existing SaqDataMaker::dynamic_programming.
class BitAllocatorDP : public BitAllocator {
public:
    BitAllocatorDP() = default;
    AllocatorKind kind() const override { return AllocatorKind::DP; }

    /// `data_variance` is per-dim variance, length == config.num_dim_padded.
    BitAllocationResult AllocateJoint(const FloatVec &data_variance,
                                      const JointAllocationConfig &config) const;
};

}  // namespace saq
