#pragma once

/// @file single_data.h
/// @brief Single-vector quantization data wrappers for CAQ and SAQ.
///
/// Ported from reference saqlib/quantization/single_data.hpp.
/// CaqSingleDataWrapper stores codes/factors for a single vector in one segment.
/// SaqSingleDataWrapper manages memory for all segments of a single vector.

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#include <glog/logging.h>

#include "saq/defines.h"
#include "saq/cluster_data.h"
#include "saq/memory.h"
#include "saq/tools.h"

namespace saq {

class SaqSingleDataWrapper;

class CaqSingleDataWrapper {
    friend SaqSingleDataWrapper;

  public:
    static constexpr size_t kNumShortFactors = 2;

    const size_t num_dim_padded_;
    const size_t num_bits_;

  private:
    size_t shortb_factors_num_;
    size_t shortb_code_bytes_;
    size_t longb_code_bytes_;

    bool should_free_ = false;
    float *short_factors_ = nullptr;
    uint8_t *short_code_ = nullptr;
    uint8_t *long_code_ = nullptr;
    ExFactor *long_factors_ = nullptr;
    PID id_ = 0;

  public:
    explicit CaqSingleDataWrapper(size_t num_dim_padded, size_t num_bits)
        : num_dim_padded_(num_dim_padded),
          num_bits_(num_bits),
          shortb_factors_num_(kNumShortFactors),
          shortb_code_bytes_(num_bits ? num_dim_padded / 8 * sizeof(uint8_t) : 0),
          longb_code_bytes_(num_bits > 1 ? num_dim_padded * (num_bits - 1) / 8 : 0) {
    }

    ~CaqSingleDataWrapper() {
        if (should_free_) {
            portable_aligned_free(short_factors_);
            portable_aligned_free(short_code_);
            portable_aligned_free(long_code_);
            portable_aligned_free(long_factors_);
        }
    }

    void allocate_data() {
        should_free_ = true;
        if (shortb_factors_num_ > 0) {
            short_factors_ = align_mm<64, float>(shortb_factors_num_);
        }
        if (shortb_code_bytes_ > 0) {
            short_code_ = align_mm<64, uint8_t>(shortb_code_bytes_);
        }
        if (longb_code_bytes_ > 0) {
            long_code_ = align_mm<64, uint8_t>(longb_code_bytes_);
        }
        long_factors_ = align_mm<64, ExFactor>(1);
    }

    void set_pointers(float *short_factors, uint8_t *short_code, uint8_t *long_code, ExFactor *long_factors) {
        short_factors_ = short_factors;
        short_code_ = short_code;
        long_code_ = long_code;
        long_factors_ = long_factors;
    }

    auto short_code() { return short_code_; }
    auto short_code() const { return short_code_; }

    auto &factor_o_l2norm() { return short_factors_[0]; }
    auto factor_o_l2norm() const { return short_factors_[0]; }

    auto &factor_ip_cent_oa() { return short_factors_[1]; }
    auto factor_ip_cent_oa() const { return short_factors_[1]; }

    uint8_t *long_code() { return long_code_; }
    uint8_t *long_code() const { return long_code_; }

    ExFactor &long_factor() { return *long_factors_; }
    const ExFactor &long_factor() const { return *long_factors_; }

    PID id() const { return id_; }
    void set_id(PID id) { id_ = id; }

    auto num_dim_padded() const { return num_dim_padded_; }
    auto num_bits() const { return num_bits_; }
};

class SaqSingleDataWrapper {
  public:
    const size_t num_segments_;

  private:
    std::vector<CaqSingleDataWrapper> segments_;
    size_t shortb_factors_fcnt_ = 0;
    size_t shortb_code_bytes_ = 0;
    size_t longb_code_bytes_ = 0;
    size_t total_memory_size_ = 0;

    size_t short_factors_offset_ = 0;
    size_t short_code_offset_ = 0;
    size_t long_code_offset_ = 0;
    size_t long_factors_offset_ = 0;

    uint8_t *memory_base_ = nullptr;
    float *short_factors_;
    uint8_t *short_code_;
    uint8_t *long_code_;
    ExFactor *long_factors_;

  public:
    explicit SaqSingleDataWrapper(const std::vector<std::pair<size_t, size_t>> &quant_plan)
        : num_segments_(quant_plan.size()) {
        segments_.reserve(quant_plan.size());

        for (size_t i = 0; i < quant_plan.size(); ++i) {
            auto dim_padded = quant_plan[i].first;
            DCHECK_EQ(0u, dim_padded % kDimPaddingSize);
            auto &c = segments_.emplace_back(dim_padded, quant_plan[i].second);

            shortb_factors_fcnt_ += c.shortb_factors_num_;
            shortb_code_bytes_ += c.shortb_code_bytes_;
            longb_code_bytes_ += c.longb_code_bytes_;
        }

        size_t offset = 0;

        short_factors_offset_ = offset;
        offset += rd_up_to_multiple_of(shortb_factors_fcnt_ * sizeof(float), 64);

        short_code_offset_ = offset;
        offset += rd_up_to_multiple_of(shortb_code_bytes_, 64);

        long_code_offset_ = offset;
        offset += longb_code_bytes_;

        long_factors_offset_ = offset;
        offset += num_segments_ * sizeof(ExFactor);

        total_memory_size_ = offset;
    }

    void set_memory_base(uint8_t *memory_base) {
        DCHECK_EQ(reinterpret_cast<uintptr_t>(memory_base) % 64, 0u) << "memory_base must be 64-byte aligned";
        memory_base_ = memory_base;

        short_factors_ = reinterpret_cast<float *>(memory_base_ + short_factors_offset_);
        short_code_ = memory_base_ + short_code_offset_;
        long_code_ = memory_base_ + long_code_offset_;
        long_factors_ = reinterpret_cast<ExFactor *>(memory_base_ + long_factors_offset_);

        size_t short_factors_offset = 0;
        size_t short_code_offset = 0;
        size_t long_code_offset = 0;

        for (size_t i = 0; i < num_segments_; ++i) {
            auto &c = segments_[i];

            float *segment_short_factors = nullptr;
            uint8_t *segment_short_code = nullptr;
            uint8_t *segment_long_code = nullptr;
            ExFactor *segment_long_factors = &long_factors_[i];

            if (c.shortb_factors_num_ > 0) {
                segment_short_factors = short_factors_ + short_factors_offset;
                short_factors_offset += c.shortb_factors_num_;
            }

            if (c.shortb_code_bytes_ > 0) {
                segment_short_code = short_code_ + short_code_offset;
                short_code_offset += c.shortb_code_bytes_;
            }

            if (c.longb_code_bytes_ > 0) {
                segment_long_code = long_code_ + long_code_offset;
                long_code_offset += c.longb_code_bytes_;
            }

            c.set_pointers(segment_short_factors, segment_short_code, segment_long_code, segment_long_factors);
        }
    }

    static size_t calculate_memory_size(const std::vector<std::pair<size_t, size_t>> &quant_plan) {
        size_t shortb_factors_fcnt = 0;
        size_t shortb_code_bytes = 0;
        size_t longb_code_bytes = 0;
        size_t num_segments = quant_plan.size();

        for (const auto &plan : quant_plan) {
            auto dim_padded = rd_up_to_multiple_of(plan.first, kDimPaddingSize);
            auto num_bits = plan.second;

            shortb_factors_fcnt += CaqSingleDataWrapper::kNumShortFactors;
            if (num_bits > 0) {
                shortb_code_bytes += dim_padded / 8;
            }
            if (num_bits > 1) {
                longb_code_bytes += dim_padded * (num_bits - 1) / 8;
            }
        }

        size_t total_size = 0;
        total_size += rd_up_to_multiple_of(shortb_factors_fcnt * sizeof(float), 64);
        total_size += rd_up_to_multiple_of(shortb_code_bytes, 64);
        total_size += longb_code_bytes;
        total_size += num_segments * sizeof(ExFactor);

        return total_size;
    }

    ~SaqSingleDataWrapper() = default;

    auto &get_segment(size_t idx) { return segments_[idx]; }
    auto &get_segment(size_t idx) const { return segments_[idx]; }

    auto num_segments() const { return num_segments_; }
    auto total_memory_size() const { return total_memory_size_; }

    void clear() {
        if (memory_base_) {
            std::memset(memory_base_, 0, total_memory_size_);
        }
    }

    bool is_valid() const {
        return memory_base_ != nullptr;
    }
};

} // namespace saq
