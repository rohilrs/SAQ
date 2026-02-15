#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <stdlib.h>
#include <vector>

#include <glog/logging.h>

#include "saq/defines.h"
#include "saq/memory.h"
#include "saq/tools.h"

namespace saq {

struct ExFactor {
    float rescale = 0;
    float error = 0;
};

class SaqCluData;

class CAQClusterData {
    friend SaqCluData;

  public:
    static constexpr size_t kNumShortFactors = 2; // factors packed into shortdata

    const size_t num_vec_;        // Num of vectors in this cluster
    const size_t num_vec_align_;  // Padded number of vectors (multiple of 32)
    const size_t num_dim_padded_; // Padded number of dimension (multiple of 64)
    const size_t num_bits_;       // bits
    const size_t num_blocks_;     // Num of blocks
  private:
    size_t shortb_factors_num_; // number of short block factors (in float)
    size_t shortb_code_bytes_;  // bytes of short block code
    size_t longb_code_bytes_;   // bytes of long block code

    size_t num_parallel_clusters_ = 1; // number of parallel clusters, that is, segments

    bool should_free_ = false;
    float *short_factors_ = nullptr;   // short factors
    uint8_t *short_code_ = nullptr;    // short code
    uint8_t *long_code_ = nullptr;     // long code
    ExFactor *long_factors_ = nullptr; // long factors of vectors
    PID *ids_ = nullptr;               // PID of vectors
    FloatVec centroid_;                // Rotated centroid of clusters

  public:
    /**
     * @brief Construct a new CAQClusterData object.
     * Data in the cluster are mapped to large arrays in memory.
     *
     * @param num_vec number of vectors
     * @param num_dim_padded padded number of dimensions (multiple of 64)
     * @param num_bits number of quantization bits
     */
    explicit CAQClusterData(size_t num_vec, size_t num_dim_padded, size_t num_bits)
        : num_vec_(num_vec),
          num_vec_align_(rd_up_to_multiple_of(num_vec, KFastScanSize)),
          num_dim_padded_(num_dim_padded),
          num_bits_(num_bits),
          num_blocks_(div_rd_up(num_vec, KFastScanSize)),
          shortb_factors_num_(KFastScanSize * kNumShortFactors),
          shortb_code_bytes_(num_bits ? num_dim_padded * KFastScanSize / 8 * sizeof(uint8_t) : 0),
          longb_code_bytes_(num_bits ? num_dim_padded * (num_bits - 1) / 8 : 0) {
        centroid_.resize(num_dim_padded);
    }

    ~CAQClusterData() {
        if (should_free_) {
            std::free(short_factors_);
            std::free(short_code_);
            std::free(long_code_);
            std::free(long_factors_);
            std::free(ids_);
        }
    }

    /**
     * @brief Return pointer to short code of i-th block in this cluster
     */
    auto short_code(size_t block_idx) { return &short_code_[shortb_code_bytes_ * block_idx]; }
    auto short_code(size_t block_idx) const { return &short_code_[shortb_code_bytes_ * block_idx]; }
    auto short_code_single(size_t vec_idx) const {
        auto block_idx = vec_idx / KFastScanSize;
        auto j = vec_idx % KFastScanSize;
        return short_code(block_idx) + num_dim_padded_ / 8 * j;
    }

    auto factor_o_l2norm(size_t block_idx) { return &short_factors_[block_idx * shortb_factors_num_]; }
    auto factor_o_l2norm(size_t block_idx) const { return &short_factors_[block_idx * shortb_factors_num_]; }

    // ip_cent_oa is optional
    auto factor_ip_cent_oa(size_t block_idx) { return factor_o_l2norm(block_idx) + KFastScanSize; }
    auto factor_ip_cent_oa(size_t block_idx) const { return factor_o_l2norm(block_idx) + KFastScanSize; }

    /**
     * @brief Return long code for i-th vector in this cluster
     */
    uint8_t *long_code(size_t vec_idx) {
        DCHECK_LT(vec_idx, num_vec_);
        return &long_code_[vec_idx * longb_code_bytes_];
    }
    uint8_t *long_code(size_t vec_idx) const {
        DCHECK_LT(vec_idx, num_vec_);
        return &long_code_[vec_idx * longb_code_bytes_];
    }

    /**
     * @brief Return long factor of i-th vector in this cluster
     */
    ExFactor &long_factor(size_t vec_idx) {
        return long_factors_[vec_idx * num_parallel_clusters_];
    }
    ExFactor &long_factor(size_t vec_idx) const {
        return long_factors_[vec_idx * num_parallel_clusters_];
    }

    auto &centroid() { return centroid_; }
    auto &centroid() const { return centroid_; }

    /**
     * @brief Return pointer to ids
     */
    PID *ids() { return this->ids_; }
    PID *ids() const { return this->ids_; }

    auto num_vec() const { return num_vec_; }
    auto num_blocks() const { return num_blocks_; }
    auto iter() const { return num_vec_ / KFastScanSize; }
    auto remain() const { return num_vec_ % KFastScanSize; }
};

class SaqCluData {
    static constexpr size_t kLongCodeAlignBytes = 16;

  public:
    const size_t num_vec_;       // Num of vectors in this segment
    const size_t num_vec_align_; // Num of vectors in this segment
    const size_t num_blocks_;    // Num of blocks
    const size_t num_segments_;  // Num of segments
  private:
    std::vector<CAQClusterData> segments_;
    size_t shortb_factors_fcnt_ = 0;  // count of short factors for all segments (in floats)
    size_t shortb_code_bytes_ = 0;    // bytes of short code for all segments
    size_t longb_code_bytes_ = 0;     // bytes of long block for all segments
    size_t longb_code_bytes_tot_ = 0; // total bytes of long block for all segments

    // ========================= persistent data below =========================
    float *short_factors_;                                    // short factors
    uint8_t *short_code_;                                     // short code
    uint8_t *long_code_;                                      // long code
    ExFactor *long_factors_;                                  // extra factors of vectors
    std::vector<PID, AlignedAllocator<PID, 64>> ids_;         // PID of vectors

  public:
    /**
     * @param num_vec number of vectors
     * @param quant_plan quantization plan for each segment. <num_dims_padded, bits>
     * @param use_compact_layout use compact long-code layout (forced true for single segment)
     */
    explicit SaqCluData(size_t num_vec, const std::vector<std::pair<size_t, size_t>> &quant_plan, bool use_compact_layout = false)
        : num_vec_(num_vec),
          num_vec_align_(rd_up_to_multiple_of(num_vec, KFastScanSize)),
          num_blocks_(div_rd_up(num_vec, KFastScanSize)),
          num_segments_(quant_plan.size()) {
        if (num_segments_ == 1)
            use_compact_layout = true;

        segments_.reserve(quant_plan.size());
        for (size_t i = 0; i < quant_plan.size(); ++i) {
            auto dim_padded = quant_plan[i].first;
            DCHECK_EQ(dim_padded % kDimPaddingSize, 0u);
            auto &c = segments_.emplace_back(num_vec, dim_padded, quant_plan[i].second);
            c.num_parallel_clusters_ = num_segments_;
            shortb_factors_fcnt_ += c.shortb_factors_num_;
            shortb_code_bytes_ += c.shortb_code_bytes_;

            if (use_compact_layout) {
                longb_code_bytes_ += c.longb_code_bytes_;
                longb_code_bytes_tot_ += rd_up_to_multiple_of(c.longb_code_bytes_ * num_vec, kLongCodeAlignBytes);
            } else {
                longb_code_bytes_ += rd_up_to_multiple_of(c.longb_code_bytes_, kLongCodeAlignBytes);
                longb_code_bytes_tot_ = longb_code_bytes_ * num_vec;
            }
        }

        // Assign short factors and short codes
        if (quant_plan.size() == 1) {
            auto blk_bytes = (shortb_factors_fcnt_ * sizeof(float) + shortb_code_bytes_);
            short_code_ = align_mm<64, uint8_t>(blk_bytes * num_blocks_);
            shortb_code_bytes_ = blk_bytes;
            short_factors_ = nullptr;
            shortb_factors_fcnt_ = 0;
            size_t ptr = 0;
            for (size_t i = 0; i < quant_plan.size(); ++i) {
                auto &c = segments_[i];

                c.short_factors_ = reinterpret_cast<float *>(short_code_ + ptr);
                ptr += c.shortb_factors_num_ * sizeof(float);
                c.shortb_factors_num_ = blk_bytes / sizeof(float);

                c.short_code_ = short_code_ + ptr;
                ptr += c.shortb_code_bytes_;
                c.shortb_code_bytes_ = blk_bytes;
            }
            assert(ptr == blk_bytes);
        } else {
            // TODO: optimize layout of short factors and codes
            short_factors_ = align_mm<64, float>(shortb_factors_fcnt_ * num_blocks_);
            short_code_ = align_mm<64, uint8_t>(shortb_code_bytes_ * num_blocks_);
            size_t shortb_factors_begin = 0;
            size_t shortb_code_begin = 0;
            for (size_t i = 0; i < quant_plan.size(); ++i) {
                auto &c = segments_[i];

                c.short_factors_ = short_factors_ + shortb_factors_begin;
                shortb_factors_begin += c.shortb_factors_num_;
                c.shortb_factors_num_ = shortb_factors_fcnt_;

                c.short_code_ = short_code_ + shortb_code_begin;
                shortb_code_begin += c.shortb_code_bytes_;
                c.shortb_code_bytes_ = shortb_code_bytes_;
            }
            assert(shortb_factors_fcnt_ == shortb_factors_begin);
            assert(shortb_code_bytes_ == shortb_code_begin);
        }

        // Assign long code and long_factor
        long_code_ = align_mm<64, uint8_t>(longb_code_bytes_tot_);
        long_factors_ = align_mm<64, ExFactor>(num_vec * num_segments_);
        ids_.resize(num_vec, 0);
        size_t longb_begin = 0;
        for (size_t i = 0; i < quant_plan.size(); ++i) {
            auto &c = segments_[i];
            if (use_compact_layout) {
                c.long_code_ = long_code_ + longb_begin;
                longb_begin += rd_up_to_multiple_of(c.longb_code_bytes_ * num_vec, kLongCodeAlignBytes);
            } else {
                c.long_code_ = long_code_ + longb_begin;
                longb_begin += rd_up_to_multiple_of(c.longb_code_bytes_, kLongCodeAlignBytes);
                c.longb_code_bytes_ = longb_code_bytes_;
            }

            c.long_factors_ = long_factors_ + i;
            c.ids_ = ids_.data();
        }
        assert(longb_begin == longb_code_bytes_tot_ || longb_begin == longb_code_bytes_);
    }

    ~SaqCluData() {
        if (short_factors_) {
            std::free(short_factors_);
        }
        std::free(short_code_);
        std::free(long_code_);
        std::free(long_factors_);
    }

    auto &get_segment(size_t idx) { return segments_[idx]; }
    auto &get_segment(size_t idx) const { return segments_[idx]; }

    /**
     * @brief Return pointer to ids
     */
    PID *ids() { return this->ids_.data(); }
    const PID *ids() const { return ids_.data(); }

    auto iter() const { return num_vec_ / KFastScanSize; }
    auto remain() const { return num_vec_ % KFastScanSize; }

    void load(std::ifstream &input);
    void save(std::ofstream &output) const;
};

} // namespace saq
