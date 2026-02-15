#pragma once

/// @file quantization_plan.h
/// @brief SaqData (quantization plan container) and SaqDataMaker (plan builder).
///
/// Ported from reference:
///   - saqlib/quantization/saq_data.hpp     -> SaqData, SaqDataMaker
///   - saqlib/quantization/quantizer_data.hpp -> BaseQuantizerData
///
/// SaqData holds the complete quantization plan: per-dimension variance,
/// segment dimensions/bits, and per-segment rotators. SaqDataMaker computes
/// the optimal plan via joint dynamic programming over segmentation + bit
/// allocation.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <fmt/core.h>

#include "saq/defines.h"
#include "saq/config.h"
#include "saq/io_utils.h"
#include "saq/rotator.h"
#include "saq/tools.h"

namespace saq {

// ============================================================================
// BaseQuantizerData — per-segment quantization metadata + rotator
// ============================================================================

/// @brief Per-segment quantization data: padded dimension count, bit width,
///        quantization config, and optional random rotation matrix.
///
/// Ported from reference saqlib::BaseQuantizerData (quantizer_data.hpp).
struct BaseQuantizerData {
    size_t num_dim_pad;        ///< Padded dimension count for this segment
    size_t num_bits;           ///< Bits per dimension for this segment
    QuantSingleConfig cfg;     ///< Quantization configuration
    RotatorPtr rotator;        ///< Per-segment random rotation (optional)

    /// @brief Initialize the rotator if random_rotation is enabled.
    void init() {
        if (cfg.random_rotation) {
            rotator = std::make_unique<Rotator>(static_cast<uint32_t>(num_dim_pad));
            rotator->orthogonalize();
        }
    }

    /// @brief Serialize to binary stream.
    void save(std::ofstream &output) const {
        output.write(reinterpret_cast<const char *>(&num_dim_pad), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&num_bits), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&cfg), sizeof(QuantSingleConfig));
        char flags = rotator ? 1 : 0;
        output.write(&flags, sizeof(char));
        if (rotator) {
            rotator->save(output);
        }
    }

    /// @brief Deserialize from binary stream.
    void load(std::ifstream &input) {
        input.read(reinterpret_cast<char *>(&num_dim_pad), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&num_bits), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&cfg), sizeof(QuantSingleConfig));
        char flags;
        input.read(&flags, sizeof(char));
        if (flags) {
            rotator = std::make_unique<Rotator>(static_cast<uint32_t>(num_dim_pad));
            rotator->load(input);
        }
    }
};

// ============================================================================
// SaqData — complete quantization plan container
// ============================================================================

/// @brief Complete SAQ quantization plan: config, variance, segment plan,
///        and per-segment BaseQuantizerData entries.
///
/// Ported from reference saqlib::SaqData (saq_data.hpp).
struct SaqData {
    using QuantPlanT = std::vector<std::pair<size_t, size_t>>; ///< (dim_length, bits) per segment

    QuantizeConfig cfg;                        ///< Quantization configuration
    size_t num_dim;                            ///< Original (unpadded) dimension
    FloatVec data_variance;                    ///< Per-dimension variance (1 x num_dim_padded)
    std::vector<BaseQuantizerData> base_datas; ///< Per-segment quantizer data
    QuantPlanT quant_plan;                     ///< Quantization plan: (dim_length, bits) per segment

    /// @brief Serialize the entire SaqData to a binary stream.
    void save(std::ofstream &output) const;

    /// @brief Deserialize from a binary stream.
    void load(std::ifstream &input);

    /// @brief Convenience: save to a named file.
    void save(const std::string &filename) const {
        std::ofstream output(filename, std::ios::binary);
        CHECK(output.is_open()) << "Failed to open file for writing: " << filename;
        save(output);
        output.close();
    }

    /// @brief Convenience: load from a named file.
    void load(const std::string &filename) {
        std::ifstream input(filename, std::ios::binary);
        CHECK(input.is_open()) << "Failed to open file for reading: " << filename;
        load(input);
        input.close();
    }
};

// ============================================================================
// SaqDataMaker — builds SaqData via DP-based segmentation + bit allocation
// ============================================================================

/// @brief Constructs a SaqData by computing per-dimension variance and then
///        running joint dynamic programming over segmentation and bit allocation.
///
/// Ported from reference saqlib::SaqDataMaker (saq_data.hpp).
class SaqDataMaker {
  protected:
    using QuantPlanT = SaqData::QuantPlanT;
    static constexpr size_t kNumShortFactors = 2;
    static constexpr size_t kMaxQuantBit = KMaxQuantizeBits;

    const size_t num_dim_;        ///< Original dimension
    const size_t num_dim_padded_; ///< Padded dimension (multiple of kDimPaddingSize)
    std::unique_ptr<SaqData> data_;

  public:
    /// @brief Construct a SaqDataMaker with the given config and dimension.
    explicit SaqDataMaker(QuantizeConfig cfg, size_t num_dim)
        : num_dim_(num_dim),
          num_dim_padded_(rd_up_to_multiple_of(num_dim, kDimPaddingSize)),
          data_(std::make_unique<SaqData>()) {
        data_->cfg = std::move(cfg);
        data_->num_dim = num_dim_;
    }

    size_t getPaddedDim() const { return num_dim_padded_; }
    const SaqData *get_data() const { return data_.get(); }
    auto return_data() { return std::move(data_); }

    bool is_variance_set() const {
        return data_->data_variance.cols() != 0;
    }

    /// @brief Set per-dimension variance directly; pads with zeros if needed.
    void set_variance(FloatVec vars) {
        if (data_->data_variance.cols() < static_cast<int>(num_dim_padded_)) {
            data_->data_variance = FloatVec::Zero(num_dim_padded_);
            data_->data_variance.head(vars.cols()) = vars;
        } else {
            data_->data_variance = std::move(vars);
        }
        prepare_quantizers();
    }

    /// @brief Compute per-dimension variance from data matrix.
    void compute_variance(const FloatRowMat &data);

  protected:
    /// @brief Create BaseQuantizerData entries from the quantization plan.
    void prepare_quantizers();

    /// @brief Analyze config and run DP or equal segmentation.
    void analyze_plan() {
        DCHECK_EQ(num_dim_padded_ % kDimPaddingSize, 0);

        if (data_->cfg.enable_segmentation) {
            if (data_->cfg.seg_eqseg > 0) {
                data_->quant_plan = equal_segmentation(data_->cfg.seg_eqseg);
            } else {
                data_->quant_plan = dynamic_programming(data_->data_variance, data_->cfg.avg_bits);
            }
        } else {
            data_->quant_plan = equal_segmentation(1);
        }
    }

    /// @brief Uniformly partition dimensions into num_segs segments with equal bits.
    QuantPlanT equal_segmentation(int num_segs);

    /// @brief Joint DP over segmentation and bit allocation to minimize
    ///        quantization distortion (sum of variance / 2^bits per segment).
    QuantPlanT dynamic_programming(const FloatVec &data_variance, float avg_bits);
};

} // namespace saq
