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

#include <Eigen/Dense>
#include <glog/logging.h>
#include <fmt/core.h>

#include "saq/bit_allocator.h"
#include "saq/bit_allocator_greedy.h"
#include "saq/codebook_encoder.h"
#include "saq/defines.h"
#include "saq/config.h"
#include "saq/io_utils.h"
#include "saq/preprocessing/codebook_builder.h"
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

    /// Per-segment, per-dimension codebooks (empty = uniform mode).
    /// segment_codebooks[seg_idx][dim_within_seg].
    std::vector<std::vector<DimensionCodebook>> segment_codebooks;

    // Per-dimension reconstruction MSE at each bit-rate, from the codebook
    // builder. costs[d][bits]. Empty unless native derivation ran. Consumed
    // by the (future) greedy bit-allocation sub-project; unused here.
    std::vector<std::vector<float>> codebook_costs;

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
// MSE table builder for greedy allocation
// ============================================================================

/// @brief Build a per-dimension empirical MSE table for greedy bit allocation.
///
/// Returns a matrix of shape (num_dim_padded, max_bits + 1) where entry (d, b)
/// is the Lloyd reconstruction MSE for dimension d quantized at b bits.
/// Uses OpenMP parallelism when available. Typical runtime: 10-20 s for
/// D=1536 with max_bits=8 and OpenMP enabled.
///
/// @param data   Rotated data matrix, shape (N, num_dim_padded).
/// @param max_bits  Per-dimension bit cap; typically KMaxQuantizeBits (13).
Eigen::MatrixXf build_mse_table_for_allocation(const FloatRowMat &data,
                                               size_t max_bits);

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

    /// Non-owning pointer to the rotated data matrix — set by set_rotated_data().
    /// Only used when cfg.allocator == AllocatorKind::Greedy.
    const FloatRowMat *rotated_data_ = nullptr;
    bool rotated_data_set_ = false;

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

    /// @brief Register the rotated data matrix for greedy allocation.
    ///
    /// Must be called before set_variance() / compute_variance() when
    /// cfg.allocator == AllocatorKind::Greedy.  SaqDataMaker does NOT take
    /// ownership; the caller must ensure @p data outlives the call to
    /// set_variance() / compute_variance().
    void set_rotated_data(const FloatRowMat &data) {
        rotated_data_    = &data;
        rotated_data_set_ = true;
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

    /// @brief Analyze config and run the appropriate allocator.
    ///
    /// Dispatch order:
    ///   1. No segmentation → equal_segmentation(1)
    ///   2. Equal-segment override → equal_segmentation(seg_eqseg)
    ///   3. Greedy + rotated data available → build_mse_table_for_allocation + BitAllocatorGreedy
    ///   4. Greedy requested but data missing → warning + DP fallback
    ///   5. DP (default) → dynamic_programming(variance, avg_bits)
    void analyze_plan() {
        DCHECK_EQ(num_dim_padded_ % kDimPaddingSize, 0);

        if (!data_->cfg.enable_segmentation) {
            data_->quant_plan = equal_segmentation(1);
            return;
        }

        if (data_->cfg.seg_eqseg > 0) {
            data_->quant_plan = equal_segmentation(data_->cfg.seg_eqseg);
            return;
        }

        if (data_->cfg.allocator == AllocatorKind::Greedy) {
            if (!rotated_data_set_) {
                LOG(WARNING) << "AllocatorKind::Greedy requested but set_rotated_data() "
                                "was not called — falling back to DP allocator.";
            } else {
                auto mse_table = build_mse_table_for_allocation(*rotated_data_, kMaxQuantBit);

                JointAllocationConfig jcfg{};
                jcfg.num_dim_padded   = num_dim_padded_;
                jcfg.dim_padding_size = kDimPaddingSize;
                jcfg.max_bits_per_dim = kMaxQuantBit;
                jcfg.num_bit_factors  = kNumShortFactors * sizeof(float) * 8;
                jcfg.total_bits       = static_cast<size_t>(data_->cfg.avg_bits * num_dim_padded_)
                                        + jcfg.num_bit_factors;

                BitAllocatorGreedy alloc;
                auto r = alloc.AllocateJoint(mse_table, jcfg);
                CHECK(r.ok()) << "Greedy allocation failed: " << r.error;
                data_->quant_plan = std::move(r.quant_plan);
                return;
            }
        }

        // Default DP path — preserves existing behavior exactly.
        data_->quant_plan = dynamic_programming(data_->data_variance, data_->cfg.avg_bits);
    }

    /// @brief Uniformly partition dimensions into num_segs segments with equal bits.
    QuantPlanT equal_segmentation(int num_segs);

    /// @brief Joint DP over segmentation and bit allocation to minimize
    ///        quantization distortion (sum of variance / 2^bits per segment).
    QuantPlanT dynamic_programming(const FloatVec &data_variance, float avg_bits);
};

} // namespace saq
