/// @file quantization_plan.cpp
/// @brief Implementation of SaqData serialization and SaqDataMaker DP/segmentation.
///
/// Ported faithfully from reference:
///   - saqlib/quantization/saq_data.hpp (SaqData::save/load, SaqDataMaker methods)

#include "saq/quantization_plan.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace saq {

// ============================================================================
// SaqData::save / SaqData::load
// ============================================================================

void SaqData::save(std::ofstream &output) const {
    output.write(reinterpret_cast<const char *>(&cfg), sizeof(QuantizeConfig));
    output.write(reinterpret_cast<const char *>(&num_dim), sizeof(size_t));
    save_floatvec(output, data_variance);

    auto size = base_datas.size();
    output.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
    for (const auto &bi : base_datas) {
        bi.save(output);
    }
}

void SaqData::load(std::ifstream &input) {
    input.read(reinterpret_cast<char *>(&cfg), sizeof(QuantizeConfig));
    input.read(reinterpret_cast<char *>(&num_dim), sizeof(size_t));
    load_floatvec(input, data_variance);
    CHECK_EQ(data_variance.cols(), static_cast<Eigen::Index>(num_dim))
        << "data_variance size mismatch with num_dim";

    size_t size;
    input.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    base_datas.clear();
    quant_plan.clear();
    base_datas.resize(size);
    for (auto &bi : base_datas) {
        bi.load(input);
        quant_plan.emplace_back(bi.num_dim_pad, bi.num_bits);
    }
}

// ============================================================================
// SaqDataMaker::compute_variance
// ============================================================================

void SaqDataMaker::compute_variance(const FloatRowMat &data) {
    CHECK_EQ(data.cols(), static_cast<Eigen::Index>(num_dim_padded_))
        << "Data dimension mismatch with padded dimension";
    FloatVec mean = data.colwise().mean();
    FloatVec data_variance = ((data.rowwise() - mean).array().square()).colwise().mean();
    set_variance(std::move(data_variance));
}

// ============================================================================
// SaqDataMaker::prepare_quantizers
// ============================================================================

void SaqDataMaker::prepare_quantizers() {
    CHECK_EQ(data_->data_variance.cols(), static_cast<Eigen::Index>(num_dim_padded_))
        << "please set_variance or compute_variance before prepare()";
    analyze_plan();

    data_->base_datas.clear();
    for (auto [dim, bit] : data_->quant_plan) {
        BaseQuantizerData bi;
        bi.num_dim_pad = dim;
        bi.num_bits = bit;
        bi.cfg = data_->cfg.single;
        bi.init();

        data_->base_datas.emplace_back(std::move(bi));
    }
}

// ============================================================================
// SaqDataMaker::equal_segmentation
// ============================================================================

SaqDataMaker::QuantPlanT SaqDataMaker::equal_segmentation(int num_segs) {
    size_t b = static_cast<size_t>(data_->cfg.avg_bits);
    LOG_IF(WARNING, b != data_->cfg.avg_bits)
        << fmt::format("Can not use float-point {} as equal-segment bits. Will use {} bit instead.",
                        data_->cfg.avg_bits, b);
    QuantPlanT quant_plan;
    size_t d = 0;
    for (int i = 0; i < num_segs && d < num_dim_padded_; i++) {
        auto t = rd_up_to_multiple_of((num_dim_padded_ - d) / (num_segs - i), kDimPaddingSize);
        quant_plan.push_back({t, b});
        d += t;
    }
    return quant_plan;
}

// ============================================================================
// SaqDataMaker::dynamic_programming
// ============================================================================

SaqDataMaker::QuantPlanT SaqDataMaker::dynamic_programming(
    const FloatVec &data_variance, float avg_bits) {

    CHECK_EQ(data_variance.cols(), static_cast<Eigen::Index>(num_dim_padded_));

    const auto num_bit_factors = kNumShortFactors * sizeof(float) * 8;
    const size_t tot_bits = static_cast<size_t>(avg_bits * num_dim_padded_ + num_bit_factors);
    const size_t max_num_segs = avg_bits < 2
        ? num_dim_padded_ / kDimPaddingSize
        : num_dim_padded_ / kDimPaddingSize / 2;
    constexpr auto valid_lmt = std::numeric_limits<double>::max();

    // DP table: f[ns][i][used_bits] = (min_error, backtrack_info)
    //   ns: number of segments used so far
    //   i: number of kDimPaddingSize-blocks covered so far
    //   used_bits: total bits consumed so far
    auto f = std::vector<std::vector<std::vector<std::pair<double, size_t>>>>(
        max_num_segs + 1,
        std::vector<std::vector<std::pair<double, size_t>>>(
            num_dim_padded_ / kDimPaddingSize + 1,
            std::vector<std::pair<double, size_t>>(
                tot_bits + 1, {valid_lmt, 0})));

    const size_t i_end = num_dim_padded_ / kDimPaddingSize;
    size_t ans_ns = 0;
    size_t ans_i = i_end;
    size_t ans_b = tot_bits;

    f[0][0][0] = {0, 0};

    for (size_t ns = 0; ns <= max_num_segs; ns++) {
        for (size_t i = 0; i <= i_end; i++) {
            for (size_t used_bits = 0; used_bits <= tot_bits; ++used_bits) {
                if (f[ns][i][used_bits].first < valid_lmt) {
                    if (i == i_end) {
                        // At the end of all dimensions: check if this is a better solution
                        if (f[ns][i][used_bits].first * (1.01) < f[ans_ns][ans_i][ans_b].first) {
                            ans_ns = ns;
                            ans_i = i;
                            ans_b = used_bits;
                        }
                        continue;
                    }
                    if (ns == max_num_segs) {
                        continue;
                    }

                    // Try extending with a new segment of j blocks with b bits each
                    double var_sum = 0;
                    for (size_t j = 1; (i + j) * kDimPaddingSize <= num_dim_padded_; j++) {
                        var_sum += data_variance
                                       .segment((i + j - 1) * kDimPaddingSize, kDimPaddingSize)
                                       .sum();

                        for (size_t b = 1; b <= kMaxQuantBit; ++b) {
                            auto B_new = used_bits + b * j * kDimPaddingSize + num_bit_factors;
                            if (B_new > tot_bits)
                                break;
                            auto v = var_sum / (1 << b);
                            auto &f_to = f[ns + 1][i + j][B_new];
                            if (f_to.first > f[ns][i][used_bits].first + v) {
                                f_to.first = f[ns][i][used_bits].first + v;
                                f_to.second = (i << 4) + b;
                            }
                        }
                    }
                    // Also try assigning 0 bits to remaining dimensions (unquantized tail)
                    auto err0 = var_sum;
                    if (f[ns][i][used_bits].first + err0 < f[1 + ns][i_end][used_bits].first) {
                        f[1 + ns][i_end][used_bits].first = f[ns][i][used_bits].first + err0;
                        f[1 + ns][i_end][used_bits].second = (i << 4) + 0;
                    }
                }
            }
        }
    }

    // Backtrack to reconstruct the optimal quantization plan
    QuantPlanT quant_plan;
    {
        size_t ns = ans_ns;
        size_t i = ans_i;
        size_t B = ans_b;
        while (i > 0) {
            auto &f_cur = f[ns][i][B];
            auto prev_i = (f_cur.second >> 4);
            auto curr_bits = f_cur.second & 0xf;
            auto curr_dim_len = (i - prev_i) * kDimPaddingSize;
            quant_plan.emplace_back(curr_dim_len, curr_bits);

            ns--;
            i = prev_i;
            if (curr_bits)
                B -= curr_bits * curr_dim_len + num_bit_factors;
        }

        // Reverse since we constructed the plan by working backwards
        std::reverse(quant_plan.begin(), quant_plan.end());
    }
    return quant_plan;
}

} // namespace saq
