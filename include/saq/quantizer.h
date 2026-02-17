#pragma once

/// @file quantizer.h
/// @brief Per-segment quantizers: QuantizerCluster (batch) and QuantizerSingle.
///
/// Ported from reference saqlib/quantization/quantizer.hpp.
/// QuantizerCluster encodes all vectors in a cluster for one segment.
/// QuantizerSingle encodes a single vector for one segment.

#include <cassert>
#include <cmath>
#include <cstddef>

#include <glog/logging.h>

#include "saq/defines.h"
#include "saq/caq_encoder.h"
#include "saq/cluster_data.h"
#include "saq/cluster_packer.h"
#include "saq/code_helper.h"
#include "saq/config.h"
#include "saq/quantization_plan.h"
#include "saq/single_data.h"
#include "saq/pool.h"
#include "saq/rotator.h"

namespace saq {

struct QuantMetrics {
    AvgMaxRecorder norm_ip_o_oa;
};

class QuantizerCluster {
  public:
    const size_t num_bits_;
    const size_t num_dim_pad_;
    const BaseQuantizerData *data_;

  public:
    mutable QuantMetrics metrics_;

    QuantizerCluster(const BaseQuantizerData *data)
        : num_bits_(data->num_bits), num_dim_pad_(data->num_dim_pad), data_(data) {
    }

    virtual ~QuantizerCluster() {}

    virtual void quantize(const FloatRowMat &or_vecs, const FloatVec &centroid, CAQClusterData &clus) const {
        CHECK_EQ(or_vecs.cols(), static_cast<Eigen::Index>(num_dim_pad_))
            << "Input vector dimension does not match quantizer dimension";
        CHECK_EQ(centroid.cols(), static_cast<Eigen::Index>(num_dim_pad_))
            << "Centroid dimension does not match quantizer dimension";

        FloatRowMat o_vecs;
        if (data_->rotator) {
            clus.centroid() = centroid * data_->rotator->get_P();
            o_vecs = (or_vecs * data_->rotator->get_P()).rowwise() - clus.centroid();
        } else {
            clus.centroid() = centroid;
            o_vecs = or_vecs.rowwise() - clus.centroid();
        }

        const size_t num_points = clus.num_vec();
        CHECK(data_->cfg.quant_type == BaseQuantType::CAQ) << "Only CAQ is supported for DataQuantizer";
        CAQEncoder encoder(num_dim_pad_, num_bits_, data_->cfg);
        ClusterPacker packer(num_dim_pad_, num_bits_, clus, data_->cfg.use_fastscan);

        QuantBaseCode base_code;
        for (size_t i = 0; i < num_points; ++i) {
            const auto &curr_vec = o_vecs.row(i);
            encoder.encode_and_fac(curr_vec, base_code, &centroid);
            packer.store_and_pack(i, base_code);
            metrics_.norm_ip_o_oa.insert(base_code.norm_ip_o_oa);
        }
        packer.finalize_and_store();
    }
};

class QuantizerSingle {
  public:
    const size_t num_bits_;
    const size_t num_dim_pad_;
    const BaseQuantizerData *data_;

  public:
    mutable QuantMetrics metrics_;

    QuantizerSingle(const BaseQuantizerData *data)
        : num_bits_(data->num_bits), num_dim_pad_(data->num_dim_pad), data_(data) {
        CHECK_EQ(data->cfg.use_fastscan, false) << "Fastscan not supported for single vector quantizer";
        CHECK(data_->cfg.quant_type == BaseQuantType::CAQ) << "Only CAQ is supported for QuantizerSingle";
    }

    virtual ~QuantizerSingle() {}

    virtual void quantize(const FloatVec &or_vecs, CaqSingleDataWrapper *caq_data) const {
        CHECK_EQ(or_vecs.cols(), static_cast<Eigen::Index>(num_dim_pad_))
            << "Input vector dimension does not match quantizer dimension";

        FloatVec o_vecs;
        if (data_->rotator) {
            o_vecs = or_vecs * data_->rotator->get_P();
        } else {
            o_vecs = or_vecs;
        }

        CAQEncoder encoder(num_dim_pad_, num_bits_, data_->cfg);
        QuantBaseCode base_code;
        encoder.encode_and_fac(o_vecs, base_code, nullptr);

        store_quantization_result(caq_data, base_code);
        metrics_.norm_ip_o_oa.insert(base_code.norm_ip_o_oa);
    }

  private:
    void store_quantization_result(CaqSingleDataWrapper *single_data, const QuantBaseCode &base_code) const {
        single_data->factor_o_l2norm() = base_code.o_l2norm;

        if (num_bits_ == 0) {
            return;
        }

        DCHECK_EQ(num_dim_pad_, static_cast<size_t>(base_code.code.size()));

        single_data->factor_ip_cent_oa() = 0.0f;

        pack_short_codes(base_code.code, single_data->short_code());

        auto &ex_fac = single_data->long_factor();
        ex_fac.rescale = base_code.fac_rescale;
        ex_fac.error = base_code.fac_error;

        if (num_bits_ > 1) {
            pack_long_codes(base_code.code, single_data->long_code());
        }
    }

    void pack_short_codes(const Eigen::VectorXi &code, uint8_t *short_code_begin) const {
        const size_t shortcode_byte_num = num_dim_pad_ / 8;
        const uint16_t short_bit = num_bits_ ? (1 << (num_bits_ - 1)) : 0;

        for (size_t j = 0; j < shortcode_byte_num; ++j) {
            uint8_t byte = 0;
            const size_t base_idx = j * 8;

            byte |= (code[static_cast<Eigen::Index>(base_idx + 0)] & short_bit) ? 0x80 : 0;
            byte |= (code[static_cast<Eigen::Index>(base_idx + 1)] & short_bit) ? 0x40 : 0;
            byte |= (code[static_cast<Eigen::Index>(base_idx + 2)] & short_bit) ? 0x20 : 0;
            byte |= (code[static_cast<Eigen::Index>(base_idx + 3)] & short_bit) ? 0x10 : 0;
            byte |= (code[static_cast<Eigen::Index>(base_idx + 4)] & short_bit) ? 0x08 : 0;
            byte |= (code[static_cast<Eigen::Index>(base_idx + 5)] & short_bit) ? 0x04 : 0;
            byte |= (code[static_cast<Eigen::Index>(base_idx + 6)] & short_bit) ? 0x02 : 0;
            byte |= (code[static_cast<Eigen::Index>(base_idx + 7)] & short_bit) ? 0x01 : 0;

            auto j_cov = j + 7 - 2 * (j % 8);
            short_code_begin[j_cov] = byte;
        }
    }

    void pack_long_codes(const Eigen::VectorXi &code, uint8_t *long_code_begin) const {
        const uint16_t short_bit = 1 << (num_bits_ - 1);
        auto compacted_code_func = get_compacted_code16_func(static_cast<int>(num_bits_ - 1));

        Uint16Vec long_code_buffer;
        long_code_buffer.resize(1, static_cast<Eigen::Index>(num_dim_pad_));
        for (size_t j = 0; j < num_dim_pad_; ++j) {
            long_code_buffer[static_cast<Eigen::Index>(j)] = code[static_cast<Eigen::Index>(j)] & (short_bit - 1);
        }

        compacted_code_func(long_code_begin, &long_code_buffer(0, 0), num_dim_pad_);
    }
};

} // namespace saq
