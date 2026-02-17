#pragma once

/// @file saq_quantizer.h
/// @brief Top-level SAQ quantizers: SAQuantizer (cluster batch) and SAQuantizerSingle.
///
/// Ported from reference saqlib/quantization/saq_quantizer.hpp.
/// SAQuantizer orchestrates per-segment QuantizerCluster instances to encode
/// all vectors in a cluster across all segments.
/// SAQuantizerSingle orchestrates per-segment QuantizerSingle instances to
/// encode a single vector across all segments.

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include <glog/logging.h>

#include "saq/defines.h"
#include "saq/quantization_plan.h"
#include "saq/quantizer.h"
#include "saq/single_data.h"
#include "saq/tools.h"

namespace saq {

class SAQuantizer {
  protected:
    const size_t num_dim_;
    const size_t num_dim_padded_;
    std::vector<std::unique_ptr<QuantizerCluster>> data_quans_;
    const SaqData *data_;

  public:
    explicit SAQuantizer(const SaqData *data)
        : num_dim_(data->num_dim),
          num_dim_padded_(rd_up_to_multiple_of(num_dim_, kDimPaddingSize)),
          data_(data) {
        for (auto &bi : data_->base_datas) {
            data_quans_.emplace_back(std::make_unique<QuantizerCluster>(&bi));
        }
    }

    void quantize_cluster(const FloatRowMat &data, const FloatVec &centroid, const std::vector<PID> &IDs,
                          SaqCluData &saq_clus) {
        CHECK_EQ(saq_clus.num_segments_, data_quans_.size());
        std::copy(IDs.begin(), IDs.end(), saq_clus.ids());

        const size_t num_points = saq_clus.num_vec_;
        for (size_t ci = 0, offset = 0; ci < saq_clus.num_segments_; ++ci) {
            auto &clus = saq_clus.get_segment(ci);
            const size_t copy_size = std::min(clus.num_dim_padded_, num_dim_ - offset);

            FloatRowMat vecs(static_cast<Eigen::Index>(num_points), static_cast<Eigen::Index>(clus.num_dim_padded_));
            vecs.setZero();
            for (size_t r = 0; r < num_points; ++r) {
                auto id = clus.ids()[r];
                vecs.row(static_cast<Eigen::Index>(r)).head(static_cast<Eigen::Index>(copy_size)) =
                    data.row(static_cast<Eigen::Index>(id)).segment(static_cast<Eigen::Index>(offset), static_cast<Eigen::Index>(copy_size));
            }

            FloatVec cen(static_cast<Eigen::Index>(clus.num_dim_padded_));
            cen.setZero();
            cen.head(static_cast<Eigen::Index>(copy_size)) =
                centroid.segment(static_cast<Eigen::Index>(offset), static_cast<Eigen::Index>(copy_size));

            data_quans_[ci]->quantize(vecs, cen, clus);
            offset += clus.num_dim_padded_;
        }
    }
};

class SAQuantizerSingle {
  protected:
    const size_t num_dim_;
    const size_t num_dim_padded_;
    std::vector<std::unique_ptr<QuantizerSingle>> data_quans_;
    const SaqData *data_;

  public:
    SAQuantizerSingle(const SaqData *data)
        : num_dim_(data->num_dim),
          num_dim_padded_(rd_up_to_multiple_of(num_dim_, kDimPaddingSize)),
          data_(data) {
        for (auto &bi : data_->base_datas) {
            data_quans_.emplace_back(std::make_unique<QuantizerSingle>(&bi));
        }
    }

    void quantize(const FloatVec &or_vec, SaqSingleDataWrapper *caq_data) const {
        CHECK_EQ(caq_data->num_segments_, data_quans_.size());

        for (size_t ci = 0, offset = 0; ci < caq_data->num_segments_; ++ci) {
            auto &clus = caq_data->get_segment(ci);

            if (auto rem_sz = num_dim_ - offset; clus.num_dim_padded_ <= rem_sz) {
                data_quans_[ci]->quantize(
                    or_vec.segment(static_cast<Eigen::Index>(offset), static_cast<Eigen::Index>(clus.num_dim_padded_)),
                    &clus);
            } else {
                FloatVec t = FloatVec::Zero(static_cast<Eigen::Index>(clus.num_dim_padded_));
                t.head(static_cast<Eigen::Index>(rem_sz)) =
                    or_vec.segment(static_cast<Eigen::Index>(offset), static_cast<Eigen::Index>(rem_sz));
                data_quans_[ci]->quantize(t, &clus);
            }

            offset += clus.num_dim_padded_;
        }
    }
};

} // namespace saq
