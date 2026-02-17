#pragma once

/// @file saq_estimator.h
/// @brief SAQ multi-segment distance estimators that aggregate per-segment CAQ estimators.
///
/// Ported from reference saqlib/quantization/saq_estimator.hpp.
/// Contains:
///   - SaqEstimatorBase<CaqEstT>: base class managing per-segment CAQ estimators
///   - SaqCluEstimator<DistType>: fastscan cluster estimator (multi-segment)
///   - SaqCluEstimatorSingle<DistType>: non-fastscan cluster estimator (multi-segment)
///   - SaqSingleEstimator<DistType>: single-vector estimator (multi-segment)

#include <cstring>
#include <immintrin.h>
#include <memory>

#include <glog/logging.h>

#include "saq/defines.h"
#include "saq/caq_estimator.h"
#include "saq/cluster_data.h"
#include "saq/quantization_plan.h"

namespace saq {

template <typename CaqEstT>
class SaqEstimatorBase {
  protected:
    std::vector<CaqEstT> estimators_;

  public:
    SaqEstimatorBase(const SaqData &data, const SearcherConfig &searcher_cfg, const FloatVec &query) {
        auto &data_variance = data.data_variance;
        auto &base_datas = data.base_datas;

        for (size_t i = 0, offset = 0; i < base_datas.size(); ++i) {
            const auto &bdata = base_datas[i];
            FloatVec curr_query;
            if (bdata.num_dim_pad + offset > data.num_dim) {
                curr_query = FloatVec::Zero(bdata.num_dim_pad);
                auto copy_size = query.cols() - offset;
                curr_query.head(copy_size) = query.segment(offset, copy_size);
            } else {
                curr_query = query.segment(offset, bdata.num_dim_pad);
            }
            auto vars2 = (data_variance.segment(offset, bdata.num_dim_pad).array() * curr_query.array().square()).sum();

            estimators_.emplace_back(bdata, searcher_cfg, curr_query).setPruneBound(std::sqrt(vars2));

            offset += bdata.num_dim_pad;
        }
    }

    virtual ~SaqEstimatorBase() = default;

    auto getRuntimeMetrics() const {
        QueryRuntimeMetrics runtime_metrics;
        for (const auto &estimator : estimators_) {
            auto metrics = estimator.getRuntimeMetrics();
            runtime_metrics.acc_bitsum += metrics.acc_bitsum;
            runtime_metrics.fast_bitsum += metrics.fast_bitsum;
        }
        return runtime_metrics;
    }
};

template <DistType kDistType = DistType::Any>
class SaqCluEstimator : public SaqEstimatorBase<CaqCluEstimator<kDistType>> {
  protected:
    static constexpr size_t FAST_ARRAY = KFastScanSize / 16;
    static_assert(FAST_ARRAY == 2, "KFastScanSize must be 32 for SAQEstimator");

    using Base = SaqEstimatorBase<CaqCluEstimator<kDistType>>;
    using Base::estimators_;

    const SaqCluData *curr_saq_cluster_;

  public:
    SaqCluEstimator(const SaqData &data, const SearcherConfig &searcher_cfg, const FloatVec &query) : Base(data, searcher_cfg, query) {
        CHECK(kDistType == DistType::Any || kDistType == searcher_cfg.dist_type) << "distance type mismatch";
    }

    virtual ~SaqCluEstimator() = default;

    auto &getEstimators() const { return estimators_; }

    void prepare(const SaqCluData *saq_clust) {
        curr_saq_cluster_ = saq_clust;
        DCHECK_EQ(estimators_.size(), saq_clust->num_segments_);
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            estimators_[c_i].prepare(&saq_clust->get_segment(c_i));
        }
    }

#if defined(__AVX512F__)
    void varsEstDist(size_t block_idx, __m512 *fst_distances) {
        fst_distances[0] = _mm512_setzero_ps();
        fst_distances[1] = _mm512_setzero_ps();
        __m512 cd[2];
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            estimators_[c_i].varsEstDist(block_idx, cd);
            fst_distances[0] = _mm512_add_ps(fst_distances[0], cd[0]);
            fst_distances[1] = _mm512_add_ps(fst_distances[1], cd[1]);
        }
    }

    void compFastDist(size_t block_idx, __m512 *fst_distances) {
        fst_distances[0] = _mm512_setzero_ps();
        fst_distances[1] = _mm512_setzero_ps();
        __m512 cd[2];
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            estimators_[c_i].compFastDist(block_idx, cd);
            fst_distances[0] = _mm512_add_ps(fst_distances[0], cd[0]);
            fst_distances[1] = _mm512_add_ps(fst_distances[1], cd[1]);
        }
    }
#endif // defined(__AVX512F__)

    float compAccurateDist(size_t idx) {
        DCHECK_LT(idx, curr_saq_cluster_->num_vec_);
        float acc_dist = 0;
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            acc_dist += estimators_[c_i].compAccurateDist(idx);
        }
        return acc_dist;
    }

    using Base::getRuntimeMetrics;
};

template <DistType kDistType = DistType::Any>
class SaqCluEstimatorSingle : public SaqEstimatorBase<CaqCluEstimatorSingle<kDistType>> {
  protected:
    using Base = SaqEstimatorBase<CaqCluEstimatorSingle<kDistType>>;
    using Base::estimators_;

    const SaqCluData *curr_saq_cluster_;

  public:
    SaqCluEstimatorSingle(const SaqData &data, const SearcherConfig &searcher_cfg, const FloatVec &query) : Base(data, searcher_cfg, query) {
        CHECK(kDistType == DistType::Any || kDistType == searcher_cfg.dist_type) << "distance type mismatch";
    }

    virtual ~SaqCluEstimatorSingle() = default;

    auto &getEstimators() const { return estimators_; }

    void prepare(const SaqCluData *saq_clust) {
        curr_saq_cluster_ = saq_clust;
        DCHECK_EQ(estimators_.size(), saq_clust->num_segments_);
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            estimators_[c_i].prepare(&saq_clust->get_segment(c_i));
        }
    }

    float varsEstDist(size_t vec_idx) {
        float res = 0;
        for (auto &estimator : estimators_) {
            res += estimator.varsEstDist(vec_idx);
        }
        return res;
    }

    float compFastDist(size_t vec_idx) {
        float res = 0;
        for (auto &estimator : estimators_) {
            res += estimator.compFastDist(vec_idx);
        }
        return res;
    }

    float compAccurateDist(size_t vec_idx) {
        DCHECK_LT(vec_idx, curr_saq_cluster_->num_vec_);
        float acc_dist = 0;
        for (auto &estimator : estimators_) {
            acc_dist += estimator.compAccurateDist(vec_idx);
        }
        return acc_dist;
    }

    using Base::getRuntimeMetrics;
};

template <DistType kDistType = DistType::Any>
class SaqSingleEstimator : public SaqEstimatorBase<CaqSingleEstimator<kDistType>> {
  protected:
    using Base = SaqEstimatorBase<CaqSingleEstimator<kDistType>>;
    using Base::estimators_;

    const SaqCluData *curr_saq_cluster_;

  public:
    SaqSingleEstimator(const SaqData &data, const SearcherConfig &searcher_cfg, const FloatVec &query) : Base(data, searcher_cfg, query) {
        CHECK(kDistType == DistType::Any || kDistType == searcher_cfg.dist_type) << "distance type mismatch";
    }

    virtual ~SaqSingleEstimator() = default;

    auto &getEstimators() const { return estimators_; }

    float varsEstDist(const SaqSingleDataWrapper &wrapper) {
        float res = 0;
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            res += estimators_[c_i].varsEstDist(wrapper.get_segment(c_i));
        }
        return res;
    }

    float compFastDist(const SaqSingleDataWrapper &wrapper) {
        float res = 0;
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            res += estimators_[c_i].compFastDist(wrapper.get_segment(c_i));
        }
        return res;
    }

    float compAccurateDist(const SaqSingleDataWrapper &wrapper) {
        float acc_dist = 0;
        for (size_t c_i = 0; c_i < estimators_.size(); ++c_i) {
            acc_dist += estimators_[c_i].compAccurateDist(wrapper.get_segment(c_i));
        }
        return acc_dist;
    }

    using Base::getRuntimeMetrics;
};

} // namespace saq
