#pragma once

/// @file codebook_encoder.h
/// @brief Encoder that uses precomputed DP-optimal codebooks instead of uniform quantization.
///
/// For each dimension, maps values to the nearest codebook centroid.
/// Computes factors (rescale, error) using actual codebook reconstruction.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include <glog/logging.h>

#include "saq/defines.h"
#include "saq/config.h"
#include "saq/caq_encoder.h"

namespace saq {

/// Per-dimension codebook: centroids sorted in ascending order.
/// For b bits, has 2^b entries.
struct DimensionCodebook {
    std::vector<float> centroids;  ///< Sorted centroids
    size_t num_entries = 0;

    /// Find nearest centroid index for a given value.
    int nearest(float value) const {
        if (num_entries <= 1) return 0;
        // Binary search on sorted centroids
        int lo = 0, hi = static_cast<int>(num_entries) - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            float boundary = (centroids[mid] + centroids[mid + 1]) * 0.5f;
            if (value <= boundary) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    float centroid_value(int code) const {
        if (code < 0 || code >= static_cast<int>(num_entries)) return 0.0f;
        return centroids[code];
    }
};

/// Encoder using per-dimension optimal codebooks.
/// Falls back to uniform CAQ when codebooks not available.
class CodebookEncoder {
    const size_t num_dim_pad_;
    const size_t num_bits_;
    const QuantSingleConfig &cfg_;
    const std::vector<DimensionCodebook> *codebooks_ = nullptr;  ///< External, not owned

  public:
    CodebookEncoder(size_t num_dim_pad, size_t num_bits, const QuantSingleConfig &cfg)
        : num_dim_pad_(num_dim_pad), num_bits_(num_bits), cfg_(cfg) {}

    void set_codebooks(const std::vector<DimensionCodebook> *cb) { codebooks_ = cb; }
    bool has_codebooks() const { return codebooks_ != nullptr && !codebooks_->empty(); }

    /// Encode using codebook nearest-centroid assignment.
    /// Produces a CaqCode with correct factors computed from codebook values.
    void encode(const FloatVec &o, CaqCode &caq) {
        if (num_bits_ == 0 || !has_codebooks()) {
            caq = CaqCode();
            return;
        }

        auto &code = caq.code;
        code.resize(num_dim_pad_);

        // Find nearest codebook entry per dimension
        double ip_o_oa = 0;
        double oa_l2sqr = 0;

        for (size_t j = 0; j < num_dim_pad_; j++) {
            float val = o[j];
            int c = 0;
            float oa_val = 0;

            if (j < codebooks_->size()) {
                const auto &cb = (*codebooks_)[j];
                c = cb.nearest(val);
                oa_val = cb.centroid_value(c);
            }

            code[j] = c;
            ip_o_oa += val * oa_val;
            oa_l2sqr += oa_val * oa_val;
        }

        caq.ip_o_oa = ip_o_oa;
        caq.oa_l2sqr = oa_l2sqr;
        caq.o_l2sqr = o.squaredNorm();
        caq.o_l2norm = std::sqrt(caq.o_l2sqr);

        // v_mx/v_mi/delta: set to match the codebook range for compatibility
        // with the storage pipeline (which expects these fields)
        float v_max = 0;
        for (size_t j = 0; j < num_dim_pad_; j++) {
            v_max = std::max(v_max, std::abs(o[j]));
        }
        caq.v_mx = v_max;
        caq.v_mi = -v_max;
        caq.delta = (2.0 * v_max) / ((1 << num_bits_));

        // Rescale factor
        caq.fac_rescale = ip_o_oa > 0 ? caq.o_l2sqr / ip_o_oa : 0;

        // Error factor
        if (ip_o_oa > 0 && oa_l2sqr > 0) {
            double cos2 = (ip_o_oa * ip_o_oa) / (caq.o_l2sqr * oa_l2sqr);
            caq.fac_error = caq.o_l2sqr * 1.9 *
                std::sqrt(std::max(0.0, (1.0 / cos2 - 1.0)) / (num_dim_pad_ - 1));
        } else {
            caq.fac_error = 0;
        }
    }

    void encode_and_fac(const FloatVec &curr_vec, QuantBaseCode &base_code,
                        const FloatVec *centroid) {
        CaqCode caq;
        encode(curr_vec, caq);

        if (num_bits_ == 0) {
            base_code = QuantBaseCode();
            return;
        }

        // For codebook path: store raw rescale (|o|^2 / <o, o_a>)
        // without the v_mx scaling that rescale_vmx_to1() adds.
        // The codebook distance computation uses this directly.
        base_code.o_l2norm = static_cast<float>(caq.o_l2norm);
        base_code.fac_rescale = static_cast<float>(caq.fac_rescale);  // = |o|^2 / <o, o_a>
        base_code.fac_error = static_cast<float>(caq.fac_error);

        // Still need rescale_vmx_to1 for the packed code pipeline (fascscan uses it)
        caq.rescale_vmx_to1();

        if (num_bits_ >= 1) {
            if (num_bits_ > 1 && centroid) {
                double ip_cent = 0;
                for (size_t j = 0; j < num_dim_pad_ && j < codebooks_->size(); j++) {
                    float oa_val = (*codebooks_)[j].centroid_value(caq.code[j]);
                    ip_cent += (*centroid)[j] * oa_val;
                }
                base_code.ip_cent_oa = static_cast<float>(ip_cent);
                base_code.norm_ip_o_oa = static_cast<float>(
                    caq.ip_o_oa / caq.o_l2norm / std::sqrt(caq.oa_l2sqr));
            }
            base_code.code = std::move(caq.code);
        }
    }
};

} // namespace saq
