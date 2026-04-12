/// @file ivf_index.cpp
/// @brief Implementation of non-template IVF methods.

#include "index/ivf_index.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "saq/preprocessing/preprocessing.h"

namespace saq {

void IVF::allocate_clusters(const std::vector<size_t> &cluster_sizes) {
    parallel_clusters_.clear();
    parallel_clusters_.reserve(num_cen_);
    for (size_t i = 0; i < num_cen_; ++i) {
        parallel_clusters_.emplace_back(cluster_sizes[i], saq_data_->quant_plan, cfg_.use_compact_layout);
    }
    LOG(INFO) << "Initializing done... num_points: " << num_data_;
}

void IVF::construct(const FloatRowMat &data, const FloatRowMat &centroids, const PID *cluster_ids,
                    int num_threads, bool use_1_centroid) {
    construct_impl(data, centroids, cluster_ids, num_threads, use_1_centroid, nullptr);
}

void IVF::construct_impl(const FloatRowMat &data, const FloatRowMat &centroids,
                         const PID *cluster_ids, int num_threads, bool use_1_centroid,
                         std::vector<std::vector<RawCodeMat>> *raw_codes_out) {
    LOG(INFO) << "Start IVF construction...\n";

    // 1. prepare initializer
    prepare_initer(&centroids);

    // 2. prepare SAQ data
    {
        if (!saq_data_maker_->is_variance_set()) {
            saq_data_maker_->compute_variance(data);
        }
        saq_data_ = saq_data_maker_->return_data();
        printQPlan(saq_data_.get());
    }

    // 3. prepare clusters
    std::vector<std::vector<PID>> id_lists(num_cen_);
    {
        std::vector<size_t> counts(num_cen_, 0);
        for (size_t i = 0; i < num_data_; ++i) {
            PID cid = cluster_ids[i];
            CHECK_LE(cid, static_cast<PID>(num_cen_)) << "Bad cluster id\n";
            id_lists[cid].push_back(static_cast<PID>(i));
            counts[cid] += 1;
        }
        allocate_clusters(counts);
    }

    // If the caller wants cached raw codes, allocate one slot per cluster.
    if (raw_codes_out) {
        raw_codes_out->clear();
        raw_codes_out->resize(num_cen_);
    }

    // 4. quantize clusters
    {
        FloatVec tot_avg_centroid;
        if (use_1_centroid) {
            tot_avg_centroid = data.colwise().mean();
        }
        SAQuantizer saq_quantizer_(saq_data_.get());
        LOG(INFO) << "Starting quantization of " << num_cen_ << " clusters...";
        StopW stopw;
#ifdef SAQ_USE_OPENMP
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
#else
        (void)num_threads;
#endif
        for (size_t i = 0; i < num_cen_; ++i) {
            if (i % 500 == 0) {
                LOG(INFO) << "Quantizing cluster " << i << "/" << num_cen_
                          << " (size=" << id_lists[i].size() << ")";
            }
            const FloatVec &cur_centroid = use_1_centroid ? tot_avg_centroid : FloatVec(centroids.row(i));
            auto &clu = parallel_clusters_[i];
            auto *per_seg_out = raw_codes_out ? &(*raw_codes_out)[i] : nullptr;
            saq_quantizer_.quantize_cluster(data, cur_centroid, id_lists[i], clu, per_seg_out);
        }
        auto tm_ms = stopw.getElapsedTimeMicro() / 1000.0;
        LOG(INFO) << "Quantization done. tm: " << tm_ms / 1e3 << " S";
    }

    // 5. Build id -> {cluster_idx, local_idx} map for decompress() and
    //    other lookup-based operations. Cheap and always useful.
    id_to_location_.clear();
    id_to_location_.reserve(num_data_);
    for (size_t cid = 0; cid < parallel_clusters_.size(); ++cid) {
        const PID *cluster_ids_ptr = parallel_clusters_[cid].ids();
        const size_t nv = parallel_clusters_[cid].num_vec_;
        for (size_t local = 0; local < nv; ++local) {
            id_to_location_[cluster_ids_ptr[local]] = {cid, local};
        }
    }
}

void IVF::fit(const FloatRowMat &X, bool apply_pca, int K, int seed, int num_threads) {
    num_data_ = static_cast<size_t>(X.rows());
    num_dim_  = static_cast<size_t>(X.cols());
    num_cen_  = static_cast<size_t>(K);
    saq_data_maker_ = std::make_unique<SaqDataMaker>(cfg_, num_dim_);

    // 1. Preprocessing (PCA + k-means). When apply_pca=true, k-means runs
    //    on already-rotated data inside fit_ivf_preprocessing, so the
    //    returned centroids are in rotated space.
    PreprocessingResult pp = fit_ivf_preprocessing(X, K, seed, apply_pca);

    // 2. Cache PCA state for decompress() inverse rotation.
    pca_applied_  = apply_pca;
    pca_mean_     = pp.pca.mean;      // row vector (1, D)
    pca_rotation_ = pp.pca.rotation;  // (D, D)

    set_variance(pp.pca.variances);

    // 3. Apply PCA to training data. Centroids are already rotated when
    //    apply_pca=true, so no extra transform is needed for them.
    //    FloatVec is a row vector -> broadcast with .rowwise() without transpose.
    FloatRowMat X_proc;
    if (apply_pca) {
        X_proc = (X.rowwise() - pp.pca.mean) * pp.pca.rotation;
    } else {
        X_proc = X;
    }
    const FloatRowMat &centroids_proc = pp.kmeans.centroids;

    // 4. Single-pass construction with raw-code caching.
    raw_codes_.clear();  // construct_impl will resize to num_cen_
    construct_impl(X_proc, centroids_proc, pp.kmeans.assignments.data(),
                   num_threads, /*use_1_centroid=*/false, &raw_codes_);
}

FloatRowMat IVF::decompress(const std::vector<PID> &ids) const {
    CHECK(saq_data_) << "decompress() called before fit() or construct()";
    CHECK(!raw_codes_.empty())
        << "decompress() requires fit() (not construct()) -- raw codes not cached";

    const auto &plan = saq_data_->quant_plan;  // vector<pair<dim_len, bits>>
    const auto &base = saq_data_->base_datas;  // vector<BaseQuantizerData>
    const size_t num_seg = plan.size();

    FloatRowMat result(static_cast<Eigen::Index>(ids.size()),
                       static_cast<Eigen::Index>(num_dim_));
    result.setZero();

    for (size_t i = 0; i < ids.size(); ++i) {
        const PID vid = ids[i];
        auto it = id_to_location_.find(vid);
        CHECK(it != id_to_location_.end())
            << "decompress(): ID " << vid << " not found in index";
        const size_t cid       = it->second.first;
        const size_t local_idx = it->second.second;

        const SaqCluData &saq_clu = parallel_clusters_[cid];

        // Build the PCA-rotated reconstruction one segment at a time.
        // FloatVec is a row vector (1, D); slice along columns with .segment().
        FloatVec o_rot = FloatVec::Zero(static_cast<Eigen::Index>(num_dim_));
        size_t offset = 0;

        for (size_t seg = 0; seg < num_seg; ++seg) {
            const auto &seg_data = base[seg];
            const auto &caq_seg  = saq_clu.get_segment(seg);
            const size_t seg_dim  = seg_data.num_dim_pad;
            const size_t num_bits = seg_data.num_bits;

            // Effective (unpadded) size for this segment; the trailing
            // padding dimensions are discarded when we write back.
            const size_t real_size = (offset < num_dim_)
                ? std::min(seg_dim, num_dim_ - offset)
                : 0;

            // Segment centroid is stored in per-segment rotated space.
            // Un-rotate via P^T to get back to the PCA-rotated space.
            FloatVec cen_unrot = seg_data.rotator
                ? (caq_seg.centroid() * seg_data.rotator->get_P().transpose()).eval()
                : FloatVec(caq_seg.centroid());

            FloatVec seg_vec;
            if (num_bits == 0) {
                // 0-bit segment: the residual reconstruction is just zero,
                // so the reconstruction collapses to the segment centroid.
                seg_vec = cen_unrot;
            } else {
                // num_bits > 0: dequantize from cached raw codes.
                const auto &seg_codes = raw_codes_[cid][seg];
                CHECK_EQ(seg_codes.cols(), static_cast<Eigen::Index>(seg_dim));

                // After rescale_vmx_to1() the stored quantizer is in the
                // normalized scale: v_mx=1, v_mi=-1, delta = 2/2^num_bits.
                // Matches CaqCode::get_oa(): oa[d] = code[d]*delta + v_mi
                // (no +0.5 half-step -- get_oa is what encode_and_fac uses
                // to compute ip_cent_oa, so the same formula is canonical
                // for reconstruction).
                const float v_mi  = -1.0f;
                const float delta = 2.0f / static_cast<float>(1u << num_bits);

                FloatVec oa_norm(static_cast<Eigen::Index>(seg_dim));
                for (size_t d = 0; d < seg_dim; ++d) {
                    const int code_d = static_cast<int>(
                        seg_codes(static_cast<Eigen::Index>(local_idx),
                                  static_cast<Eigen::Index>(d)));
                    oa_norm[static_cast<Eigen::Index>(d)] =
                        static_cast<float>(code_d) * delta + v_mi;
                }

                // CAQ preserves the residual's L2 norm via o_l2norm. The
                // rescale_vmx_to1 step normalizes delta to match v_mx=1, so
                // the reconstructed direction must be rescaled to |o|.
                const size_t block_idx = local_idx / KFastScanSize;
                const size_t lane      = local_idx % KFastScanSize;
                const float  o_l2norm  = caq_seg.factor_o_l2norm(block_idx)[lane];

                const float norm_oa = oa_norm.norm();
                if (norm_oa > 1e-9f) {
                    oa_norm *= (o_l2norm / norm_oa);
                }

                // Un-rotate per-segment rotator: (row_vec * P)^-1 = row_vec * P^T.
                FloatVec oa_unrot = seg_data.rotator
                    ? (oa_norm * seg_data.rotator->get_P().transpose()).eval()
                    : oa_norm;

                seg_vec = oa_unrot + cen_unrot;
            }

            // Write into the PCA-rotated full-dim vector (only real_size -- skip padding).
            if (real_size > 0) {
                o_rot.segment(static_cast<Eigen::Index>(offset),
                              static_cast<Eigen::Index>(real_size)) =
                    seg_vec.segment(0, static_cast<Eigen::Index>(real_size));
            }
            offset += seg_dim;
        }

        // Inverse PCA: x_original = o_rot * R^T + mean.
        FloatVec x_orig;
        if (pca_applied_) {
            x_orig = (o_rot * pca_rotation_.transpose()).eval();
            x_orig += pca_mean_;
        } else {
            x_orig = o_rot;
        }

        result.row(static_cast<Eigen::Index>(i)) = x_orig;
    }

    return result;
}

void IVF::save(const char *filename) const {
    if (parallel_clusters_.empty()) {
        LOG(ERROR) << "IVF not constructed\n";
        return;
    }
    std::ofstream output(filename, std::ios::binary);
    output.write(reinterpret_cast<const char *>(&num_data_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&num_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&num_cen_), sizeof(size_t));
    this->initer_->save(output, filename);
    saq_data_->save(output);
    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(num_cen_);
    for (const auto &cur_cluster : parallel_clusters_) {
        cluster_sizes.push_back(cur_cluster.num_vec_);
    }
    output.write(reinterpret_cast<const char *>(cluster_sizes.data()), sizeof(size_t) * num_cen_);
    for (const auto &pclu : parallel_clusters_) {
        pclu.save(output);
    }

    // PCA state (populated by fit(); defaults are fine when only construct() was called).
    output.write(reinterpret_cast<const char *>(&pca_applied_), sizeof(bool));
    if (pca_applied_) {
        const size_t D = static_cast<size_t>(pca_mean_.cols());  // FloatVec is (1, D)
        output.write(reinterpret_cast<const char *>(&D), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(pca_mean_.data()), D * sizeof(float));
        output.write(reinterpret_cast<const char *>(pca_rotation_.data()), D * D * sizeof(float));
    }

    // Raw codes (optional; only present if fit() was called).
    const bool has_raw_codes = !raw_codes_.empty();
    output.write(reinterpret_cast<const char *>(&has_raw_codes), sizeof(bool));
    if (has_raw_codes) {
        const size_t nc = raw_codes_.size();
        output.write(reinterpret_cast<const char *>(&nc), sizeof(size_t));
        for (const auto &cluster_segs : raw_codes_) {
            const size_t ns = cluster_segs.size();
            output.write(reinterpret_cast<const char *>(&ns), sizeof(size_t));
            for (const auto &m : cluster_segs) {
                const size_t rows = static_cast<size_t>(m.rows());
                const size_t cols = static_cast<size_t>(m.cols());
                output.write(reinterpret_cast<const char *>(&rows), sizeof(size_t));
                output.write(reinterpret_cast<const char *>(&cols), sizeof(size_t));
                if (rows > 0 && cols > 0) {
                    output.write(reinterpret_cast<const char *>(m.data()),
                                 rows * cols * sizeof(uint16_t));
                }
            }
        }
    }

    output.close();
}

void IVF::load(const char *filename) {
    free_memory();
    LOG(INFO) << "Loading IVF...\n";
    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());
    input.read(reinterpret_cast<char *>(&this->num_data_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&this->num_dim_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&this->num_cen_), sizeof(size_t));
    prepare_initer(nullptr);
    this->initer_->load(input, filename);
    saq_data_ = std::make_unique<SaqData>();
    saq_data_->load(input);
    std::vector<size_t> cluster_sizes(num_cen_, 0);
    input.read(reinterpret_cast<char *>(cluster_sizes.data()), sizeof(size_t) * num_cen_);
    DCHECK_EQ(num_data_, std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), static_cast<size_t>(0)));
    allocate_clusters(cluster_sizes);
    for (size_t i = 0; i < num_cen_; ++i) {
        parallel_clusters_[i].load(input);
    }

    // PCA state
    input.read(reinterpret_cast<char *>(&pca_applied_), sizeof(bool));
    if (pca_applied_) {
        size_t D = 0;
        input.read(reinterpret_cast<char *>(&D), sizeof(size_t));
        pca_mean_.resize(1, static_cast<Eigen::Index>(D));
        input.read(reinterpret_cast<char *>(pca_mean_.data()), D * sizeof(float));
        pca_rotation_.resize(static_cast<Eigen::Index>(D), static_cast<Eigen::Index>(D));
        input.read(reinterpret_cast<char *>(pca_rotation_.data()), D * D * sizeof(float));
    }

    // Raw codes
    bool has_raw_codes = false;
    input.read(reinterpret_cast<char *>(&has_raw_codes), sizeof(bool));
    raw_codes_.clear();
    if (has_raw_codes) {
        size_t nc = 0;
        input.read(reinterpret_cast<char *>(&nc), sizeof(size_t));
        raw_codes_.resize(nc);
        for (size_t c = 0; c < nc; ++c) {
            size_t ns = 0;
            input.read(reinterpret_cast<char *>(&ns), sizeof(size_t));
            raw_codes_[c].resize(ns);
            for (size_t s = 0; s < ns; ++s) {
                size_t rows = 0, cols = 0;
                input.read(reinterpret_cast<char *>(&rows), sizeof(size_t));
                input.read(reinterpret_cast<char *>(&cols), sizeof(size_t));
                raw_codes_[c][s].resize(static_cast<Eigen::Index>(rows),
                                        static_cast<Eigen::Index>(cols));
                if (rows > 0 && cols > 0) {
                    input.read(reinterpret_cast<char *>(raw_codes_[c][s].data()),
                               rows * cols * sizeof(uint16_t));
                }
            }
        }
    }

    // Rebuild id_to_location_ from loaded cluster data.
    id_to_location_.clear();
    id_to_location_.reserve(num_data_);
    for (size_t cid = 0; cid < parallel_clusters_.size(); ++cid) {
        const PID *cluster_ids = parallel_clusters_[cid].ids();
        const size_t nv = parallel_clusters_[cid].num_vec_;
        for (size_t local = 0; local < nv; ++local) {
            id_to_location_[cluster_ids[local]] = {cid, local};
        }
    }

    input.close();
    LOG(INFO) << "Index loaded\n";
}

} // namespace saq
