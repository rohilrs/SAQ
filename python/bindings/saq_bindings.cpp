/// @file saq_bindings.cpp
/// @brief Python bindings for the SAQ library using pybind11.
///
/// Wraps the IVF index, config structs, and enums for Python usage.
/// Uses pybind11/eigen.h for automatic numpy <-> Eigen matrix conversion.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "index/ivf_index.h"
#include "saq/codebook_encoder.h"
#include "saq/preprocessing/codebook_builder.h"
#include "saq/bit_allocator_dp.h"
#include "saq/bit_allocator_greedy.h"
#include "saq/config.h"
#include "saq/defines.h"
#include "saq/io_utils.h"

#include <span>

namespace py = pybind11;
using namespace saq;

PYBIND11_MODULE(_saq_core, m) {
    m.doc() = "SAQ: Scalar Additive Quantization (C++ core)";

    // ---- Enums ----
    py::enum_<DistType>(m, "DistType")
        .value("L2Sqr", DistType::L2Sqr)
        .value("IP", DistType::IP)
        .export_values();

    py::enum_<BaseQuantType>(m, "BaseQuantType")
        .value("CAQ", BaseQuantType::CAQ)
        .value("RBQ", BaseQuantType::RBQ)
        .value("LVQ", BaseQuantType::LVQ)
        .export_values();

    py::enum_<AllocatorKind>(m, "AllocatorKind")
        .value("DP", AllocatorKind::DP)
        .value("Greedy", AllocatorKind::Greedy)
        .export_values();

    // ---- QuantSingleConfig ----
    py::class_<QuantSingleConfig>(m, "QuantSingleConfig")
        .def(py::init<>())
        .def_readwrite("quant_type", &QuantSingleConfig::quant_type)
        .def_readwrite("random_rotation", &QuantSingleConfig::random_rotation)
        .def_readwrite("use_fastscan", &QuantSingleConfig::use_fastscan)
        .def_readwrite("caq_adj_rd_lmt", &QuantSingleConfig::caq_adj_rd_lmt)
        .def_readwrite("caq_adj_eps", &QuantSingleConfig::caq_adj_eps);

    // ---- QuantizeConfig ----
    py::class_<QuantizeConfig>(m, "QuantizeConfig")
        .def(py::init<>())
        .def_readwrite("avg_bits", &QuantizeConfig::avg_bits)
        .def_readwrite("seg_eqseg", &QuantizeConfig::seg_eqseg)
        .def_readwrite("enable_segmentation", &QuantizeConfig::enable_segmentation)
        .def_readwrite("use_compact_layout", &QuantizeConfig::use_compact_layout)
        .def_readwrite("allocator", &QuantizeConfig::allocator)
        .def_readwrite("single", &QuantizeConfig::single);

    // ---- SearcherConfig ----
    py::class_<SearcherConfig>(m, "SearcherConfig")
        .def(py::init<>())
        .def_readwrite("dist_type", &SearcherConfig::dist_type)
        .def_readwrite("searcher_vars_bound_m", &SearcherConfig::searcher_vars_bound_m);

    // ---- IVF ----
    py::class_<IVF>(m, "IVF")
        .def(py::init<>(), "Default constructor (use load() to populate).")
        .def(py::init<size_t, size_t, size_t, QuantizeConfig>(),
             py::arg("n"), py::arg("dim"), py::arg("k"), py::arg("config"),
             "Create IVF index. Args: num_vectors, dimension, num_clusters, config.")
        .def("set_variance",
             [](IVF &self, py::array_t<float, py::array::c_style> variances) {
                 py::buffer_info buf = variances.request();
                 if (buf.ndim != 1 && buf.ndim != 2) {
                     throw std::runtime_error("variances must be 1D or 2D");
                 }
                 size_t dim = (buf.ndim == 1) ? buf.shape[0] : buf.shape[1];
                 const float *ptr = static_cast<const float *>(buf.ptr);
                 FloatVec var_vec = Eigen::Map<const FloatVec>(ptr, dim);
                 self.set_variance(std::move(var_vec));
             },
             py::arg("variances"),
             "Set per-dimension variance (1D float array).")
        .def("construct",
             [](IVF &self, Eigen::Ref<const FloatRowMat> data,
                Eigen::Ref<const FloatRowMat> centroids,
                py::array_t<uint32_t, py::array::c_style> cluster_ids,
                int num_threads) {
                 py::buffer_info ids_buf = cluster_ids.request();
                 if (ids_buf.ndim == 2) {
                     // Flatten (N,1) to (N,)
                     if (ids_buf.shape[1] != 1) {
                         throw std::runtime_error("cluster_ids must be 1D or (N,1)");
                     }
                 }
                 const PID *ids_ptr = static_cast<const PID *>(ids_buf.ptr);
                 {
                     py::gil_scoped_release release;
                     self.construct(data, centroids, ids_ptr, num_threads);
                 }
             },
             py::arg("data"), py::arg("centroids"), py::arg("cluster_ids"),
             py::arg("num_threads") = 8,
             "Build IVF index from data, centroids, and cluster assignments.")
        .def("search",
             [](IVF &self, Eigen::Ref<const Eigen::RowVectorXf> query,
                size_t topk, size_t nprobe, SearcherConfig searcher_cfg) {
                 py::array_t<uint32_t> results(topk);
                 auto *results_ptr = static_cast<PID *>(results.mutable_data());
                 {
                     py::gil_scoped_release release;
                     if (searcher_cfg.dist_type == DistType::IP) {
                         self.search<DistType::IP>(query, topk, nprobe, searcher_cfg, results_ptr);
                     } else {
                         self.search<DistType::L2Sqr>(query, topk, nprobe, searcher_cfg, results_ptr);
                     }
                 }
                 return results;
             },
             py::arg("query"), py::arg("topk"), py::arg("nprobe"),
             py::arg("config") = SearcherConfig(),
             "Search for topk nearest neighbors. Returns uint32 array of IDs.")
        .def("search_batch",
             [](IVF &self, Eigen::Ref<const FloatRowMat> queries,
                size_t topk, size_t nprobe, SearcherConfig searcher_cfg) {
                 size_t nq = static_cast<size_t>(queries.rows());
                 py::array_t<uint32_t> results({static_cast<py::ssize_t>(nq),
                                                 static_cast<py::ssize_t>(topk)});
                 auto *results_ptr = static_cast<PID *>(results.mutable_data());
                 {
                     py::gil_scoped_release release;
                     for (size_t q = 0; q < nq; ++q) {
                         PID *row_ptr = results_ptr + q * topk;
                         if (searcher_cfg.dist_type == DistType::IP) {
                             self.search<DistType::IP>(queries.row(q), topk, nprobe,
                                                       searcher_cfg, row_ptr);
                         } else {
                             self.search<DistType::L2Sqr>(queries.row(q), topk, nprobe,
                                                           searcher_cfg, row_ptr);
                         }
                     }
                 }
                 return results;
             },
             py::arg("queries"), py::arg("topk"), py::arg("nprobe"),
             py::arg("config") = SearcherConfig(),
             "Batch search. Returns uint32 array of shape (nq, topk).")
        .def("save",
             [](const IVF &self, const std::string &filename) {
                 py::gil_scoped_release release;
                 self.save(filename.c_str());
             },
             py::arg("filename"), "Save index to file.")
        .def("load",
             [](IVF &self, const std::string &filename) {
                 py::gil_scoped_release release;
                 self.load(filename.c_str());
             },
             py::arg("filename"), "Load index from file.")
        .def_property_readonly("num_data", &IVF::num_data)
        .def_property_readonly("num_dim", &IVF::num_dim)
        .def_property_readonly("k", &IVF::k)
        .def("fit",
             [](IVF &self, Eigen::Ref<const FloatRowMat> X,
                bool apply_pca, int K, int seed, int num_threads) {
                 py::gil_scoped_release release;
                 self.fit(X, apply_pca, K, seed, num_threads);
             },
             py::arg("X"), py::arg("apply_pca") = true,
             py::arg("K") = 4096, py::arg("seed") = 0,
             py::arg("num_threads") = 8,
             "Run preprocessing + construction from raw (N, D) vectors.")
        .def("decompress",
             [](IVF &self, py::array_t<uint32_t, py::array::c_style> ids) {
                 py::buffer_info buf = ids.request();
                 if (buf.ndim != 1) {
                     throw std::runtime_error("ids must be a 1D array");
                 }
                 const PID *ptr = static_cast<const PID *>(buf.ptr);
                 size_t n = static_cast<size_t>(buf.shape[0]);
                 std::vector<PID> id_vec(ptr, ptr + n);
                 FloatRowMat result;
                 {
                     py::gil_scoped_release release;
                     result = self.decompress(id_vec);
                 }
                 return result;
             },
             py::arg("ids"),
             "Approximate reconstruction of vectors by global ID. Returns float32 (n, dim).")
        .def("set_codebooks",
             [](IVF &self, py::list codebooks_list) {
                 // codebooks_list: list of list of numpy arrays
                 // codebooks_list[seg][dim] = 1D float array of sorted centroids
                 std::vector<std::vector<DimensionCodebook>> cbs;
                 for (auto seg_obj : codebooks_list) {
                     py::list seg_list = seg_obj.cast<py::list>();
                     std::vector<DimensionCodebook> seg_cbs;
                     for (auto dim_obj : seg_list) {
                         auto arr = dim_obj.cast<py::array_t<float, py::array::c_style>>();
                         auto buf = arr.request();
                         DimensionCodebook cb;
                         cb.num_entries = buf.shape[0];
                         cb.centroids.assign(
                             static_cast<float*>(buf.ptr),
                             static_cast<float*>(buf.ptr) + cb.num_entries);
                         seg_cbs.push_back(std::move(cb));
                     }
                     cbs.push_back(std::move(seg_cbs));
                 }
                 self.set_codebooks(std::move(cbs));
             },
             py::arg("codebooks"),
             "Set per-segment, per-dimension codebooks. Each segment is a list of 1D float arrays (sorted centroids).")
        .def("set_gaussian_codebooks",
             [](IVF &self, py::dict codebooks_dict, py::array_t<float> variances) {
                 // codebooks_dict: {bits: numpy 1D array of base centroids}
                 // variances: 1D float array of per-dimension variance
                 std::vector<std::vector<float>> base_centroids;
                 size_t max_bits = 0;
                 for (auto item : codebooks_dict) {
                     size_t b = item.first.cast<size_t>();
                     if (b > max_bits) max_bits = b;
                 }
                 base_centroids.resize(max_bits + 1);
                 for (auto item : codebooks_dict) {
                     size_t b = item.first.cast<size_t>();
                     auto arr = item.second.cast<py::array_t<float, py::array::c_style>>();
                     auto buf = arr.request();
                     size_t expected = static_cast<size_t>(1) << b;
                     if (static_cast<size_t>(buf.shape[0]) != expected) {
                         throw std::invalid_argument(
                             "codebook for bits=" + std::to_string(b) +
                             " has " + std::to_string(buf.shape[0]) +
                             " entries, expected " + std::to_string(expected));
                     }
                     base_centroids[b].assign(
                         static_cast<float*>(buf.ptr),
                         static_cast<float*>(buf.ptr) + buf.shape[0]);
                 }

                 auto var_buf = variances.request();
                 size_t ndim = var_buf.shape[0];
                 std::vector<float> stds(ndim);
                 float* var_ptr = static_cast<float*>(var_buf.ptr);
                 for (size_t i = 0; i < ndim; ++i)
                     stds[i] = std::sqrt(var_ptr[i]);

                 self.set_gaussian_codebooks(std::move(base_centroids), std::move(stds));
             },
             py::arg("codebooks"), py::arg("variances"),
             "Set Gaussian base codebooks + per-dimension variances. "
             "codebooks: dict {bits: 1D float array with 2^bits entries}, variances: 1D float array.")
        .def("set_derive_codebooks",
             [](IVF &self, size_t max_bits, size_t restarts,
                size_t max_iters, uint64_t seed, size_t sample_size) {
                 LloydOpts opts;
                 opts.max_bits    = max_bits;
                 opts.restarts    = restarts;
                 opts.max_iters   = max_iters;
                 opts.seed        = seed;
                 opts.sample_size = sample_size;
                 self.set_derive_codebooks(opts);
             },
             py::arg("max_bits") = 13, py::arg("restarts") = 1,
             py::arg("max_iters") = 50, py::arg("seed") = 0,
             py::arg("sample_size") = 0,
             "Enable native data-driven Lloyd (k-means) codebook derivation "
             "during construct()/fit(). Mutually exclusive with set_codebooks() "
             "and set_gaussian_codebooks(). Builds per-dimension Lloyd codebooks "
             "from the (PCA-transformed) data at the bit-counts chosen by the "
             "allocator. This is the 'our method' Lloyd-codebook path.")
        .def("set_derive_codebooks_exact",
             [](IVF &self, size_t max_bits) { self.set_derive_codebooks_exact(max_bits); },
             py::arg("max_bits") = 13,
             "Like set_derive_codebooks() but derives EXACT (globally optimal) "
             "per-dimension codebooks (divide-and-conquer DP) instead of Lloyd. "
             "Exact, parameter-free (no restarts), faster than Lloyd.")
        .def_property_readonly("has_codebooks", &IVF::has_codebooks);

    // ================================================================
    //  Approximation-quality primitives (professor's experiments #1-#2):
    //  per-dim codebook builders (DP-optimal vs cumsum-kmeans) and the
    //  joint bit allocators (DP-Bennett vs empirical greedy). Bound as
    //  free functions so experiments drive them per-column from Python.
    // ================================================================

    // ---- DimensionCodebook ----
    py::class_<DimensionCodebook>(m, "DimensionCodebook")
        .def_property_readonly("num_entries",
            [](const DimensionCodebook &c) { return c.num_entries; })
        .def_property_readonly("centroids",
            [](const DimensionCodebook &c) {
                return py::array_t<float>(
                    static_cast<py::ssize_t>(c.centroids.size()), c.centroids.data());
            })
        .def("nearest", &DimensionCodebook::nearest, py::arg("value"),
             "Nearest centroid index for a value (binary search).");

    // ---- CodebookResult ----
    py::class_<CodebookResult>(m, "CodebookResult")
        .def_property_readonly("costs",
            [](const CodebookResult &r) {
                return py::array_t<float>(
                    static_cast<py::ssize_t>(r.costs.size()), r.costs.data());
            },
            "Per-bit reconstruction MSE [0..max_bits] (histogram-approximate for DP).")
        .def_property_readonly("codebooks",
            [](const CodebookResult &r) {
                py::list out;
                for (const auto &cb : r.codebooks) out.append(cb);
                return out;
            },
            "Per-bit DimensionCodebook [0..max_bits].");

    // ---- Lloyd options ----
    py::enum_<CodebookInit>(m, "CodebookInit")
        .value("EqualMassQuantile", CodebookInit::EqualMassQuantile)
        .value("UniformSpaced", CodebookInit::UniformSpaced)
        .value("KMeansPlusPlus", CodebookInit::KMeansPlusPlus)
        .value("CubeRootDensity", CodebookInit::CubeRootDensity)
        .export_values();

    py::class_<LloydOpts>(m, "LloydOpts")
        .def(py::init<>())
        .def_readwrite("max_bits", &LloydOpts::max_bits)
        .def_readwrite("init", &LloydOpts::init)
        .def_readwrite("restarts", &LloydOpts::restarts)
        .def_readwrite("max_iters", &LloydOpts::max_iters)
        .def_readwrite("tol", &LloydOpts::tol)
        .def_readwrite("seed", &LloydOpts::seed)
        .def_readwrite("sample_size", &LloydOpts::sample_size);

    // ---- Codebook builders (free functions over a 1-D column) ----
    m.def("build_codebook_dp",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> values,
             size_t max_bits, size_t num_bins) {
              py::buffer_info buf = values.request();
              if (buf.ndim != 1) throw std::runtime_error("values must be 1D");
              std::span<const float> sp(static_cast<const float *>(buf.ptr),
                                        static_cast<size_t>(buf.shape[0]));
              return build_codebook_dp(sp, max_bits, num_bins);
          },
          py::arg("values"), py::arg("max_bits") = 8, py::arg("num_bins") = 500,
          "DP-optimal contiguous 1-D clustering (the reference). "
          "Returns CodebookResult(costs, codebooks). Valid for max_bits <= 8.");

    m.def("build_codebook_lloyd",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> values,
             const LloydOpts &opts) {
              py::buffer_info buf = values.request();
              if (buf.ndim != 1) throw std::runtime_error("values must be 1D");
              std::span<const float> sp(static_cast<const float *>(buf.ptr),
                                        static_cast<size_t>(buf.shape[0]));
              return build_codebook_lloyd(sp, opts);
          },
          py::arg("values"), py::arg("opts") = LloydOpts(),
          "Fast Lloyd (cumsum k-means) codebook over a 1-D column. "
          "Returns CodebookResult(costs, codebooks).");

    m.def("build_codebook_exact",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> values,
             size_t max_bits) {
              py::buffer_info buf = values.request();
              if (buf.ndim != 1) throw std::runtime_error("values must be 1D");
              std::span<const float> sp(static_cast<const float *>(buf.ptr),
                                        static_cast<size_t>(buf.shape[0]));
              return build_codebook_exact(sp, max_bits);
          },
          py::arg("values"), py::arg("max_bits") = 8,
          "Exact (globally optimal) 1-D k-means over raw sorted values, no binning "
          "(divide-and-conquer DP, O(k n log n); all bit-rates in one pass). "
          "Returns CodebookResult(costs, codebooks).");

    m.def("codebook_mse",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> values,
             const DimensionCodebook &cb) {
              py::buffer_info buf = values.request();
              if (buf.ndim != 1) throw std::runtime_error("values must be 1D");
              std::span<const float> sp(static_cast<const float *>(buf.ptr),
                                        static_cast<size_t>(buf.shape[0]));
              return codebook_mse(sp, cb);
          },
          py::arg("values"), py::arg("codebook"),
          "Exact MSE of raw values under a codebook (nearest-centroid). "
          "Use to re-score DP/Lloyd codebooks on raw data for a fair comparison.");

    // ---- Joint allocation config + result ----
    py::class_<JointAllocationConfig>(m, "JointAllocationConfig")
        .def(py::init<>())
        .def_readwrite("total_bits", &JointAllocationConfig::total_bits)
        .def_readwrite("max_bits_per_dim", &JointAllocationConfig::max_bits_per_dim)
        .def_readwrite("dim_padding_size", &JointAllocationConfig::dim_padding_size)
        .def_readwrite("num_dim_padded", &JointAllocationConfig::num_dim_padded)
        .def_readwrite("num_bit_factors", &JointAllocationConfig::num_bit_factors);

    py::class_<BitAllocationResult>(m, "BitAllocationResult")
        .def_property_readonly("quant_plan",
            [](const BitAllocationResult &r) {
                py::list out;
                for (const auto &p : r.quant_plan)
                    out.append(py::make_tuple(p.first, p.second));
                return out;
            },
            "List of (dim_length, bits) segments, contiguous.")
        .def_readonly("total_bits_used", &BitAllocationResult::total_bits_used)
        .def_readonly("total_distortion", &BitAllocationResult::total_distortion)
        .def_readonly("error", &BitAllocationResult::error)
        .def("ok", &BitAllocationResult::ok);

    // ---- Allocators (free functions) ----
    m.def("allocate_dp",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> variance,
             const JointAllocationConfig &cfg) {
              py::buffer_info buf = variance.request();
              if (buf.ndim != 1) throw std::runtime_error("variance must be 1D");
              FloatVec v = Eigen::Map<const FloatVec>(
                  static_cast<const float *>(buf.ptr),
                  static_cast<Eigen::Index>(buf.shape[0]));
              return BitAllocatorDP().AllocateJoint(v, cfg);
          },
          py::arg("variance"), py::arg("config"),
          "DP joint segmentation+allocation under the analytic Bennett cost "
          "model (var/2^b). variance is per-dim, length == config.num_dim_padded.");

    m.def("allocate_greedy",
          [](Eigen::MatrixXf mse_table, const JointAllocationConfig &cfg) {
              return BitAllocatorGreedy().AllocateJoint(mse_table, cfg);
          },
          py::arg("mse_table"), py::arg("config"),
          "Greedy joint allocation using an empirical "
          "(num_dim_padded, max_bits_per_dim+1) per-dim per-bit MSE table. "
          "Row b=0 should hold per-dim variance (the no-quantization cost).");

    // ---- Utility functions ----
    m.def("load_fvecs",
          [](const std::string &filename) {
              FloatRowMat mat;
              load_something<float, FloatRowMat>(filename.c_str(), mat);
              return mat;
          },
          py::arg("filename"),
          "Load a .fvecs file. Returns float32 array of shape (n, dim).");

    m.def("load_ivecs",
          [](const std::string &filename) {
              UintRowMat mat;
              load_something<uint32_t, UintRowMat>(filename.c_str(), mat);
              return mat;
          },
          py::arg("filename"),
          "Load a .ivecs file. Returns uint32 array of shape (n, dim).");
}
