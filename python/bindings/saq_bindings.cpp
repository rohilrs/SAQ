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
#include "saq/config.h"
#include "saq/defines.h"
#include "saq/io_utils.h"

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
        .def_property_readonly("has_codebooks", &IVF::has_codebooks);

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
