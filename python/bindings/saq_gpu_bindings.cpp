/// @file saq_gpu_bindings.cpp
/// @brief Python bindings for the SAQ GPU library using pybind11.

#ifdef SAQ_USE_CUDA

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "saq/codebook_encoder.h"
#include "saq/gpu/gpu_ivf.h"
#include "saq/config.h"
#include "saq/defines.h"

namespace py = pybind11;
using namespace saq;

PYBIND11_MODULE(_saq_gpu, m) {
    m.doc() = "SAQ GPU: CUDA-accelerated Scalar Additive Quantization";

    // ---- GpuIVF ----
    py::class_<gpu::GpuIVF>(m, "GpuIVF")
        .def(py::init<size_t, size_t, size_t, QuantizeConfig>(),
             py::arg("n"), py::arg("dim"), py::arg("k"), py::arg("config"),
             "Create GPU IVF index. Args: num_vectors, dimension, num_clusters, config.")
        .def("set_variance",
             [](gpu::GpuIVF &self, py::array_t<float, py::array::c_style> variances) {
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
             [](gpu::GpuIVF &self, Eigen::Ref<const FloatRowMat> data,
                Eigen::Ref<const FloatRowMat> centroids,
                py::array_t<uint32_t, py::array::c_style> cluster_ids) {
                 py::buffer_info ids_buf = cluster_ids.request();
                 if (ids_buf.ndim == 2) {
                     if (ids_buf.shape[1] != 1) {
                         throw std::runtime_error("cluster_ids must be 1D or (N,1)");
                     }
                 }
                 const PID *ids_ptr = static_cast<const PID *>(ids_buf.ptr);
                 {
                     py::gil_scoped_release release;
                     self.construct(data, centroids, ids_ptr);
                 }
             },
             py::arg("data"), py::arg("centroids"), py::arg("cluster_ids"),
             "Build GPU IVF index from data, centroids, and cluster assignments.")
        .def("search_batch",
             [](gpu::GpuIVF &self, Eigen::Ref<const FloatRowMat> queries,
                size_t topk, size_t nprobe, SearcherConfig searcher_cfg,
                bool return_dists) -> py::object {
                 size_t nq = static_cast<size_t>(queries.rows());
                 py::array_t<uint32_t> results({static_cast<py::ssize_t>(nq),
                                                 static_cast<py::ssize_t>(topk)});
                 auto *results_ptr = static_cast<PID *>(results.mutable_data());
                 if (!return_dists) {
                     {
                         py::gil_scoped_release release;
                         self.search_batch(queries, topk, nprobe, searcher_cfg, results_ptr);
                     }
                     return std::move(results);
                 }
                 py::array_t<float> dists({static_cast<py::ssize_t>(nq),
                                           static_cast<py::ssize_t>(topk)});
                 auto *dists_ptr = static_cast<float *>(dists.mutable_data());
                 {
                     py::gil_scoped_release release;
                     self.search_batch(queries, topk, nprobe, searcher_cfg, results_ptr, dists_ptr);
                 }
                 return py::make_tuple(std::move(results), std::move(dists));
             },
             py::arg("queries"), py::arg("topk"), py::arg("nprobe"),
             py::arg("config") = SearcherConfig(),
             py::arg("return_dists") = false,
             "GPU batch search. Returns uint32 (nq,topk) IDs; if return_dists=True "
             "returns (ids, dists) where dists are the top-k ADC distances.")
        .def_property_readonly("num_data", &gpu::GpuIVF::num_data)
        .def_property_readonly("num_dim", &gpu::GpuIVF::num_dim)
        .def_property_readonly("k", &gpu::GpuIVF::k)
        .def("set_gaussian_codebooks",
             [](gpu::GpuIVF &self, py::dict codebooks_dict,
                py::array_t<float, py::array::c_style> variances) {
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
             "Set Gaussian base codebooks for GPU index. "
             "codebooks: dict {bits: 1D float array with 2^bits entries}, variances: 1D float array.")
        .def("set_codebooks",
             [](gpu::GpuIVF &self, py::list codebooks_list) {
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
             "Set per-segment, per-dimension codebooks for GPU index.")
        .def_property_readonly("has_codebooks", &gpu::GpuIVF::has_codebooks);
}

#endif // SAQ_USE_CUDA
