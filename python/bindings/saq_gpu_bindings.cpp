/// @file saq_gpu_bindings.cpp
/// @brief Python bindings for the SAQ GPU library using pybind11.

#ifdef SAQ_USE_CUDA

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
                size_t topk, size_t nprobe, SearcherConfig searcher_cfg) {
                 size_t nq = static_cast<size_t>(queries.rows());
                 py::array_t<uint32_t> results({static_cast<py::ssize_t>(nq),
                                                 static_cast<py::ssize_t>(topk)});
                 auto *results_ptr = static_cast<PID *>(results.mutable_data());
                 {
                     py::gil_scoped_release release;
                     self.search_batch(queries, topk, nprobe, searcher_cfg, results_ptr);
                 }
                 return results;
             },
             py::arg("queries"), py::arg("topk"), py::arg("nprobe"),
             py::arg("config") = SearcherConfig(),
             "GPU batch search. Returns uint32 array of shape (nq, topk).")
        .def_property_readonly("num_data", &gpu::GpuIVF::num_data)
        .def_property_readonly("num_dim", &gpu::GpuIVF::num_dim)
        .def_property_readonly("k", &gpu::GpuIVF::k);
}

#endif // SAQ_USE_CUDA
