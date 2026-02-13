/// @file saq_bindings.cpp
/// @brief Python bindings for the SAQ library using pybind11.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "index/ivf_index.h"
#include "saq/saq_quantizer.h"

namespace py = pybind11;
using namespace saq;

// Helper: validate contiguous float32 array with expected shape.
static const float* CheckFloat32(const py::array_t<float>& arr,
                                  const std::string& name) {
  if (!(arr.flags() & py::array::c_style)) {
    throw std::runtime_error(name + " must be C-contiguous float32");
  }
  return arr.data();
}

PYBIND11_MODULE(_saq_core, m) {
  m.doc() = "SAQ: Scalar Additive Quantization (C++ core)";

  // ---- Enums ----
  py::enum_<DistanceMetric>(m, "DistanceMetric")
      .value("L2", DistanceMetric::kL2)
      .value("InnerProduct", DistanceMetric::kInnerProduct)
      .export_values();

  // ---- SAQTrainConfig ----
  py::class_<SAQTrainConfig>(m, "SAQTrainConfig")
      .def(py::init<>())
      .def_readwrite("total_bits", &SAQTrainConfig::total_bits)
      .def_readwrite("use_pca", &SAQTrainConfig::use_pca)
      .def_readwrite("pca_dim", &SAQTrainConfig::pca_dim)
      .def_readwrite("seed", &SAQTrainConfig::seed)
      .def_readwrite("max_bits_per_dim", &SAQTrainConfig::max_bits_per_dim)
      .def_readwrite("min_bits_per_dim", &SAQTrainConfig::min_bits_per_dim)
      .def_readwrite("min_dims_per_segment", &SAQTrainConfig::min_dims_per_segment)
      .def_readwrite("max_dims_per_segment", &SAQTrainConfig::max_dims_per_segment)
      .def_readwrite("metric", &SAQTrainConfig::metric)
      .def_readwrite("use_segment_rotation", &SAQTrainConfig::use_segment_rotation);

  // ---- SAQEncodeConfig ----
  py::class_<SAQEncodeConfig>(m, "SAQEncodeConfig")
      .def(py::init<>())
      .def_readwrite("use_caq", &SAQEncodeConfig::use_caq);

  // ---- IVFConfig ----
  py::class_<IVFConfig>(m, "IVFConfig")
      .def(py::init<>())
      .def_readwrite("num_clusters", &IVFConfig::num_clusters)
      .def_readwrite("nprobe", &IVFConfig::nprobe)
      .def_readwrite("max_vectors_per_cluster", &IVFConfig::max_vectors_per_cluster)
      .def_readwrite("use_hnsw_initializer", &IVFConfig::use_hnsw_initializer)
      .def_readwrite("hnsw_m", &IVFConfig::hnsw_m)
      .def_readwrite("hnsw_ef_construction", &IVFConfig::hnsw_ef_construction)
      .def_readwrite("metric", &IVFConfig::metric);

  // ---- IVFTrainConfig ----
  py::class_<IVFTrainConfig>(m, "IVFTrainConfig")
      .def(py::init<>())
      .def_readwrite("ivf", &IVFTrainConfig::ivf)
      .def_readwrite("saq", &IVFTrainConfig::saq)
      .def_readwrite("seed", &IVFTrainConfig::seed);

  // ---- SAQQuantizer ----
  py::class_<SAQQuantizer>(m, "SAQQuantizer")
      .def(py::init<>())
      .def(
          "train",
          [](SAQQuantizer& self, py::array_t<float> data,
             const SAQTrainConfig& config) {
            py::buffer_info buf = data.request();
            if (buf.ndim != 2) {
              throw std::runtime_error("data must be 2D (n_vectors x dim)");
            }
            auto n = static_cast<uint32_t>(buf.shape[0]);
            auto dim = static_cast<uint32_t>(buf.shape[1]);
            const float* ptr = CheckFloat32(data, "data");
            std::string err;
            {
              py::gil_scoped_release release;
              err = self.Train(ptr, n, dim, config);
            }
            if (!err.empty()) {
              throw std::runtime_error(err);
            }
          },
          py::arg("data"), py::arg("config") = SAQTrainConfig(),
          "Train the quantizer on a dataset (n_vectors x dim).")
      .def(
          "encode_batch",
          [](const SAQQuantizer& self, py::array_t<float> vectors,
             const SAQEncodeConfig& config) {
            py::buffer_info buf = vectors.request();
            if (buf.ndim != 2) {
              throw std::runtime_error("vectors must be 2D");
            }
            auto n = static_cast<uint32_t>(buf.shape[0]);
            const float* ptr = CheckFloat32(vectors, "vectors");
            std::vector<ScalarEncodedVector> encoded;
            bool ok;
            {
              py::gil_scoped_release release;
              ok = self.EncodeBatch(ptr, n, encoded, config);
            }
            if (!ok) {
              throw std::runtime_error("EncodeBatch failed");
            }
            // Return codes as (n x working_dim) uint8 array + v_maxs as float array
            uint32_t wd = self.WorkingDim();
            py::array_t<uint8_t> codes_out({(py::ssize_t)n, (py::ssize_t)wd});
            py::array_t<float> vmaxs_out(n);
            auto codes_ptr = codes_out.mutable_data();
            auto vmaxs_ptr = vmaxs_out.mutable_data();
            for (uint32_t i = 0; i < n; ++i) {
              std::memcpy(codes_ptr + i * wd, encoded[i].codes.data(), wd);
              vmaxs_ptr[i] = encoded[i].v_max;
            }
            return py::make_tuple(codes_out, vmaxs_out);
          },
          py::arg("vectors"), py::arg("config") = SAQEncodeConfig(),
          "Encode vectors. Returns (codes: uint8[n,wd], v_maxs: float[n]).")
      .def(
          "decode",
          [](const SAQQuantizer& self, py::array_t<uint8_t> codes,
             float v_max) {
            auto wd = self.WorkingDim();
            auto dim = self.Dim();
            if (static_cast<uint32_t>(codes.size()) != wd) {
              throw std::runtime_error("codes size must match working_dim");
            }
            ScalarEncodedVector enc;
            enc.codes.assign(codes.data(), codes.data() + wd);
            enc.v_max = v_max;
            py::array_t<float> out(dim);
            bool ok = self.Decode(enc, out.mutable_data());
            if (!ok) {
              throw std::runtime_error("Decode failed");
            }
            return out;
          },
          py::arg("codes"), py::arg("v_max"),
          "Decode a single encoded vector back to float.")
      .def(
          "search",
          [](const SAQQuantizer& self, py::array_t<float> query,
             py::array_t<uint8_t> db_codes, py::array_t<float> db_vmaxs,
             uint32_t k) {
            auto dim = self.Dim();
            auto wd = self.WorkingDim();
            if (static_cast<uint32_t>(query.size()) != dim) {
              throw std::runtime_error("query size must match dim");
            }
            py::buffer_info codes_buf = db_codes.request();
            if (codes_buf.ndim != 2) {
              throw std::runtime_error("db_codes must be 2D");
            }
            auto n = static_cast<uint32_t>(codes_buf.shape[0]);
            // Build encoded vector list
            std::vector<ScalarEncodedVector> db(n);
            const auto* codes_ptr = db_codes.data();
            const auto* vmaxs_ptr = db_vmaxs.data();
            for (uint32_t i = 0; i < n; ++i) {
              db[i].codes.assign(codes_ptr + i * wd,
                                 codes_ptr + (i + 1) * wd);
              db[i].v_max = vmaxs_ptr[i];
            }
            std::vector<SearchResult> results;
            {
              py::gil_scoped_release release;
              self.Search(query.data(), db, k, results);
            }
            // Return (indices, distances)
            py::array_t<uint32_t> indices(results.size());
            py::array_t<float> distances(results.size());
            auto* idx_ptr = indices.mutable_data();
            auto* dist_ptr = distances.mutable_data();
            for (size_t i = 0; i < results.size(); ++i) {
              idx_ptr[i] = results[i].index;
              dist_ptr[i] = results[i].distance;
            }
            return py::make_tuple(indices, distances);
          },
          py::arg("query"), py::arg("db_codes"), py::arg("db_vmaxs"),
          py::arg("k"),
          "Search for k nearest neighbors. Returns (indices, distances).")
      .def_property_readonly("dim", &SAQQuantizer::Dim)
      .def_property_readonly("working_dim", &SAQQuantizer::WorkingDim)
      .def_property_readonly("num_segments", &SAQQuantizer::NumSegments)
      .def_property_readonly("total_bits", &SAQQuantizer::TotalBits)
      .def_property_readonly("is_trained", &SAQQuantizer::IsTrained);

  // ---- IVFIndex ----
  py::class_<IVFIndex>(m, "IVFIndex")
      .def(py::init<>())
      .def(
          "build",
          [](IVFIndex& self, py::array_t<float> data,
             py::array_t<float> centroids,
             py::array_t<uint32_t> cluster_ids,
             const IVFTrainConfig& config) {
            py::buffer_info dbuf = data.request();
            py::buffer_info cbuf = centroids.request();
            if (dbuf.ndim != 2 || cbuf.ndim != 2) {
              throw std::runtime_error("data and centroids must be 2D");
            }
            auto n = static_cast<uint32_t>(dbuf.shape[0]);
            auto dim = static_cast<uint32_t>(dbuf.shape[1]);
            const float* dptr = CheckFloat32(data, "data");
            const float* cptr = CheckFloat32(centroids, "centroids");
            const uint32_t* iptr = cluster_ids.data();
            std::string err;
            {
              py::gil_scoped_release release;
              err = self.Build(dptr, n, dim, cptr, iptr, config);
            }
            if (!err.empty()) {
              throw std::runtime_error(err);
            }
          },
          py::arg("data"), py::arg("centroids"), py::arg("cluster_ids"),
          py::arg("config") = IVFTrainConfig(),
          "Build IVF index from pre-computed clustering.")
      .def(
          "search",
          [](const IVFIndex& self, py::array_t<float> query, uint32_t k,
             uint32_t nprobe) {
            if (static_cast<uint32_t>(query.size()) != self.Dimension()) {
              throw std::runtime_error("query size must match dimension");
            }
            std::vector<IVFSearchResult> results;
            {
              py::gil_scoped_release release;
              self.Search(query.data(), k, results, nprobe);
            }
            py::array_t<uint32_t> indices(results.size());
            py::array_t<float> distances(results.size());
            for (size_t i = 0; i < results.size(); ++i) {
              indices.mutable_data()[i] = results[i].index;
              distances.mutable_data()[i] = results[i].distance;
            }
            return py::make_tuple(indices, distances);
          },
          py::arg("query"), py::arg("k"), py::arg("nprobe") = 0,
          "Search for k nearest neighbors. Returns (indices, distances).")
      .def(
          "search_batch",
          [](const IVFIndex& self, py::array_t<float> queries, uint32_t k,
             uint32_t nprobe) {
            py::buffer_info buf = queries.request();
            if (buf.ndim != 2) {
              throw std::runtime_error("queries must be 2D");
            }
            auto nq = static_cast<uint32_t>(buf.shape[0]);
            const float* ptr = CheckFloat32(queries, "queries");
            std::vector<std::vector<IVFSearchResult>> results;
            {
              py::gil_scoped_release release;
              self.SearchBatch(ptr, nq, k, results, nprobe);
            }
            // Return (indices: uint32[nq, k], distances: float[nq, k])
            py::array_t<uint32_t> indices({(py::ssize_t)nq, (py::ssize_t)k});
            py::array_t<float> distances({(py::ssize_t)nq, (py::ssize_t)k});
            auto* idx_ptr = indices.mutable_data();
            auto* dist_ptr = distances.mutable_data();
            for (uint32_t q = 0; q < nq; ++q) {
              for (uint32_t i = 0; i < k; ++i) {
                size_t offset = q * k + i;
                if (i < results[q].size()) {
                  idx_ptr[offset] = results[q][i].index;
                  dist_ptr[offset] = results[q][i].distance;
                } else {
                  idx_ptr[offset] = UINT32_MAX;
                  dist_ptr[offset] = std::numeric_limits<float>::max();
                }
              }
            }
            return py::make_tuple(indices, distances);
          },
          py::arg("queries"), py::arg("k"), py::arg("nprobe") = 0,
          "Batch search. Returns (indices[nq,k], distances[nq,k]).")
      .def("save", &IVFIndex::Save, py::arg("filename"),
           "Save index to file.")
      .def(
          "load",
          [](IVFIndex& self, const std::string& filename) {
            std::string err = self.Load(filename);
            if (!err.empty()) {
              throw std::runtime_error(err);
            }
          },
          py::arg("filename"), "Load index from file.")
      .def(
          "reconstruct",
          [](const IVFIndex& self, uint32_t global_id) {
            py::array_t<float> out(self.Dimension());
            bool ok = self.Reconstruct(global_id, out.mutable_data());
            if (!ok) {
              throw std::runtime_error("Reconstruct failed for id " +
                                       std::to_string(global_id));
            }
            return out;
          },
          py::arg("global_id"), "Reconstruct a vector by global ID.")
      .def_property_readonly("num_vectors", &IVFIndex::NumVectors)
      .def_property_readonly("dimension", &IVFIndex::Dimension)
      .def_property_readonly("num_clusters", &IVFIndex::NumClusters)
      .def_property("nprobe", &IVFIndex::DefaultNprobe,
                    &IVFIndex::SetDefaultNprobe)
      .def_property_readonly("is_built", &IVFIndex::IsBuilt);
}
