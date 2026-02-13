/// @file saq_quantizer.cpp
/// @brief Implementation of the main SAQ quantizer using scalar quantization.
///
/// Implements the SAQ paper (arXiv:2509.12086): PCA projection,
/// joint dimension segmentation + bit allocation via DP,
/// per-segment random orthonormal rotation, scalar (uniform grid)
/// quantization, and asymmetric inner product estimation.

#include "saq/saq_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#ifdef SAQ_USE_OPENMP
#include <omp.h>
#endif

namespace saq {

namespace {

/// @brief Generate a random orthonormal matrix via modified Gram-Schmidt.
/// @param dim Matrix dimension (dim x dim).
/// @param rng Random number generator.
/// @return Row-major orthonormal matrix of size dim*dim.
std::vector<float> GenerateOrthonormalMatrix(uint32_t dim, std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> matrix(static_cast<size_t>(dim) * dim);

  // Fill with random normal values
  for (auto& val : matrix) {
    val = normal(rng);
  }

  // Modified Gram-Schmidt orthonormalization (row-major)
  for (uint32_t i = 0; i < dim; ++i) {
    float* row_i = matrix.data() + static_cast<size_t>(i) * dim;

    // Subtract projections onto all previous rows
    for (uint32_t j = 0; j < i; ++j) {
      const float* row_j = matrix.data() + static_cast<size_t>(j) * dim;

      float dot = 0.0f;
      for (uint32_t d = 0; d < dim; ++d) {
        dot += row_i[d] * row_j[d];
      }

      for (uint32_t d = 0; d < dim; ++d) {
        row_i[d] -= dot * row_j[d];
      }
    }

    // Normalize
    float norm_sq = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
      norm_sq += row_i[d] * row_i[d];
    }
    float norm = std::sqrt(norm_sq);

    if (norm > 1e-10f) {
      float inv_norm = 1.0f / norm;
      for (uint32_t d = 0; d < dim; ++d) {
        row_i[d] *= inv_norm;
      }
    }
  }

  return matrix;
}

}  // namespace

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

std::string SAQQuantizer::Train(const float* data, uint32_t n_vectors,
                                uint32_t dim, const SAQTrainConfig& config) {
  if (data == nullptr || n_vectors == 0 || dim == 0) {
    return "Invalid training data";
  }

  if (config.total_bits == 0) {
    return "Total bits must be positive";
  }

  trained_ = false;
  plan_ = QuantizationPlan{};
  plan_.version = 2;
  plan_.dimension = dim;
  plan_.total_bits = config.total_bits;
  plan_.seed = config.seed;
  plan_.use_pca = config.use_pca;

  // Working data (potentially PCA-transformed)
  std::vector<float> work_data;
  const float* work_ptr = data;
  uint32_t work_dim = dim;

  // Step 1: Optional PCA projection (full rotation when pca_dim == dim,
  //         dimensionality reduction when pca_dim < dim).
  //         The paper uses PCA to order dimensions by eigenvalue for
  //         effective DP bit allocation, even without reducing dim.
  if (config.use_pca && config.pca_dim > 0) {
    if (!pca_.Train(data, n_vectors, dim, config.pca_dim)) {
      return "PCA training failed";
    }

    work_dim = config.pca_dim;
    work_data.resize(static_cast<size_t>(n_vectors) * work_dim);
    pca_.ProjectBatch(data, work_data.data(), n_vectors);
    work_ptr = work_data.data();
    pca_.ExportParams(&plan_.pca);
  }

  // Step 2: Compute per-dimension variances for the DP
  DimensionSegmenter segmenter;
  if (!segmenter.ComputeStats(work_ptr, n_vectors, work_dim)) {
    return "Failed to compute dimension statistics";
  }

  const auto& stats = segmenter.GetStats();
  std::vector<float> dim_variances(work_dim);
  for (uint32_t d = 0; d < work_dim; ++d) {
    dim_variances[d] = stats[d].variance;
  }

  // Step 3: Joint segmentation + bit allocation via DP (Algorithm 2)
  BitAllocatorDP allocator;
  JointAllocationConfig joint_config;
  joint_config.total_bits = config.total_bits;
  joint_config.min_bits_per_dim = config.min_bits_per_dim;
  joint_config.max_bits_per_dim = config.max_bits_per_dim;
  joint_config.min_dims_per_segment = config.min_dims_per_segment;
  joint_config.max_dims_per_segment = config.max_dims_per_segment;

  auto result = allocator.AllocateJoint(dim_variances, joint_config);
  if (!result.IsValid()) {
    return "Joint segmentation/allocation failed: " + result.error;
  }

  plan_.segments = result.segments;
  plan_.segment_count = static_cast<uint32_t>(plan_.segments.size());

  // Step 4: Generate per-segment rotation matrices (optional)
  if (config.use_segment_rotation) {
    std::string rot_error = GenerateRotations(config.seed);
    if (!rot_error.empty()) {
      return rot_error;
    }
  }

  // Step 5: Initialize CAQ adjuster
  if (!caq_adjuster_.Initialize(plan_.segments)) {
    return "Failed to initialize CAQ adjuster";
  }

  trained_ = true;
  return "";
}

// ---------------------------------------------------------------------------
// Plan loading
// ---------------------------------------------------------------------------

std::string SAQQuantizer::LoadPlan(const QuantizationPlan& plan) {
  plan_ = plan;
  trained_ = false;

  // Import PCA if enabled
  if (plan_.use_pca) {
    if (!pca_.ImportParams(plan_.pca)) {
      return "Failed to import PCA parameters";
    }
  }

  // Initialize CAQ adjuster
  if (!plan_.segments.empty()) {
    caq_adjuster_.Initialize(plan_.segments);
  }

  trained_ = true;
  return "";
}

// ---------------------------------------------------------------------------
// Per-segment rotation
// ---------------------------------------------------------------------------

std::string SAQQuantizer::GenerateRotations(uint32_t seed) {
  plan_.rotations.clear();
  plan_.rotations.reserve(plan_.segments.size());

  std::mt19937 rng(seed);

  for (const auto& seg : plan_.segments) {
    SegmentRotation rot;
    rot.segment_id = seg.id;
    rot.dim_count = seg.dim_count;

    if (seg.dim_count <= 1) {
      // Identity for 1-D segments
      rot.matrix = {1.0f};
    } else {
      rot.matrix = GenerateOrthonormalMatrix(seg.dim_count, rng);
    }

    plan_.rotations.push_back(std::move(rot));
  }

  return "";
}

void SAQQuantizer::ApplyRotation(const float* input, float* output) const {
  if (plan_.rotations.empty()) {
    std::memcpy(output, input, WorkingDim() * sizeof(float));
    return;
  }

  for (size_t s = 0; s < plan_.segments.size(); ++s) {
    const auto& seg = plan_.segments[s];
    const auto& rot = plan_.rotations[s];
    const float* seg_in = input + seg.start_dim;
    float* seg_out = output + seg.start_dim;

    // output = R * input  (R is row-major)
    for (uint32_t i = 0; i < seg.dim_count; ++i) {
      float sum = 0.0f;
      const float* row = rot.matrix.data() +
                          static_cast<size_t>(i) * seg.dim_count;
      for (uint32_t j = 0; j < seg.dim_count; ++j) {
        sum += row[j] * seg_in[j];
      }
      seg_out[i] = sum;
    }
  }
}

void SAQQuantizer::ApplyInverseRotation(const float* input,
                                        float* output) const {
  if (plan_.rotations.empty()) {
    std::memcpy(output, input, WorkingDim() * sizeof(float));
    return;
  }

  for (size_t s = 0; s < plan_.segments.size(); ++s) {
    const auto& seg = plan_.segments[s];
    const auto& rot = plan_.rotations[s];
    const float* seg_in = input + seg.start_dim;
    float* seg_out = output + seg.start_dim;

    // Inverse of orthonormal R is R^T: output = R^T * input
    for (uint32_t i = 0; i < seg.dim_count; ++i) {
      float sum = 0.0f;
      for (uint32_t j = 0; j < seg.dim_count; ++j) {
        sum += rot.matrix[static_cast<size_t>(j) * seg.dim_count + i] *
               seg_in[j];
      }
      seg_out[i] = sum;
    }
  }
}

// ---------------------------------------------------------------------------
// Scalar quantization / dequantization
// ---------------------------------------------------------------------------

void SAQQuantizer::ScalarQuantize(const float* rotated, uint32_t working_dim,
                                  ScalarEncodedVector& encoded) const {
  encoded.codes.resize(working_dim);

  // v_max = max |rotated[i]| across all dimensions
  float v_max = 0.0f;
  for (uint32_t i = 0; i < working_dim; ++i) {
    v_max = std::max(v_max, std::abs(rotated[i]));
  }

  // Guard against zero vectors
  if (v_max < 1e-30f) {
    v_max = 1e-30f;
  }
  encoded.v_max = v_max;

  // Quantize each dimension: c[i] = floor((v[i] + v_max) / delta)
  for (const auto& seg : plan_.segments) {
    if (seg.bits == 0) {
      for (uint32_t d = 0; d < seg.dim_count; ++d) {
        encoded.codes[seg.start_dim + d] = 0;
      }
      continue;
    }

    uint32_t levels = 1u << seg.bits;  // 2^B
    float delta = (2.0f * v_max) / static_cast<float>(levels);
    float inv_delta = 1.0f / delta;

    for (uint32_t d = 0; d < seg.dim_count; ++d) {
      uint32_t idx = seg.start_dim + d;
      auto code = static_cast<int32_t>(
          std::floor((rotated[idx] + v_max) * inv_delta));
      code = std::max(0, std::min(code, static_cast<int32_t>(levels - 1)));
      encoded.codes[idx] = static_cast<uint8_t>(code);
    }
  }
}

void SAQQuantizer::ScalarDequantize(const ScalarEncodedVector& encoded,
                                    float* rotated) const {
  float v_max = encoded.v_max;

  for (const auto& seg : plan_.segments) {
    if (seg.bits == 0) {
      for (uint32_t d = 0; d < seg.dim_count; ++d) {
        rotated[seg.start_dim + d] = 0.0f;
      }
      continue;
    }

    uint32_t levels = 1u << seg.bits;
    float delta = (2.0f * v_max) / static_cast<float>(levels);

    for (uint32_t d = 0; d < seg.dim_count; ++d) {
      uint32_t idx = seg.start_dim + d;
      // Reconstruction: o_bar[i] = delta * (c[i] + 0.5) - v_max
      rotated[idx] =
          delta * (static_cast<float>(encoded.codes[idx]) + 0.5f) - v_max;
    }
  }
}

// ---------------------------------------------------------------------------
// Encoding / Decoding
// ---------------------------------------------------------------------------

bool SAQQuantizer::Encode(const float* vector, ScalarEncodedVector& encoded,
                          const SAQEncodeConfig& config) const {
  if (!trained_ || vector == nullptr) {
    return false;
  }

  uint32_t work_dim = WorkingDim();

  // Step 1: PCA projection
  std::vector<float> projected;
  const float* work_vec = vector;
  if (plan_.use_pca) {
    projected.resize(plan_.pca.output_dim);
    pca_.Project(vector, projected.data());
    work_vec = projected.data();
  }

  // Step 2: Per-segment rotation
  std::vector<float> rotated(work_dim);
  ApplyRotation(work_vec, rotated.data());

  // Step 3: Scalar quantization
  ScalarQuantize(rotated.data(), work_dim, encoded);

  // Step 4: CAQ refinement (Algorithm 1) — maximize cosine similarity
  if (config.use_caq && caq_adjuster_.IsInitialized()) {
    caq_adjuster_.RefineScalar(rotated.data(), encoded.codes.data(),
                               encoded.v_max, config.caq_config);
  }

  return true;
}

bool SAQQuantizer::EncodeBatch(const float* vectors, uint32_t n_vectors,
                               std::vector<ScalarEncodedVector>& encoded,
                               const SAQEncodeConfig& config) const {
  if (!trained_ || vectors == nullptr) {
    return false;
  }

  encoded.resize(n_vectors);

  bool success = true;
#ifdef SAQ_USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (uint32_t v = 0; v < n_vectors; ++v) {
    const float* vec = vectors + static_cast<size_t>(v) * plan_.dimension;
    if (!Encode(vec, encoded[v], config)) {
      success = false;
    }
  }

  return success;
}

bool SAQQuantizer::Decode(const ScalarEncodedVector& encoded,
                          float* vector) const {
  if (!trained_ || vector == nullptr) {
    return false;
  }

  uint32_t work_dim = WorkingDim();

  // Step 1: Dequantize (produces rotated-space vector)
  std::vector<float> rotated(work_dim);
  ScalarDequantize(encoded, rotated.data());

  // Step 2: Inverse per-segment rotation
  std::vector<float> work_vec(work_dim);
  ApplyInverseRotation(rotated.data(), work_vec.data());

  // Step 3: Inverse PCA
  if (plan_.use_pca) {
    pca_.InverseProject(work_vec.data(), vector);
  } else {
    std::memcpy(vector, work_vec.data(), plan_.dimension * sizeof(float));
  }

  return true;
}

// ---------------------------------------------------------------------------
// Distance estimation (SAQ paper formula)
// ---------------------------------------------------------------------------

float SAQQuantizer::EstimateInnerProduct(
    const float* query, const ScalarEncodedVector& encoded) const {
  // Paper formula (operates in rotated space):
  //   <o_bar, q> = delta * <codes, q> + q_sum * (-v_max + delta/2)
  //
  // Where delta = 2*v_max / 2^B per segment.
  // The query must already be in rotated space.

  float v_max = encoded.v_max;
  float total_ip = 0.0f;

  for (const auto& seg : plan_.segments) {
    if (seg.bits == 0) {
      continue;
    }

    uint32_t levels = 1u << seg.bits;
    float delta = (2.0f * v_max) / static_cast<float>(levels);

    float code_dot_q = 0.0f;
    float q_sum = 0.0f;

    for (uint32_t d = 0; d < seg.dim_count; ++d) {
      uint32_t idx = seg.start_dim + d;
      float q_val = query[idx];
      code_dot_q += static_cast<float>(encoded.codes[idx]) * q_val;
      q_sum += q_val;
    }

    total_ip += delta * code_dot_q + q_sum * (-v_max + delta * 0.5f);
  }

  return total_ip;
}

// ---------------------------------------------------------------------------
// Query transformation
// ---------------------------------------------------------------------------

bool SAQQuantizer::TransformQuery(const float* query,
                                  float* rotated_query) const {
  if (!trained_ || query == nullptr || rotated_query == nullptr) {
    return false;
  }

  uint32_t work_dim = WorkingDim();

  // PCA projection
  std::vector<float> projected;
  const float* work_vec = query;
  if (plan_.use_pca) {
    projected.resize(plan_.pca.output_dim);
    pca_.Project(query, projected.data());
    work_vec = projected.data();
  }

  // Per-segment rotation
  ApplyRotation(work_vec, rotated_query);
  return true;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

void SAQQuantizer::Search(
    const float* query,
    const std::vector<ScalarEncodedVector>& encoded_db, uint32_t k,
    std::vector<SearchResult>& results) const {
  results.clear();

  if (!trained_ || query == nullptr || encoded_db.empty()) {
    return;
  }

  auto n_vectors = static_cast<uint32_t>(encoded_db.size());
  k = std::min(k, n_vectors);

  uint32_t work_dim = WorkingDim();

  // Transform query: PCA then rotation
  std::vector<float> projected;
  const float* work_query = query;
  if (plan_.use_pca) {
    projected.resize(plan_.pca.output_dim);
    pca_.Project(query, projected.data());
    work_query = projected.data();
  }

  std::vector<float> rotated_query(work_dim);
  ApplyRotation(work_query, rotated_query.data());

  // Precompute ||q_rotated||² (rotation preserves norms)
  float query_norm_sq = 0.0f;
  for (uint32_t d = 0; d < work_dim; ++d) {
    query_norm_sq += rotated_query[d] * rotated_query[d];
  }

  // Max-heap of (distance, index) — keeps k smallest L2 distances.
  using Entry = std::pair<float, uint32_t>;
  std::priority_queue<Entry> top_k;

  for (uint32_t v = 0; v < n_vectors; ++v) {
    const auto& enc = encoded_db[v];
    float ip = EstimateInnerProduct(rotated_query.data(), enc);

    // Compute ||ō||² from codes (same approach as IVF index)
    float norm_sq = 0.0f;
    for (const auto& seg : plan_.segments) {
      if (seg.bits == 0) continue;
      uint32_t levels = 1u << seg.bits;
      float delta = (2.0f * enc.v_max) / static_cast<float>(levels);
      for (uint32_t d = 0; d < seg.dim_count; ++d) {
        uint32_t idx = seg.start_dim + d;
        float recon = delta * (static_cast<float>(enc.codes[idx]) + 0.5f)
                      - enc.v_max;
        norm_sq += recon * recon;
      }
    }

    // L2: ||q - ō||² = ||q||² - 2<q,ō> + ||ō||²
    float dist = query_norm_sq - 2.0f * ip + norm_sq;

    if (top_k.size() < k) {
      top_k.emplace(dist, v);
    } else if (dist < top_k.top().first) {
      top_k.pop();
      top_k.emplace(dist, v);
    }
  }

  // Extract results sorted by distance (ascending)
  results.resize(top_k.size());
  for (size_t i = results.size(); i > 0; --i) {
    results[i - 1].index = top_k.top().second;
    results[i - 1].distance = top_k.top().first;
    top_k.pop();
  }
}

}  // namespace saq
