#pragma once

/// @file saq_quantizer.h
/// @brief Main SAQ quantizer implementing scalar additive quantization.
///
/// Implements the SAQ paper (arXiv:2509.12086): PCA projection,
/// joint dimension segmentation + bit allocation via DP,
/// per-segment rotation, scalar (uniform grid) quantization,
/// and code adjustment for cosine similarity optimization.

#include "saq/bit_allocation_dp.h"
#include "saq/caq_code_adjustment.h"
#include "saq/dimension_segmentation.h"
#include "saq/distance_estimator.h"
#include "saq/pca_projection.h"
#include "saq/quantization_plan.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace saq {

/// @brief Training configuration for SAQ.
struct SAQTrainConfig {
  /// Target bits per vector (total budget across all segments).
  uint32_t total_bits = 64;

  /// Whether to apply PCA before quantization.
  bool use_pca = false;

  /// Target dimension after PCA (0 = no reduction).
  uint32_t pca_dim = 0;

  /// Random seed for reproducibility.
  uint32_t seed = 42;

  /// Maximum bits per dimension in any segment.
  uint32_t max_bits_per_dim = 8;

  /// Minimum bits per dimension in any segment (0 = segment can be skipped).
  uint32_t min_bits_per_dim = 0;

  /// Minimum number of dimensions per segment.
  uint32_t min_dims_per_segment = 1;

  /// Maximum number of dimensions per segment (0 = no limit).
  uint32_t max_dims_per_segment = 0;

  /// Distance metric.
  DistanceMetric metric = DistanceMetric::kL2;

  /// Whether to generate per-segment rotation matrices.
  bool use_segment_rotation = true;
};

/// @brief Encoding configuration.
struct SAQEncodeConfig {
  /// Whether to use CAQ refinement (code adjustment).
  bool use_caq = true;

  /// CAQ configuration.
  CAQConfig caq_config;
};

/// @brief Search result for a single query.
struct SearchResult {
  uint32_t index = 0;     ///< Index of the result vector.
  float distance = 0.0f;  ///< Distance to query.
};

/// @brief Encoded vector in scalar quantization format.
///
/// Each dimension is quantized to B bits (per the segment's allocation).
/// The codes are packed as uint8_t per dimension (supports up to 8 bits).
/// A per-vector v_max stores the scaling factor for reconstruction.
struct ScalarEncodedVector {
  std::vector<uint8_t> codes;  ///< Per-dimension codes, size = working_dim.
  float v_max = 0.0f;          ///< Max absolute value for this vector.
};

/// @brief The main SAQ quantizer using scalar quantization.
///
/// Implements the full SAQ pipeline:
/// 1. PCA projection (optional dimensionality reduction)
/// 2. Joint dimension segmentation + bit allocation (DP)
/// 3. Per-segment random orthonormal rotation
/// 4. Scalar (uniform grid) quantization per dimension
/// 5. Code adjustment (CAQ) for cosine similarity
class SAQQuantizer {
 public:
  SAQQuantizer() = default;
  ~SAQQuantizer() = default;

  // Non-copyable, movable
  SAQQuantizer(const SAQQuantizer&) = delete;
  SAQQuantizer& operator=(const SAQQuantizer&) = delete;
  SAQQuantizer(SAQQuantizer&&) = default;
  SAQQuantizer& operator=(SAQQuantizer&&) = default;

  /// @brief Train the quantizer on a dataset.
  /// @param data Training vectors, row-major (n_vectors x dim).
  /// @param n_vectors Number of training vectors.
  /// @param dim Vector dimensionality.
  /// @param config Training configuration.
  /// @return Error message, empty on success.
  std::string Train(const float* data, uint32_t n_vectors, uint32_t dim,
                    const SAQTrainConfig& config);

  /// @brief Encode a single vector to scalar codes.
  /// @param vector Input vector.
  /// @param encoded Output encoded vector.
  /// @param config Encoding configuration.
  /// @return True on success.
  bool Encode(const float* vector, ScalarEncodedVector& encoded,
              const SAQEncodeConfig& config = {}) const;

  /// @brief Encode a batch of vectors.
  /// @param vectors Input vectors, row-major (n_vectors x dim).
  /// @param n_vectors Number of vectors.
  /// @param encoded Output encoded vectors.
  /// @param config Encoding configuration.
  /// @return True on success.
  bool EncodeBatch(const float* vectors, uint32_t n_vectors,
                   std::vector<ScalarEncodedVector>& encoded,
                   const SAQEncodeConfig& config = {}) const;

  /// @brief Decode an encoded vector back to float.
  /// @param encoded Encoded vector.
  /// @param vector Output vector (must have size = dim).
  /// @return True on success.
  bool Decode(const ScalarEncodedVector& encoded, float* vector) const;

  /// @brief Estimate inner product between query and encoded vector.
  ///
  /// Uses the SAQ paper formula:
  /// <o_bar, q> = delta * <codes, q> + q_sum * (-v_max + delta/2)
  ///
  /// @param query Query vector (working dim).
  /// @param encoded Encoded database vector.
  /// @return Estimated inner product.
  float EstimateInnerProduct(const float* query,
                             const ScalarEncodedVector& encoded) const;

  /// @brief Search for k nearest neighbors.
  /// @param query Query vector.
  /// @param encoded_db Database of encoded vectors.
  /// @param k Number of neighbors to return.
  /// @param results Output results (sorted by distance).
  void Search(const float* query,
              const std::vector<ScalarEncodedVector>& encoded_db,
              uint32_t k, std::vector<SearchResult>& results) const;

  /// @brief Get the quantization plan.
  const QuantizationPlan& Plan() const { return plan_; }

  /// @brief Get a mutable reference to the plan (for serialization).
  QuantizationPlan& MutablePlan() { return plan_; }

  /// @brief Load from a quantization plan.
  /// @param plan The plan to load.
  /// @return Error message, empty on success.
  std::string LoadPlan(const QuantizationPlan& plan);

  /// @brief Get input dimensionality.
  uint32_t Dim() const { return plan_.dimension; }

  /// @brief Get working dimensionality (after PCA if enabled).
  uint32_t WorkingDim() const {
    return plan_.use_pca ? plan_.pca.output_dim : plan_.dimension;
  }

  /// @brief Get number of segments.
  uint32_t NumSegments() const { return plan_.segment_count; }

  /// @brief Get total bits per vector.
  uint32_t TotalBits() const { return plan_.total_bits; }

  /// @brief Check if trained.
  bool IsTrained() const { return trained_; }

  /// @brief Transform a query vector to rotated space for distance estimation.
  /// Applies PCA projection (if enabled) and per-segment rotation.
  /// @param query Input query vector (dim).
  /// @param rotated_query Output vector in rotated space (working_dim).
  /// @return True on success.
  bool TransformQuery(const float* query, float* rotated_query) const;

 private:
  /// @brief Generate per-segment rotation matrices.
  std::string GenerateRotations(uint32_t seed);

  /// @brief Apply per-segment rotation to a working-dim vector.
  void ApplyRotation(const float* input, float* output) const;

  /// @brief Apply inverse per-segment rotation.
  void ApplyInverseRotation(const float* input, float* output) const;

  /// @brief Scalar-quantize a rotated vector.
  void ScalarQuantize(const float* rotated, uint32_t working_dim,
                      ScalarEncodedVector& encoded) const;

  /// @brief Reconstruct from scalar codes in rotated space.
  void ScalarDequantize(const ScalarEncodedVector& encoded,
                        float* rotated) const;

  QuantizationPlan plan_;
  PCAProjection pca_;
  CAQAdjuster caq_adjuster_;
  DistanceEstimator distance_estimator_;
  bool trained_ = false;
};

}  // namespace saq
