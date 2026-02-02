#pragma once

/// @file saq_quantizer.h
/// @brief Main SAQ quantizer integrating all components.
///
/// Provides the unified interface for training, encoding, and searching
/// with Segmented Additive Quantization. Orchestrates PCA projection,
/// dimension segmentation, bit allocation, codebook training, and
/// asymmetric distance computation.

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
  /// Target bits per vector.
  uint32_t total_bits = 64;

  /// Number of segments to partition dimensions.
  uint32_t num_segments = 8;

  /// Whether to apply PCA before quantization.
  bool use_pca = false;

  /// Target dimension after PCA (0 = no reduction).
  uint32_t pca_dim = 0;

  /// K-means iterations for codebook training.
  uint32_t kmeans_iterations = 20;

  /// Random seed for reproducibility.
  uint32_t seed = 42;

  /// Segmentation strategy.
  SegmentationStrategy segmentation_strategy = SegmentationStrategy::kVarianceBased;

  /// Minimum bits per segment.
  uint32_t min_bits_per_segment = 2;

  /// Maximum bits per segment.
  uint32_t max_bits_per_segment = 16;

  /// Distance metric.
  DistanceMetric metric = DistanceMetric::kL2;
};

/// @brief Encoding configuration.
struct SAQEncodeConfig {
  /// Whether to use CAQ refinement.
  bool use_caq = true;

  /// CAQ configuration.
  CAQConfig caq_config;
};

/// @brief Search result for a single query.
struct SearchResult {
  uint32_t index = 0;     ///< Index of the result vector.
  float distance = 0.0f;  ///< Distance to query.
};

/// @brief The main SAQ quantizer.
///
/// Provides complete functionality for:
/// - Training: Learn PCA, segments, bit allocation, and codebooks
/// - Encoding: Compress vectors to codes
/// - Decoding: Reconstruct vectors from codes
/// - Searching: Find nearest neighbors using asymmetric distance
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
  /// @param data Training vectors, row-major (n_vectors × dim).
  /// @param n_vectors Number of training vectors.
  /// @param dim Vector dimensionality.
  /// @param config Training configuration.
  /// @return Error message, empty on success.
  std::string Train(const float* data, uint32_t n_vectors, uint32_t dim,
                    const SAQTrainConfig& config);

  /// @brief Encode a single vector.
  /// @param vector Input vector.
  /// @param codes Output codes (must have size = num_segments).
  /// @param config Encoding configuration.
  /// @return True on success.
  bool Encode(const float* vector, uint32_t* codes,
              const SAQEncodeConfig& config = {}) const;

  /// @brief Encode a batch of vectors.
  /// @param vectors Input vectors, row-major (n_vectors × dim).
  /// @param n_vectors Number of vectors.
  /// @param codes Output codes, row-major (n_vectors × num_segments).
  /// @param config Encoding configuration.
  /// @return True on success.
  bool EncodeBatch(const float* vectors, uint32_t n_vectors, uint32_t* codes,
                   const SAQEncodeConfig& config = {}) const;

  /// @brief Decode codes to a vector.
  /// @param codes Input codes.
  /// @param vector Output vector (must have size = dim).
  /// @return True on success.
  bool Decode(const uint32_t* codes, float* vector) const;

  /// @brief Search for k nearest neighbors.
  /// @param query Query vector.
  /// @param codes Database codes, row-major (n_vectors × num_segments).
  /// @param n_vectors Number of database vectors.
  /// @param k Number of neighbors to return.
  /// @param results Output results (sorted by distance).
  void Search(const float* query, const uint32_t* codes, uint32_t n_vectors,
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

 private:
  /// @brief Train codebooks using k-means.
  std::string TrainCodebooks(const float* data, uint32_t n_vectors,
                              uint32_t dim, const SAQTrainConfig& config);

  /// @brief Initialize internal components from plan.
  std::string InitializeComponents();

  QuantizationPlan plan_;
  PCAProjection pca_;
  CAQAdjuster caq_adjuster_;
  DistanceEstimator distance_estimator_;
  bool trained_ = false;
};

}  // namespace saq
