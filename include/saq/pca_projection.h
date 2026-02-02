#pragma once

/// @file pca_projection.h
/// @brief PCA-based dimensionality reduction for SAQ preprocessing.

#include "saq/quantization_plan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace saq {

/// @brief PCA (Principal Component Analysis) for dimensionality reduction.
///
/// Computes principal components from training data and projects vectors
/// onto a lower-dimensional subspace ordered by variance.
class PCAProjection {
 public:
  PCAProjection() = default;
  ~PCAProjection() = default;

  // Non-copyable, movable
  PCAProjection(const PCAProjection&) = delete;
  PCAProjection& operator=(const PCAProjection&) = delete;
  PCAProjection(PCAProjection&&) = default;
  PCAProjection& operator=(PCAProjection&&) = default;

  /// @brief Train PCA from data matrix.
  /// @param data Row-major matrix of shape (n_samples, input_dim).
  /// @param n_samples Number of training vectors.
  /// @param input_dim Original dimensionality.
  /// @param output_dim Target dimensionality (must be <= input_dim).
  /// @param center If true, subtract mean before computing covariance.
  /// @return True on success, false on failure (check error()).
  bool Train(const float* data, uint32_t n_samples, uint32_t input_dim,
             uint32_t output_dim, bool center = true);

  /// @brief Project a single vector to lower dimension.
  /// @param input Vector of size input_dim.
  /// @param output Vector of size output_dim (must be pre-allocated).
  void Project(const float* input, float* output) const;

  /// @brief Project multiple vectors (batch).
  /// @param input Row-major matrix of shape (n, input_dim).
  /// @param output Row-major matrix of shape (n, output_dim), pre-allocated.
  /// @param n Number of vectors.
  void ProjectBatch(const float* input, float* output, uint32_t n) const;

  /// @brief Inverse project (reconstruct) from lower dimension.
  /// @param input Vector of size output_dim.
  /// @param output Vector of size input_dim (must be pre-allocated).
  void InverseProject(const float* input, float* output) const;

  /// @brief Get the learned mean vector.
  /// @return Pointer to mean vector of size input_dim, or nullptr if not trained.
  const float* Mean() const;

  /// @brief Get the principal components matrix.
  /// @return Pointer to row-major matrix of shape (output_dim, input_dim).
  const float* Components() const;

  /// @brief Get the eigenvalues (variance explained by each component).
  /// @return Vector of eigenvalues in descending order.
  const std::vector<float>& Eigenvalues() const { return eigenvalues_; }

  /// @brief Export to PCAParams for serialization.
  /// @param params Output parameter struct.
  void ExportParams(PCAParams* params) const;

  /// @brief Import from PCAParams (skip training).
  /// @param params Input parameter struct.
  /// @return True if valid, false otherwise.
  bool ImportParams(const PCAParams& params);

  /// @brief Check if PCA has been trained or loaded.
  bool IsTrained() const { return trained_; }

  /// @brief Get input dimensionality.
  uint32_t InputDim() const { return input_dim_; }

  /// @brief Get output dimensionality.
  uint32_t OutputDim() const { return output_dim_; }

  /// @brief Get last error message.
  const std::string& Error() const { return error_; }

 private:
  bool trained_ = false;
  uint32_t input_dim_ = 0;
  uint32_t output_dim_ = 0;

  std::vector<float> mean_;        // size: input_dim
  std::vector<float> components_;  // size: output_dim * input_dim (row-major)
  std::vector<float> eigenvalues_; // size: output_dim

  std::string error_;
};

}  // namespace saq

