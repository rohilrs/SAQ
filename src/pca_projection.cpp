/// @file pca_projection.cpp
/// @brief Implementation of PCAProjection class.
///
/// Uses Eigen's SelfAdjointEigenSolver for numerically stable
/// eigendecomposition at any dimension.

#include "saq/pca_projection.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <Eigen/Dense>

#ifdef SAQ_USE_OPENMP
#include <omp.h>
#endif

namespace saq {

namespace {

/// @brief Compute mean of each column (dimension).
void ComputeMean(const float* data, uint32_t n_samples, uint32_t dim,
                 float* mean) {
  std::fill(mean, mean + dim, 0.0f);
  for (uint32_t i = 0; i < n_samples; ++i) {
    for (uint32_t j = 0; j < dim; ++j) {
      mean[j] += data[i * dim + j];
    }
  }
  const float inv_n = 1.0f / static_cast<float>(n_samples);
  for (uint32_t j = 0; j < dim; ++j) {
    mean[j] *= inv_n;
  }
}

}  // anonymous namespace

bool PCAProjection::Train(const float* data, uint32_t n_samples,
                          uint32_t input_dim, uint32_t output_dim,
                          bool center) {
  if (data == nullptr) {
    error_ = "data is null";
    return false;
  }
  if (n_samples < 2) {
    error_ = "need at least 2 samples";
    return false;
  }
  if (input_dim == 0) {
    error_ = "input_dim must be > 0";
    return false;
  }
  if (output_dim == 0 || output_dim > input_dim) {
    error_ = "output_dim must be in [1, input_dim]";
    return false;
  }

  input_dim_ = input_dim;
  output_dim_ = output_dim;

  // --- Step 1: Compute mean in double precision ---
  mean_.resize(input_dim);
  if (center) {
    std::vector<double> mean_d(input_dim, 0.0);
    for (uint32_t i = 0; i < n_samples; ++i) {
      for (uint32_t j = 0; j < input_dim; ++j) {
        mean_d[j] += static_cast<double>(data[i * input_dim + j]);
      }
    }
    const double inv_n = 1.0 / static_cast<double>(n_samples);
    for (uint32_t j = 0; j < input_dim; ++j) {
      mean_d[j] *= inv_n;
      mean_[j] = static_cast<float>(mean_d[j]);
    }
  } else {
    std::fill(mean_.begin(), mean_.end(), 0.0f);
  }

  // --- Step 2: Incremental covariance in double precision ---
  // Instead of materializing the full N x D centered matrix (~N*D*4 bytes),
  // we process data in blocks and accumulate the D x D covariance directly.
  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(input_dim, input_dim);

  constexpr uint32_t kBlockSize = 256;
  for (uint32_t start = 0; start < n_samples; start += kBlockSize) {
    const uint32_t block_n = std::min(kBlockSize, n_samples - start);

    // Build a small centered block in double precision
    Eigen::MatrixXd block_data(block_n, input_dim);
    for (uint32_t i = 0; i < block_n; ++i) {
      const float* row = data + static_cast<size_t>(start + i) * input_dim;
      for (uint32_t j = 0; j < input_dim; ++j) {
        block_data(i, j) = static_cast<double>(row[j]) -
                            static_cast<double>(mean_[j]);
      }
    }

    // Accumulate outer product: cov += block^T * block
    cov.noalias() += block_data.transpose() * block_data;
  }

  // Normalize by (n_samples - 1) for unbiased covariance estimate
  cov /= static_cast<double>(n_samples - 1);

  // --- Step 3: Eigendecomposition in double precision ---
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
  if (solver.info() != Eigen::Success) {
    error_ = "Eigen decomposition failed";
    return false;
  }

  const auto& eigenvalues = solver.eigenvalues();   // ascending order
  const auto& eigenvectors = solver.eigenvectors();  // columns

  // --- Step 4: Store results as float (descending eigenvalue order) ---
  components_.resize(static_cast<size_t>(output_dim) * input_dim);
  eigenvalues_.resize(output_dim);

  for (uint32_t k = 0; k < output_dim; ++k) {
    // Map descending index to ascending: largest eigenvalue is at col (dim-1)
    uint32_t src_col = input_dim - 1 - k;
    eigenvalues_[k] = static_cast<float>(eigenvalues(src_col));

    // Copy eigenvector column into components row k
    for (uint32_t j = 0; j < input_dim; ++j) {
      components_[k * input_dim + j] =
          static_cast<float>(eigenvectors(j, src_col));
    }
  }

  trained_ = true;
  error_.clear();
  return true;
}

void PCAProjection::Project(const float* input, float* output) const {
  if (!trained_) {
    return;
  }

  // output[k] = sum_j((input[j] - mean[j]) * components[k][j])
  for (uint32_t k = 0; k < output_dim_; ++k) {
    double sum = 0.0;
    const float* comp_row = components_.data() + k * input_dim_;
    for (uint32_t j = 0; j < input_dim_; ++j) {
      sum += static_cast<double>(input[j] - mean_[j]) *
             static_cast<double>(comp_row[j]);
    }
    output[k] = static_cast<float>(sum);
  }
}

void PCAProjection::ProjectBatch(const float* input, float* output,
                                 uint32_t n) const {
  if (!trained_) {
    return;
  }

#ifdef SAQ_USE_OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (uint32_t i = 0; i < n; ++i) {
    Project(input + i * input_dim_, output + i * output_dim_);
  }
}

void PCAProjection::InverseProject(const float* input, float* output) const {
  if (!trained_) {
    return;
  }

  // output[j] = mean[j] + sum_k(input[k] * components[k][j])
  for (uint32_t j = 0; j < input_dim_; ++j) {
    double sum = static_cast<double>(mean_[j]);
    for (uint32_t k = 0; k < output_dim_; ++k) {
      sum += static_cast<double>(input[k]) *
             static_cast<double>(components_[k * input_dim_ + j]);
    }
    output[j] = static_cast<float>(sum);
  }
}

const float* PCAProjection::Mean() const {
  return trained_ ? mean_.data() : nullptr;
}

const float* PCAProjection::Components() const {
  return trained_ ? components_.data() : nullptr;
}

void PCAProjection::ExportParams(PCAParams* params) const {
  if (!params) return;

  params->input_dim = input_dim_;
  params->output_dim = output_dim_;
  params->enabled = trained_;
  params->mean = mean_;
  params->components = components_;
}

bool PCAProjection::ImportParams(const PCAParams& params) {
  if (!params.enabled) {
    error_ = "PCAParams not enabled";
    return false;
  }
  if (params.input_dim == 0) {
    error_ = "input_dim is 0";
    return false;
  }
  if (params.output_dim == 0 || params.output_dim > params.input_dim) {
    error_ = "invalid output_dim";
    return false;
  }
  if (params.mean.size() != params.input_dim) {
    error_ = "mean size mismatch";
    return false;
  }
  size_t expected_comp_size =
      static_cast<size_t>(params.output_dim) * params.input_dim;
  if (params.components.size() != expected_comp_size) {
    error_ = "components size mismatch";
    return false;
  }

  input_dim_ = params.input_dim;
  output_dim_ = params.output_dim;
  mean_ = params.mean;
  components_ = params.components;
  eigenvalues_.clear();  // Not stored in PCAParams
  trained_ = true;
  error_.clear();
  return true;
}

}  // namespace saq
