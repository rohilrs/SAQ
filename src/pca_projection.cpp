/// @file pca_projection.cpp
/// @brief Implementation of PCAProjection class.

#include "saq/pca_projection.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

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

/// @brief Compute covariance matrix (dim x dim) from centered data.
/// @param centered Row-major centered data (n_samples x dim).
/// @param cov Output covariance matrix (dim x dim), row-major.
void ComputeCovariance(const float* centered, uint32_t n_samples, uint32_t dim,
                       float* cov) {
  const float inv_n = 1.0f / static_cast<float>(n_samples - 1);

  // cov[i][j] = sum_k(centered[k][i] * centered[k][j]) / (n-1)
#ifdef SAQ_USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (uint32_t i = 0; i < dim; ++i) {
    for (uint32_t j = i; j < dim; ++j) {
      double sum = 0.0;
      for (uint32_t k = 0; k < n_samples; ++k) {
        sum += static_cast<double>(centered[k * dim + i]) *
               static_cast<double>(centered[k * dim + j]);
      }
      float val = static_cast<float>(sum * inv_n);
      cov[i * dim + j] = val;
      cov[j * dim + i] = val;  // Symmetric
    }
  }
}

/// @brief Power iteration to find dominant eigenvector.
/// @param matrix Symmetric matrix (dim x dim).
/// @param dim Matrix dimension.
/// @param eigenvector Output eigenvector (size dim), normalized.
/// @param max_iter Maximum iterations.
/// @param tol Convergence tolerance.
/// @return Eigenvalue.
float PowerIteration(const float* matrix, uint32_t dim, float* eigenvector,
                     uint32_t max_iter = 100, float tol = 1e-6f) {
  // Random initialization
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (uint32_t i = 0; i < dim; ++i) {
    eigenvector[i] = dist(rng);
  }

  // Normalize
  float norm = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    norm += eigenvector[i] * eigenvector[i];
  }
  norm = std::sqrt(norm);
  for (uint32_t i = 0; i < dim; ++i) {
    eigenvector[i] /= norm;
  }

  std::vector<float> temp(dim);
  float eigenvalue = 0.0f;

  for (uint32_t iter = 0; iter < max_iter; ++iter) {
    // temp = matrix * eigenvector
    for (uint32_t i = 0; i < dim; ++i) {
      double sum = 0.0;
      for (uint32_t j = 0; j < dim; ++j) {
        sum += static_cast<double>(matrix[i * dim + j]) *
               static_cast<double>(eigenvector[j]);
      }
      temp[i] = static_cast<float>(sum);
    }

    // Compute new eigenvalue estimate (Rayleigh quotient)
    float new_eigenvalue = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
      new_eigenvalue += eigenvector[i] * temp[i];
    }

    // Normalize temp
    norm = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
      norm += temp[i] * temp[i];
    }
    norm = std::sqrt(norm);
    if (norm < 1e-10f) {
      break;
    }
    for (uint32_t i = 0; i < dim; ++i) {
      temp[i] /= norm;
    }

    // Check convergence
    float diff = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
      float d = temp[i] - eigenvector[i];
      diff += d * d;
    }

    std::copy(temp.begin(), temp.end(), eigenvector);
    eigenvalue = new_eigenvalue;

    if (diff < tol * tol) {
      break;
    }
  }

  return eigenvalue;
}

/// @brief Deflate matrix by removing contribution of eigenvector.
/// matrix = matrix - eigenvalue * eigenvector * eigenvector^T
void DeflateMatrix(float* matrix, uint32_t dim, float eigenvalue,
                   const float* eigenvector) {
  for (uint32_t i = 0; i < dim; ++i) {
    for (uint32_t j = 0; j < dim; ++j) {
      matrix[i * dim + j] -= eigenvalue * eigenvector[i] * eigenvector[j];
    }
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

  // Compute mean
  mean_.resize(input_dim);
  if (center) {
    ComputeMean(data, n_samples, input_dim, mean_.data());
  } else {
    std::fill(mean_.begin(), mean_.end(), 0.0f);
  }

  // Center the data
  std::vector<float> centered(static_cast<size_t>(n_samples) * input_dim);
  for (uint32_t i = 0; i < n_samples; ++i) {
    for (uint32_t j = 0; j < input_dim; ++j) {
      centered[i * input_dim + j] = data[i * input_dim + j] - mean_[j];
    }
  }

  // Compute covariance matrix
  std::vector<float> cov(static_cast<size_t>(input_dim) * input_dim);
  ComputeCovariance(centered.data(), n_samples, input_dim, cov.data());

  // Extract top-k eigenvectors via power iteration with deflation
  components_.resize(static_cast<size_t>(output_dim) * input_dim);
  eigenvalues_.resize(output_dim);

  std::vector<float> eigenvector(input_dim);

  for (uint32_t k = 0; k < output_dim; ++k) {
    float eigenvalue = PowerIteration(cov.data(), input_dim, eigenvector.data());
    eigenvalues_[k] = eigenvalue;

    // Store eigenvector as row k of components
    std::copy(eigenvector.begin(), eigenvector.end(),
              components_.begin() + k * input_dim);

    // Deflate for next iteration
    DeflateMatrix(cov.data(), input_dim, eigenvalue, eigenvector.data());
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
