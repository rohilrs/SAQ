#pragma once

/// @file rotator.h
/// @brief Rotator and PCARotator classes for random orthogonal rotation
///        and PCA-based rotation. Ported from reference saqlib/utils/rotator.hpp.

#include <fstream>
#include <memory>

#include <Eigen/Dense>

#include "saq/defines.h"

namespace saq {

/// @brief Random orthogonal rotation matrix via Householder QR decomposition.
///
/// Generates a D x D orthogonal matrix Q from a random Gaussian matrix,
/// then applies rotation as output = input * Q (or Q^T depending on usage).
class Rotator {
  protected:
    size_t D;      ///< Padded dimension
    FloatRowMat P; ///< Rotation matrix (D x D)

  public:
    explicit Rotator(uint32_t dim)
        : D(dim) {
        this->P = FloatRowMat::Identity(D, D);
    }

    explicit Rotator() {}
    virtual ~Rotator() {}

    /// @brief Set the rotation matrix directly.
    void set(FloatRowMat mat) {
        this->P = std::move(mat);
    }

    /// @brief Generate a random orthogonal matrix via Householder QR.
    ///
    /// Creates a D x D random Gaussian matrix, computes its QR decomposition,
    /// and stores P = Q^T (the transpose of the orthogonal factor).
    void orthogonalize() {
        FloatRowMat RAND(FloatRowMat::Random(D, D));
        Eigen::HouseholderQR<FloatRowMat> qr(RAND);
        FloatRowMat Q = qr.householderQ();
        this->P = Q.transpose(); // inverse of Q = Q.T
    }

    size_t size() const { return D; }

    /// @brief Load the rotation matrix from disk (element-by-element).
    virtual void load(std::ifstream &input) {
        float element;
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                input.read(reinterpret_cast<char *>(&element), sizeof(float));
                P(i, j) = element;
            }
        }
    }

    /// @brief Save the rotation matrix to disk (element-by-element).
    virtual void save(std::ofstream &output) const {
        float element;
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                element = P(i, j);
                output.write(reinterpret_cast<const char *>(&element), sizeof(float));
            }
        }
    }

    /// @brief Rotate matrix A and store the result in RAND_A.
    /// RAND_A = A * P
    virtual void rotate(const FloatRowMat &A, FloatRowMat &RAND_A) const {
        RAND_A = A * P;
    }

    /// @brief Rotate a single row vector A and store the result in RAND_A.
    /// RAND_A = A * P
    virtual void rotate(const Eigen::RowVectorXf &A, Eigen::RowVectorXf &RAND_A) const {
        RAND_A = A * P;
    }

    /// @brief Access the rotation matrix (const).
    auto &get_P() const { return P; }
};

using RotatorPtr = std::unique_ptr<Rotator>;

/// @brief PCA rotation: stores mean vector and rotation matrix.
///
/// Used for PCA-based dimensionality transformation: centers data by
/// subtracting the mean, then applies the rotation matrix.
class PCARotator {
  public:
    size_t D = 0;  ///< Padded dimension
    FloatVec mean; ///< Mean vector (1 x D)
    FloatRowMat P; ///< Rotation matrix (D x D)

    /// @brief Set the rotation matrix and mean vector.
    void set(FloatRowMat mat, FloatVec mean_vec) {
        this->D = mean_vec.cols();
        this->mean = std::move(mean_vec);
        this->P = std::move(mat);
    }

    /// @brief Load PCARotator from disk (binary format).
    void load(std::ifstream &input) {
        input.read(reinterpret_cast<char *>(&D), sizeof(size_t));
        mean.resize(D);
        P.resize(D, D);

        input.read(reinterpret_cast<char *>(mean.data()), D * sizeof(float));
        P = FloatRowMat::Zero(D, D);
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                input.read(reinterpret_cast<char *>(&P(i, j)), sizeof(float));
            }
        }
    }

    /// @brief Save PCARotator to disk (binary format).
    void save(std::ofstream &output) const {
        output.write(reinterpret_cast<const char *>(&D), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(mean.data()), D * sizeof(float));
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                output.write(reinterpret_cast<const char *>(&P(i, j)), sizeof(float));
            }
        }
    }
};

} // namespace saq
