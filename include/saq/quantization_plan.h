#pragma once

/// @file quantization_plan.h
/// @brief Data structures for SAQ quantization plan serialization.
///
/// Defines the QuantizationPlan struct and related types that describe
/// how vectors are segmented, how bits are allocated, and what codebooks
/// are used for quantization.

#include <cstdint>
#include <string>
#include <vector>

namespace saq {

/// @brief Serialization format for QuantizationPlan.
enum class SerializationFormat : uint8_t {
  kBinary = 0,  ///< Compact binary format.
  kJson = 1     ///< Human-readable JSON format.
};

/// @brief Parameters for optional PCA dimensionality reduction.
struct PCAParams {
  uint32_t input_dim = 0;         ///< Original vector dimensionality.
  uint32_t output_dim = 0;        ///< Reduced dimensionality after PCA.
  bool enabled = false;           ///< Whether PCA is applied.
  std::vector<float> mean;        ///< Mean vector, size: input_dim.
  std::vector<float> components;  ///< Principal components, size: output_dim * input_dim (row-major).
};

/// @brief A contiguous range of dimensions treated as one quantization unit.
struct Segment {
  uint32_t id = 0;          ///< Unique segment identifier.
  uint32_t start_dim = 0;   ///< Starting dimension index (inclusive).
  uint32_t dim_count = 0;   ///< Number of dimensions in this segment.
  uint32_t bits = 0;        ///< Bits per dimension (B in the SAQ paper).
                             ///< Total bits for segment = bits * dim_count.
};

/// @brief Codebook for quantizing a segment's dimensions (legacy, k-means based).
struct Codebook {
  uint32_t segment_id = 0;    ///< ID of the segment this codebook serves.
  uint32_t bits = 0;          ///< Bits used for quantization (log2 of centroids).
  uint32_t centroids = 0;     ///< Number of centroids, typically 2^bits.
  uint32_t dim_count = 0;     ///< Dimensionality of each centroid.
  std::vector<float> data;    ///< Centroid vectors, size: centroids * dim_count.
};

/// @brief Per-segment rotation matrix for decorrelating dimensions.
struct SegmentRotation {
  uint32_t segment_id = 0;     ///< ID of the segment this rotation serves.
  uint32_t dim_count = 0;      ///< Size of the rotation matrix (dim_count x dim_count).
  std::vector<float> matrix;   ///< Orthonormal rotation matrix, row-major.
};

/// @brief Complete specification for SAQ vector quantization.
///
/// Contains all parameters needed to encode and decode vectors:
/// PCA projection, dimension segments, bit allocations, and per-segment
/// rotation matrices. Uses scalar (uniform grid) quantization per dimension.
struct QuantizationPlan {
  uint32_t version = 2;           ///< Schema version (2 = scalar quantization).
  uint32_t dimension = 0;         ///< Input vector dimensionality.
  uint32_t total_bits = 0;        ///< Total bits per encoded vector.
  uint32_t segment_count = 0;     ///< Number of dimension segments.
  uint32_t seed = 0;              ///< RNG seed for reproducibility.
  bool use_pca = false;           ///< Whether PCA preprocessing is enabled.

  PCAParams pca;                  ///< PCA parameters (if use_pca is true).
  std::vector<Segment> segments;  ///< Dimension segment definitions.
  std::vector<SegmentRotation> rotations; ///< Per-segment rotation matrices.

  // Legacy fields (kept for backward compatibility)
  uint32_t codebook_count = 0;    ///< Number of codebooks (legacy).
  std::vector<Codebook> codebooks;///< Codebooks for each segment (legacy).

  /// @brief Validate internal consistency.
  /// @param error Optional output for error message.
  /// @return True if valid.
  bool Validate(std::string* error) const;

  /// @brief Serialize to compact binary format.
  /// @return Binary data.
  std::vector<uint8_t> SerializeBinary() const;

  /// @brief Deserialize from binary format.
  /// @param data Binary data.
  /// @param error Optional output for error message.
  /// @return True on success.
  bool DeserializeBinary(const std::vector<uint8_t>& data, std::string* error);

  /// @brief Serialize to JSON string.
  /// @param pretty If true, format with indentation.
  /// @return JSON string.
  std::string SerializeJson(bool pretty = false) const;

  /// @brief Deserialize from JSON string.
  /// @param json JSON string.
  /// @param error Optional output for error message.
  /// @return True on success.
  bool DeserializeJson(const std::string& json, std::string* error);
};

}  // namespace saq

