#pragma once

/// @file fast_scan.h
/// @brief FastScan SIMD-accelerated distance table lookup.
///
/// Implements FAISS-style FastScan for rapid distance estimation using
/// SIMD shuffle instructions for parallel table lookups. Supports 4-bit
/// and 8-bit quantization codes.
///
/// Key optimizations:
/// - Uses vpshufb (AVX2/AVX-512) for 16-way parallel table lookup
/// - Blocked code layout for cache-friendly access
/// - Processes 32 vectors per iteration (AVX2) or 64 (AVX-512)

#include <cstdint>
#include <vector>

#if defined(_MSC_VER)
  #include <intrin.h>
#else
  #include <immintrin.h>
#endif

namespace saq {

// ============================================================================
// FastScan Configuration
// ============================================================================

/// @brief Block size for FastScan (number of vectors processed together).
/// Must be multiple of 32 for AVX2, 64 for AVX-512.
constexpr uint32_t kFastScanBlockSize = 32;

/// @brief Maximum number of segments supported by FastScan.
constexpr uint32_t kFastScanMaxSegments = 64;

// ============================================================================
// Packed Code Layout
// ============================================================================

/// @brief Packed codes for FastScan.
///
/// Codes are reorganized from row-major (vector, segment) to blocked layout:
/// - Vectors are grouped into blocks of kFastScanBlockSize
/// - Within each block, codes are interleaved for SIMD access
/// - 4-bit codes: 2 codes per byte, low nibble first
/// - 8-bit codes: 1 code per byte
struct FastScanCodes {
  uint32_t num_vectors = 0;    ///< Total number of vectors.
  uint32_t num_segments = 0;   ///< Number of segments.
  uint32_t bits_per_code = 4;  ///< Bits per code (4 or 8).
  uint32_t block_size = 32;    ///< Vectors per block.
  std::vector<uint8_t> data;   ///< Packed code data.
  
  /// @brief Get size in bytes per block.
  size_t BytesPerBlock() const {
    if (bits_per_code == 4) {
      return static_cast<size_t>(block_size) * num_segments / 2;
    } else {
      return static_cast<size_t>(block_size) * num_segments;
    }
  }
  
  /// @brief Get number of blocks.
  uint32_t NumBlocks() const {
    return (num_vectors + block_size - 1) / block_size;
  }
};

/// @brief Packed lookup table for FastScan.
///
/// Tables are packed for efficient SIMD loading:
/// - For 4-bit codes: 16 entries × 4 bytes = 64 bytes per segment (fits in AVX-512)
/// - Quantized to uint8 for compact storage
struct FastScanLUT {
  uint32_t num_segments = 0;
  float scale = 1.0f;         ///< Scale factor for dequantization.
  float bias = 0.0f;          ///< Bias for dequantization.
  std::vector<uint8_t> data;  ///< Quantized LUT entries.
};

// ============================================================================
// Code Packing Functions
// ============================================================================

/// @brief Pack codes from row-major layout to FastScan blocked layout.
///
/// @param codes Input codes in row-major order (n_vectors × num_segments).
/// @param n_vectors Number of vectors.
/// @param num_segments Number of segments.
/// @param bits_per_code Bits per code (4 or 8).
/// @param output Output packed codes.
/// @return True on success.
bool PackCodes4bit(const uint32_t* codes, uint32_t n_vectors,
                    uint32_t num_segments, FastScanCodes& output);

/// @brief Pack 8-bit codes to FastScan blocked layout.
bool PackCodes8bit(const uint32_t* codes, uint32_t n_vectors,
                    uint32_t num_segments, FastScanCodes& output);

/// @brief Unpack codes from FastScan layout back to row-major.
bool UnpackCodes(const FastScanCodes& packed, uint32_t* codes);

// ============================================================================
// LUT Packing Functions
// ============================================================================

/// @brief Pack float distance table to quantized FastScan LUT.
///
/// Quantizes float distances to uint8 for SIMD processing.
/// Final distances are recovered as: result * scale + bias
///
/// @param tables Pointers to distance tables for each segment.
/// @param num_segments Number of segments.
/// @param centroids_per_segment Number of centroids per segment (max 16 for 4-bit).
/// @param output Output packed LUT.
/// @return True on success.
bool PackLUT4bit(const float* const* tables, uint32_t num_segments,
                  uint32_t centroids_per_segment, FastScanLUT& output);

/// @brief Pack float distance table with variable centroid counts.
///
/// @param tables Pointers to distance tables for each segment.
/// @param num_segments Number of segments.
/// @param centroids_per_segment Array of centroid counts per segment (each must be ≤16).
/// @param output Output packed LUT.
/// @return True on success.
bool PackLUT4bitVariable(const float* const* tables, uint32_t num_segments,
                          const uint32_t* centroids_per_segment, FastScanLUT& output);

/// @brief Pack LUT for 8-bit codes (256 centroids per segment).
bool PackLUT8bit(const float* const* tables, uint32_t num_segments,
                  FastScanLUT& output);

/// @brief Pack LUT for 8-bit codes with variable centroid counts.
bool PackLUT8bitVariable(const float* const* tables, uint32_t num_segments,
                          const uint32_t* centroids_per_segment, FastScanLUT& output);

// ============================================================================
// FastScan Distance Estimation
// ============================================================================

/// @brief Estimate distances using FastScan for 4-bit codes.
///
/// Uses SIMD shuffle for parallel table lookups. Processes 32 vectors
/// at a time (AVX2) or 64 (AVX-512).
///
/// @param packed_codes Packed codes in FastScan layout.
/// @param packed_lut Packed lookup table.
/// @param distances Output distances (size: num_vectors).
void FastScanEstimate4bit(const FastScanCodes& packed_codes,
                           const FastScanLUT& packed_lut,
                           float* distances);

/// @brief Estimate distances for 8-bit codes.
void FastScanEstimate8bit(const FastScanCodes& packed_codes,
                           const FastScanLUT& packed_lut,
                           float* distances);

/// @brief Estimate distances with selector mask.
///
/// Only computes distances for vectors where mask[i] is true.
/// Useful for IVF where only vectors in selected clusters are scanned.
///
/// @param packed_codes Packed codes.
/// @param packed_lut Packed LUT.
/// @param mask Boolean mask (size: num_vectors).
/// @param distances Output distances.
void FastScanEstimateSelected4bit(const FastScanCodes& packed_codes,
                                   const FastScanLUT& packed_lut,
                                   const uint8_t* mask,
                                   float* distances);

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Check if FastScan can be used for the given configuration.
///
/// FastScan requires:
/// - All segments have same number of centroids (16 or 256)
/// - num_segments is even (for efficient packing)
///
/// @param centroids_per_segment Array of centroid counts.
/// @param num_segments Number of segments.
/// @return True if FastScan is applicable.
bool CanUseFastScan(const uint32_t* centroids_per_segment,
                     uint32_t num_segments);

/// @brief Get the recommended bits per code for FastScan.
/// @param max_centroids Maximum centroids across all segments.
/// @return 4 for ≤16 centroids, 8 for ≤256, 0 if unsupported.
uint32_t RecommendedFastScanBits(uint32_t max_centroids);

}  // namespace saq
