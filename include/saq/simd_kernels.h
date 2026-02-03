#pragma once

/// @file simd_kernels.h
/// @brief SIMD-optimized kernels for SAQ distance computation.
///
/// Provides AVX-512/AVX2 vectorized implementations for:
/// - L2 distance computation
/// - Inner product computation
/// - Distance table lookups
/// - Batch distance estimation
///
/// Falls back to scalar implementations when SIMD is unavailable.

#include <cstdint>
#include <vector>

// SIMD detection - compile-time flags
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(__AVX512F__))
  #define SAQ_HAVE_AVX512 1
#endif

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
  #define SAQ_HAVE_AVX2 1
#endif

#if defined(__AVX__) || (defined(_MSC_VER) && defined(__AVX__))
  #define SAQ_HAVE_AVX 1
#endif

#if defined(__SSE4_1__) || (defined(_MSC_VER) && defined(__SSE4_1__))
  #define SAQ_HAVE_SSE4 1
#endif

// Include intrinsics headers
#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(SAQ_HAVE_AVX512) || defined(SAQ_HAVE_AVX2) || defined(SAQ_HAVE_AVX)
  #include <immintrin.h>
#elif defined(SAQ_HAVE_SSE4)
  #include <smmintrin.h>
#endif

namespace saq {
namespace simd {

// ============================================================================
// Configuration
// ============================================================================

/// @brief SIMD vector width in floats.
#if defined(SAQ_HAVE_AVX512)
  constexpr uint32_t kSimdWidth = 16;  // 512 bits / 32 bits
#elif defined(SAQ_HAVE_AVX2) || defined(SAQ_HAVE_AVX)
  constexpr uint32_t kSimdWidth = 8;   // 256 bits / 32 bits
#elif defined(SAQ_HAVE_SSE4)
  constexpr uint32_t kSimdWidth = 4;   // 128 bits / 32 bits
#else
  constexpr uint32_t kSimdWidth = 1;   // Scalar fallback
#endif

/// @brief Batch size for distance estimation (must be multiple of kSimdWidth).
constexpr uint32_t kBatchSize = 32;

// ============================================================================
// Distance Kernels
// ============================================================================

/// @brief Compute squared L2 distance between two vectors.
/// @param a First vector.
/// @param b Second vector.
/// @param dim Dimensionality.
/// @return Squared L2 distance.
float L2DistanceSquared(const float* a, const float* b, uint32_t dim);

/// @brief Compute inner product between two vectors.
/// @param a First vector.
/// @param b Second vector.
/// @param dim Dimensionality.
/// @return Inner product.
float InnerProduct(const float* a, const float* b, uint32_t dim);

/// @brief Compute L2 distances from query to multiple vectors.
/// @param query Query vector.
/// @param vectors Database vectors (n × dim, row-major).
/// @param n Number of vectors.
/// @param dim Dimensionality.
/// @param distances Output distances (size n).
void L2DistancesBatch(const float* query, const float* vectors,
                       uint32_t n, uint32_t dim, float* distances);

/// @brief Compute inner products from query to multiple vectors.
/// @param query Query vector.
/// @param vectors Database vectors (n × dim, row-major).
/// @param n Number of vectors.
/// @param dim Dimensionality.
/// @param products Output inner products (size n).
void InnerProductsBatch(const float* query, const float* vectors,
                         uint32_t n, uint32_t dim, float* products);

// ============================================================================
// Distance Table Operations
// ============================================================================

/// @brief Compute distance table for one segment.
///
/// Computes distances from query segment to all centroids in the codebook.
///
/// @param query Query segment vector.
/// @param centroids Codebook centroids (num_centroids × dim).
/// @param num_centroids Number of centroids.
/// @param dim Segment dimensionality.
/// @param is_ip True for inner product, false for L2.
/// @param table Output distance table (size num_centroids).
void ComputeSegmentTable(const float* query, const float* centroids,
                          uint32_t num_centroids, uint32_t dim,
                          bool is_ip, float* table);

/// @brief Estimate distances for a batch of encoded vectors.
///
/// Uses precomputed distance tables to estimate distances via table lookups.
///
/// @param tables Array of pointers to distance tables for each segment.
/// @param codes Encoded vectors (n × num_segments, row-major).
/// @param n Number of vectors.
/// @param num_segments Number of segments.
/// @param distances Output distances (size n).
void EstimateDistancesBatch(const float* const* tables,
                             const uint32_t* codes,
                             uint32_t n, uint32_t num_segments,
                             float* distances);

/// @brief Estimate distances for a batch using flat table layout.
///
/// @param flat_table Flattened distance table (sum of all centroid counts).
/// @param segment_offsets Offset into flat_table for each segment.
/// @param codes Encoded vectors (n × num_segments).
/// @param n Number of vectors.
/// @param num_segments Number of segments.
/// @param distances Output distances.
void EstimateDistancesBatchFlat(const float* flat_table,
                                 const uint32_t* segment_offsets,
                                 const uint32_t* codes,
                                 uint32_t n, uint32_t num_segments,
                                 float* distances);

// ============================================================================
// Horizontal Reductions
// ============================================================================

/// @brief Sum all elements in an array.
/// @param data Input array.
/// @param n Number of elements.
/// @return Sum of elements.
float HorizontalSum(const float* data, uint32_t n);

/// @brief Find minimum value and its index.
/// @param data Input array.
/// @param n Number of elements.
/// @param min_idx Output index of minimum.
/// @return Minimum value.
float FindMin(const float* data, uint32_t n, uint32_t* min_idx);

/// @brief Find k smallest values and their indices.
/// @param data Input array.
/// @param n Number of elements.
/// @param k Number of smallest to find.
/// @param values Output values (size k).
/// @param indices Output indices (size k).
void FindKSmallest(const float* data, uint32_t n, uint32_t k,
                    float* values, uint32_t* indices);

// ============================================================================
// Memory Operations
// ============================================================================

/// @brief Prefetch data into cache.
/// @param ptr Pointer to prefetch.
template <int Level = 0>
inline void Prefetch(const void* ptr) {
#if defined(_MSC_VER) || defined(SAQ_HAVE_AVX512) || defined(SAQ_HAVE_AVX2) || defined(SAQ_HAVE_AVX)
  if constexpr (Level == 0) {
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
  } else if constexpr (Level == 1) {
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T1);
  } else {
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T2);
  }
#else
  (void)ptr;
#endif
}

/// @brief Check if pointer is aligned.
/// @param ptr Pointer to check.
/// @param alignment Alignment in bytes.
/// @return True if aligned.
inline bool IsAligned(const void* ptr, size_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// ============================================================================
// Platform Detection
// ============================================================================

/// @brief Get the SIMD level supported on this CPU.
/// @return String describing SIMD support.
const char* GetSimdLevel();

/// @brief Check if AVX-512 is available at runtime.
/// @return True if AVX-512 is supported.
bool HasAVX512();

/// @brief Check if AVX2 is available at runtime.
/// @return True if AVX2 is supported.
bool HasAVX2();

}  // namespace simd
}  // namespace saq
