/// @file simd_kernels.cpp
/// @brief SIMD-optimized kernels implementation for SAQ.

#include "saq/simd_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>

namespace saq {
namespace simd {

// ============================================================================
// Platform Detection (Runtime)
// ============================================================================

#if defined(_MSC_VER)
// MSVC runtime CPUID
static void cpuid(int info[4], int function_id) {
  __cpuid(info, function_id);
}

static void cpuidex(int info[4], int function_id, int subfunction_id) {
  __cpuidex(info, function_id, subfunction_id);
}
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static void cpuid(int info[4], int function_id) {
  __cpuid(function_id, info[0], info[1], info[2], info[3]);
}

static void cpuidex(int info[4], int function_id, int subfunction_id) {
  __cpuid_count(function_id, subfunction_id, info[0], info[1], info[2], info[3]);
}
#endif

bool HasAVX512() {
#if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
  int info[4];
  cpuidex(info, 7, 0);
  // Check AVX-512F bit (bit 16 of EBX)
  return (info[1] & (1 << 16)) != 0;
#else
  return false;
#endif
}

bool HasAVX2() {
#if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
  int info[4];
  cpuidex(info, 7, 0);
  // Check AVX2 bit (bit 5 of EBX)
  return (info[1] & (1 << 5)) != 0;
#else
  return false;
#endif
}

const char* GetSimdLevel() {
  if (HasAVX512()) return "AVX-512";
  if (HasAVX2()) return "AVX2";
#if defined(_MSC_VER)
  int info[4];
  cpuid(info, 1);
  if ((info[2] & (1 << 28)) != 0) return "AVX";
  if ((info[2] & (1 << 19)) != 0) return "SSE4.1";
#endif
  return "Scalar";
}

// ============================================================================
// Scalar Fallback Implementations
// ============================================================================

namespace scalar {

float L2DistanceSquared(const float* a, const float* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

float InnerProduct(const float* a, const float* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

void L2DistancesBatch(const float* query, const float* vectors,
                       uint32_t n, uint32_t dim, float* distances) {
  for (uint32_t i = 0; i < n; ++i) {
    distances[i] = L2DistanceSquared(query, vectors + i * dim, dim);
  }
}

void InnerProductsBatch(const float* query, const float* vectors,
                         uint32_t n, uint32_t dim, float* products) {
  for (uint32_t i = 0; i < n; ++i) {
    products[i] = InnerProduct(query, vectors + i * dim, dim);
  }
}

void ComputeSegmentTable(const float* query, const float* centroids,
                          uint32_t num_centroids, uint32_t dim,
                          bool is_ip, float* table) {
  if (is_ip) {
    for (uint32_t c = 0; c < num_centroids; ++c) {
      table[c] = InnerProduct(query, centroids + c * dim, dim);
    }
  } else {
    for (uint32_t c = 0; c < num_centroids; ++c) {
      table[c] = L2DistanceSquared(query, centroids + c * dim, dim);
    }
  }
}

void EstimateDistancesBatch(const float* const* tables,
                             const uint32_t* codes,
                             uint32_t n, uint32_t num_segments,
                             float* distances) {
  for (uint32_t i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (uint32_t s = 0; s < num_segments; ++s) {
      const uint32_t code = codes[i * num_segments + s];
      sum += tables[s][code];
    }
    distances[i] = sum;
  }
}

void EstimateDistancesBatchFlat(const float* flat_table,
                                 const uint32_t* segment_offsets,
                                 const uint32_t* codes,
                                 uint32_t n, uint32_t num_segments,
                                 float* distances) {
  for (uint32_t i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (uint32_t s = 0; s < num_segments; ++s) {
      const uint32_t code = codes[i * num_segments + s];
      sum += flat_table[segment_offsets[s] + code];
    }
    distances[i] = sum;
  }
}

float HorizontalSum(const float* data, uint32_t n) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < n; ++i) {
    sum += data[i];
  }
  return sum;
}

float FindMin(const float* data, uint32_t n, uint32_t* min_idx) {
  float min_val = std::numeric_limits<float>::max();
  uint32_t idx = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (data[i] < min_val) {
      min_val = data[i];
      idx = i;
    }
  }
  if (min_idx) *min_idx = idx;
  return min_val;
}

void FindKSmallest(const float* data, uint32_t n, uint32_t k,
                    float* values, uint32_t* indices) {
  // Use max-heap to track k smallest
  std::priority_queue<std::pair<float, uint32_t>> heap;
  
  for (uint32_t i = 0; i < n; ++i) {
    if (heap.size() < k) {
      heap.push({data[i], i});
    } else if (data[i] < heap.top().first) {
      heap.pop();
      heap.push({data[i], i});
    }
  }
  
  // Extract results in ascending order
  const uint32_t result_size = static_cast<uint32_t>(heap.size());
  for (uint32_t i = result_size; i > 0; --i) {
    values[i - 1] = heap.top().first;
    indices[i - 1] = heap.top().second;
    heap.pop();
  }
}

}  // namespace scalar

// ============================================================================
// AVX2 Implementations
// ============================================================================

#if defined(SAQ_HAVE_AVX2) || defined(_MSC_VER)
namespace avx2 {

inline float HorizontalSum256(__m256 v) {
  // Sum 8 floats in a 256-bit register
  __m128 high = _mm256_extractf128_ps(v, 1);
  __m128 low = _mm256_castps256_ps128(v);
  __m128 sum128 = _mm_add_ps(high, low);
  __m128 shuf = _mm_movehdup_ps(sum128);
  sum128 = _mm_add_ps(sum128, shuf);
  shuf = _mm_movehl_ps(shuf, sum128);
  sum128 = _mm_add_ss(sum128, shuf);
  return _mm_cvtss_f32(sum128);
}

float L2DistanceSquared(const float* a, const float* b, uint32_t dim) {
  __m256 sum = _mm256_setzero_ps();
  
  uint32_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
    sum = _mm256_fmadd_ps(diff, diff, sum);
  }
  
  float result = HorizontalSum256(sum);
  
  // Handle remainder
  for (; i < dim; ++i) {
    const float diff = a[i] - b[i];
    result += diff * diff;
  }
  
  return result;
}

float InnerProduct(const float* a, const float* b, uint32_t dim) {
  __m256 sum = _mm256_setzero_ps();
  
  uint32_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_fmadd_ps(va, vb, sum);
  }
  
  float result = HorizontalSum256(sum);
  
  // Handle remainder
  for (; i < dim; ++i) {
    result += a[i] * b[i];
  }
  
  return result;
}

void L2DistancesBatch(const float* query, const float* vectors,
                       uint32_t n, uint32_t dim, float* distances) {
  for (uint32_t i = 0; i < n; ++i) {
    distances[i] = L2DistanceSquared(query, vectors + i * dim, dim);
  }
}

void InnerProductsBatch(const float* query, const float* vectors,
                         uint32_t n, uint32_t dim, float* products) {
  for (uint32_t i = 0; i < n; ++i) {
    products[i] = InnerProduct(query, vectors + i * dim, dim);
  }
}

}  // namespace avx2
#endif  // SAQ_HAVE_AVX2

// ============================================================================
// AVX-512 Implementations
// ============================================================================

#if defined(SAQ_HAVE_AVX512)
namespace avx512 {

inline float HorizontalSum512(__m512 v) {
  // Reduce 16 floats to 8 floats
  __m256 low = _mm512_castps512_ps256(v);
  __m256 high = _mm512_extractf32x8_ps(v, 1);
  __m256 sum256 = _mm256_add_ps(low, high);
  return avx2::HorizontalSum256(sum256);
}

float L2DistanceSquared(const float* a, const float* b, uint32_t dim) {
  __m512 sum = _mm512_setzero_ps();
  
  uint32_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    __m512 diff = _mm512_sub_ps(va, vb);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }
  
  float result = HorizontalSum512(sum);
  
  // Handle remainder with AVX2
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
    __m256 sq = _mm256_mul_ps(diff, diff);
    result += avx2::HorizontalSum256(sq);
  }
  
  // Handle final remainder
  for (; i < dim; ++i) {
    const float diff = a[i] - b[i];
    result += diff * diff;
  }
  
  return result;
}

float InnerProduct(const float* a, const float* b, uint32_t dim) {
  __m512 sum = _mm512_setzero_ps();
  
  uint32_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    sum = _mm512_fmadd_ps(va, vb, sum);
  }
  
  float result = HorizontalSum512(sum);
  
  // Handle remainder with AVX2
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 prod = _mm256_mul_ps(va, vb);
    result += avx2::HorizontalSum256(prod);
  }
  
  // Handle final remainder
  for (; i < dim; ++i) {
    result += a[i] * b[i];
  }
  
  return result;
}

void L2DistancesBatch(const float* query, const float* vectors,
                       uint32_t n, uint32_t dim, float* distances) {
  // Prefetch ahead for better cache utilization
  constexpr uint32_t kPrefetchDistance = 8;
  
  for (uint32_t i = 0; i < n; ++i) {
    if (i + kPrefetchDistance < n) {
      Prefetch<0>(vectors + (i + kPrefetchDistance) * dim);
    }
    distances[i] = L2DistanceSquared(query, vectors + i * dim, dim);
  }
}

void InnerProductsBatch(const float* query, const float* vectors,
                         uint32_t n, uint32_t dim, float* products) {
  constexpr uint32_t kPrefetchDistance = 8;
  
  for (uint32_t i = 0; i < n; ++i) {
    if (i + kPrefetchDistance < n) {
      Prefetch<0>(vectors + (i + kPrefetchDistance) * dim);
    }
    products[i] = InnerProduct(query, vectors + i * dim, dim);
  }
}

}  // namespace avx512
#endif  // SAQ_HAVE_AVX512

// ============================================================================
// Dispatch Functions (Select best implementation at runtime)
// ============================================================================

float L2DistanceSquared(const float* a, const float* b, uint32_t dim) {
#if defined(SAQ_HAVE_AVX512)
  if (HasAVX512()) {
    return avx512::L2DistanceSquared(a, b, dim);
  }
#endif
#if defined(SAQ_HAVE_AVX2) || defined(_MSC_VER)
  if (HasAVX2()) {
    return avx2::L2DistanceSquared(a, b, dim);
  }
#endif
  return scalar::L2DistanceSquared(a, b, dim);
}

float InnerProduct(const float* a, const float* b, uint32_t dim) {
#if defined(SAQ_HAVE_AVX512)
  if (HasAVX512()) {
    return avx512::InnerProduct(a, b, dim);
  }
#endif
#if defined(SAQ_HAVE_AVX2) || defined(_MSC_VER)
  if (HasAVX2()) {
    return avx2::InnerProduct(a, b, dim);
  }
#endif
  return scalar::InnerProduct(a, b, dim);
}

void L2DistancesBatch(const float* query, const float* vectors,
                       uint32_t n, uint32_t dim, float* distances) {
#if defined(SAQ_HAVE_AVX512)
  if (HasAVX512()) {
    return avx512::L2DistancesBatch(query, vectors, n, dim, distances);
  }
#endif
#if defined(SAQ_HAVE_AVX2) || defined(_MSC_VER)
  if (HasAVX2()) {
    return avx2::L2DistancesBatch(query, vectors, n, dim, distances);
  }
#endif
  return scalar::L2DistancesBatch(query, vectors, n, dim, distances);
}

void InnerProductsBatch(const float* query, const float* vectors,
                         uint32_t n, uint32_t dim, float* products) {
#if defined(SAQ_HAVE_AVX512)
  if (HasAVX512()) {
    return avx512::InnerProductsBatch(query, vectors, n, dim, products);
  }
#endif
#if defined(SAQ_HAVE_AVX2) || defined(_MSC_VER)
  if (HasAVX2()) {
    return avx2::InnerProductsBatch(query, vectors, n, dim, products);
  }
#endif
  return scalar::InnerProductsBatch(query, vectors, n, dim, products);
}

void ComputeSegmentTable(const float* query, const float* centroids,
                          uint32_t num_centroids, uint32_t dim,
                          bool is_ip, float* table) {
  // Use the batched distance computation
  if (is_ip) {
    InnerProductsBatch(query, centroids, num_centroids, dim, table);
  } else {
    L2DistancesBatch(query, centroids, num_centroids, dim, table);
  }
}

void EstimateDistancesBatch(const float* const* tables,
                             const uint32_t* codes,
                             uint32_t n, uint32_t num_segments,
                             float* distances) {
  // Table lookups are memory-bound, SIMD helps less here
  // Use scalar with prefetching
  constexpr uint32_t kPrefetchDistance = 4;
  
  for (uint32_t i = 0; i < n; ++i) {
    if (i + kPrefetchDistance < n) {
      Prefetch<0>(codes + (i + kPrefetchDistance) * num_segments);
    }
    
    float sum = 0.0f;
    for (uint32_t s = 0; s < num_segments; ++s) {
      const uint32_t code = codes[i * num_segments + s];
      sum += tables[s][code];
    }
    distances[i] = sum;
  }
}

void EstimateDistancesBatchFlat(const float* flat_table,
                                 const uint32_t* segment_offsets,
                                 const uint32_t* codes,
                                 uint32_t n, uint32_t num_segments,
                                 float* distances) {
  return scalar::EstimateDistancesBatchFlat(
      flat_table, segment_offsets, codes, n, num_segments, distances);
}

float HorizontalSum(const float* data, uint32_t n) {
#if defined(SAQ_HAVE_AVX512)
  if (HasAVX512() && n >= 16) {
    __m512 sum = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
      sum = _mm512_add_ps(sum, _mm512_loadu_ps(data + i));
    }
    float result = avx512::HorizontalSum512(sum);
    for (; i < n; ++i) result += data[i];
    return result;
  }
#endif
#if defined(SAQ_HAVE_AVX2) || defined(_MSC_VER)
  if (HasAVX2() && n >= 8) {
    __m256 sum = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(data + i));
    }
    float result = avx2::HorizontalSum256(sum);
    for (; i < n; ++i) result += data[i];
    return result;
  }
#endif
  return scalar::HorizontalSum(data, n);
}

float FindMin(const float* data, uint32_t n, uint32_t* min_idx) {
  // For small n, scalar is fine
  // For large n, SIMD min-reduce would help but index tracking is complex
  return scalar::FindMin(data, n, min_idx);
}

void FindKSmallest(const float* data, uint32_t n, uint32_t k,
                    float* values, uint32_t* indices) {
  return scalar::FindKSmallest(data, n, k, values, indices);
}

}  // namespace simd
}  // namespace saq
