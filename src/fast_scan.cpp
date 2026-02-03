/// @file fast_scan.cpp
/// @brief FastScan SIMD-accelerated distance table lookup implementation.

#include "index/fast_scan/fast_scan.h"
#include "saq/simd_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace saq {

// ============================================================================
// Code Packing (4-bit)
// ============================================================================

bool PackCodes4bit(const uint32_t* codes, uint32_t n_vectors,
                    uint32_t num_segments, FastScanCodes& output) {
  if (codes == nullptr || n_vectors == 0 || num_segments == 0) {
    return false;
  }
  
  // Validate codes are in range [0, 15]
  for (uint32_t i = 0; i < n_vectors * num_segments; ++i) {
    if (codes[i] >= 16) {
      return false;  // Code too large for 4-bit
    }
  }
  
  output.num_vectors = n_vectors;
  output.num_segments = num_segments;
  output.bits_per_code = 4;
  output.block_size = kFastScanBlockSize;
  
  const uint32_t num_blocks = output.NumBlocks();
  const uint32_t padded_vectors = num_blocks * kFastScanBlockSize;
  
  // For 4-bit: pack 2 codes per byte
  // Layout: for each block, for each segment pair, 32 bytes (32 vectors × 2 segments / 2)
  // Simplified: segment-major within block for better SIMD access
  const size_t bytes_per_block = static_cast<size_t>(kFastScanBlockSize) * num_segments / 2;
  output.data.resize(num_blocks * bytes_per_block, 0);
  
  for (uint32_t block = 0; block < num_blocks; ++block) {
    const uint32_t base_vec = block * kFastScanBlockSize;
    uint8_t* block_ptr = output.data.data() + block * bytes_per_block;
    
    // Pack codes segment by segment within block
    for (uint32_t seg = 0; seg < num_segments; seg += 2) {
      for (uint32_t v = 0; v < kFastScanBlockSize; ++v) {
        const uint32_t vec_idx = base_vec + v;
        
        uint8_t lo = 0, hi = 0;
        if (vec_idx < n_vectors) {
          lo = static_cast<uint8_t>(codes[vec_idx * num_segments + seg]);
          if (seg + 1 < num_segments) {
            hi = static_cast<uint8_t>(codes[vec_idx * num_segments + seg + 1]);
          }
        }
        
        // Pack: lo in lower nibble, hi in upper nibble
        const size_t offset = static_cast<size_t>(seg / 2) * kFastScanBlockSize + v;
        block_ptr[offset] = (hi << 4) | lo;
      }
    }
  }
  
  return true;
}

// ============================================================================
// Code Packing (8-bit)
// ============================================================================

bool PackCodes8bit(const uint32_t* codes, uint32_t n_vectors,
                    uint32_t num_segments, FastScanCodes& output) {
  if (codes == nullptr || n_vectors == 0 || num_segments == 0) {
    return false;
  }
  
  // Validate codes are in range [0, 255]
  for (uint32_t i = 0; i < n_vectors * num_segments; ++i) {
    if (codes[i] >= 256) {
      return false;
    }
  }
  
  output.num_vectors = n_vectors;
  output.num_segments = num_segments;
  output.bits_per_code = 8;
  output.block_size = kFastScanBlockSize;
  
  const uint32_t num_blocks = output.NumBlocks();
  const size_t bytes_per_block = static_cast<size_t>(kFastScanBlockSize) * num_segments;
  output.data.resize(num_blocks * bytes_per_block, 0);
  
  for (uint32_t block = 0; block < num_blocks; ++block) {
    const uint32_t base_vec = block * kFastScanBlockSize;
    uint8_t* block_ptr = output.data.data() + block * bytes_per_block;
    
    // Pack codes segment by segment
    for (uint32_t seg = 0; seg < num_segments; ++seg) {
      for (uint32_t v = 0; v < kFastScanBlockSize; ++v) {
        const uint32_t vec_idx = base_vec + v;
        uint8_t code = 0;
        if (vec_idx < n_vectors) {
          code = static_cast<uint8_t>(codes[vec_idx * num_segments + seg]);
        }
        block_ptr[seg * kFastScanBlockSize + v] = code;
      }
    }
  }
  
  return true;
}

// ============================================================================
// Code Unpacking
// ============================================================================

bool UnpackCodes(const FastScanCodes& packed, uint32_t* codes) {
  if (codes == nullptr) {
    return false;
  }
  
  const uint32_t num_blocks = packed.NumBlocks();
  
  if (packed.bits_per_code == 4) {
    const size_t bytes_per_block = static_cast<size_t>(packed.block_size) * 
                                    packed.num_segments / 2;
    
    for (uint32_t block = 0; block < num_blocks; ++block) {
      const uint32_t base_vec = block * packed.block_size;
      const uint8_t* block_ptr = packed.data.data() + block * bytes_per_block;
      
      for (uint32_t seg = 0; seg < packed.num_segments; seg += 2) {
        for (uint32_t v = 0; v < packed.block_size; ++v) {
          const uint32_t vec_idx = base_vec + v;
          if (vec_idx >= packed.num_vectors) continue;
          
          const size_t offset = static_cast<size_t>(seg / 2) * packed.block_size + v;
          const uint8_t byte = block_ptr[offset];
          
          codes[vec_idx * packed.num_segments + seg] = byte & 0x0F;
          if (seg + 1 < packed.num_segments) {
            codes[vec_idx * packed.num_segments + seg + 1] = byte >> 4;
          }
        }
      }
    }
  } else {
    const size_t bytes_per_block = static_cast<size_t>(packed.block_size) * 
                                    packed.num_segments;
    
    for (uint32_t block = 0; block < num_blocks; ++block) {
      const uint32_t base_vec = block * packed.block_size;
      const uint8_t* block_ptr = packed.data.data() + block * bytes_per_block;
      
      for (uint32_t seg = 0; seg < packed.num_segments; ++seg) {
        for (uint32_t v = 0; v < packed.block_size; ++v) {
          const uint32_t vec_idx = base_vec + v;
          if (vec_idx >= packed.num_vectors) continue;
          
          codes[vec_idx * packed.num_segments + seg] = 
              block_ptr[seg * packed.block_size + v];
        }
      }
    }
  }
  
  return true;
}

// ============================================================================
// LUT Packing
// ============================================================================

bool PackLUT4bit(const float* const* tables, uint32_t num_segments,
                  uint32_t centroids_per_segment, FastScanLUT& output) {
  if (tables == nullptr || num_segments == 0 || centroids_per_segment > 16) {
    return false;
  }
  
  output.num_segments = num_segments;
  
  // Find min/max across all tables for quantization
  float min_val = std::numeric_limits<float>::max();
  float max_val = std::numeric_limits<float>::lowest();
  
  for (uint32_t s = 0; s < num_segments; ++s) {
    for (uint32_t c = 0; c < centroids_per_segment; ++c) {
      min_val = std::min(min_val, tables[s][c]);
      max_val = std::max(max_val, tables[s][c]);
    }
  }
  
  // Compute scale and bias for uint8 quantization
  const float range = max_val - min_val;
  if (range < 1e-10f) {
    output.scale = 1.0f;
    output.bias = min_val;
  } else {
    output.scale = range / 255.0f;
    output.bias = min_val;
  }
  
  // Pack: 16 entries per segment (pad with large distances), aligned to 16 bytes
  // Layout: [seg0: 16 bytes][seg1: 16 bytes]...
  output.data.resize(static_cast<size_t>(num_segments) * 16);
  
  // Use max value for padding (so unused codes have high distance)
  const uint8_t pad_value = 255;
  
  for (uint32_t s = 0; s < num_segments; ++s) {
    // Fill actual centroids
    for (uint32_t c = 0; c < centroids_per_segment; ++c) {
      const float normalized = (tables[s][c] - output.bias) / output.scale;
      output.data[s * 16 + c] = static_cast<uint8_t>(
          std::min(255.0f, std::max(0.0f, normalized + 0.5f)));
    }
    // Pad remaining slots
    for (uint32_t c = centroids_per_segment; c < 16; ++c) {
      output.data[s * 16 + c] = pad_value;
    }
  }
  
  return true;
}

bool PackLUT4bitVariable(const float* const* tables, uint32_t num_segments,
                          const uint32_t* centroids_per_segment, FastScanLUT& output) {
  if (tables == nullptr || num_segments == 0 || centroids_per_segment == nullptr) {
    return false;
  }
  
  // Validate all segments have ≤16 centroids
  for (uint32_t s = 0; s < num_segments; ++s) {
    if (centroids_per_segment[s] > 16 || centroids_per_segment[s] == 0) {
      return false;
    }
  }
  
  output.num_segments = num_segments;
  
  // Find min/max across all tables for quantization
  float min_val = std::numeric_limits<float>::max();
  float max_val = std::numeric_limits<float>::lowest();
  
  for (uint32_t s = 0; s < num_segments; ++s) {
    for (uint32_t c = 0; c < centroids_per_segment[s]; ++c) {
      min_val = std::min(min_val, tables[s][c]);
      max_val = std::max(max_val, tables[s][c]);
    }
  }
  
  // Compute scale and bias for uint8 quantization
  const float range = max_val - min_val;
  if (range < 1e-10f) {
    output.scale = 1.0f;
    output.bias = min_val;
  } else {
    output.scale = range / 255.0f;
    output.bias = min_val;
  }
  
  // Pack: 16 entries per segment (pad with large distances), aligned to 16 bytes
  output.data.resize(static_cast<size_t>(num_segments) * 16);
  
  // Use max value for padding
  const uint8_t pad_value = 255;
  
  for (uint32_t s = 0; s < num_segments; ++s) {
    // Fill actual centroids
    for (uint32_t c = 0; c < centroids_per_segment[s]; ++c) {
      const float normalized = (tables[s][c] - output.bias) / output.scale;
      output.data[s * 16 + c] = static_cast<uint8_t>(
          std::min(255.0f, std::max(0.0f, normalized + 0.5f)));
    }
    // Pad remaining slots with high value
    for (uint32_t c = centroids_per_segment[s]; c < 16; ++c) {
      output.data[s * 16 + c] = pad_value;
    }
  }
  
  return true;
}

bool PackLUT8bit(const float* const* tables, uint32_t num_segments,
                  FastScanLUT& output) {
  // This version assumes all 256 centroids - use PackLUT8bitVariable for variable counts
  std::vector<uint32_t> centroids(num_segments, 256);
  return PackLUT8bitVariable(tables, num_segments, centroids.data(), output);
}

bool PackLUT8bitVariable(const float* const* tables, uint32_t num_segments,
                          const uint32_t* centroids_per_segment, FastScanLUT& output) {
  if (tables == nullptr || num_segments == 0 || centroids_per_segment == nullptr) {
    return false;
  }
  
  output.num_segments = num_segments;
  
  // Find min/max across all valid entries
  float min_val = std::numeric_limits<float>::max();
  float max_val = std::numeric_limits<float>::lowest();
  
  for (uint32_t s = 0; s < num_segments; ++s) {
    for (uint32_t c = 0; c < centroids_per_segment[s]; ++c) {
      min_val = std::min(min_val, tables[s][c]);
      max_val = std::max(max_val, tables[s][c]);
    }
  }
  
  const float range = max_val - min_val;
  if (range < 1e-10f) {
    output.scale = 1.0f;
    output.bias = min_val;
  } else {
    output.scale = range / 255.0f;
    output.bias = min_val;
  }
  
  // Pack: 256 entries per segment, pad unused with high value
  output.data.resize(static_cast<size_t>(num_segments) * 256);
  const uint8_t pad_value = 255;
  
  for (uint32_t s = 0; s < num_segments; ++s) {
    // Fill actual centroids
    for (uint32_t c = 0; c < centroids_per_segment[s]; ++c) {
      const float normalized = (tables[s][c] - output.bias) / output.scale;
      output.data[s * 256 + c] = static_cast<uint8_t>(
          std::min(255.0f, std::max(0.0f, normalized + 0.5f)));
    }
    // Pad remaining slots
    for (uint32_t c = centroids_per_segment[s]; c < 256; ++c) {
      output.data[s * 256 + c] = pad_value;
    }
  }
  
  return true;
}

// ============================================================================
// FastScan Distance Estimation (4-bit) - AVX2/AVX-512
// ============================================================================

#if defined(__AVX2__) || defined(_MSC_VER)

void FastScanEstimate4bit_AVX2(const FastScanCodes& packed_codes,
                                 const FastScanLUT& packed_lut,
                                 float* distances) {
  const uint32_t num_blocks = packed_codes.NumBlocks();
  const uint32_t num_segments = packed_codes.num_segments;
  const size_t bytes_per_block = packed_codes.BytesPerBlock();
  
  const float scale = packed_lut.scale;
  const float bias = packed_lut.bias * static_cast<float>(num_segments);
  
  // Process each block of 32 vectors
  for (uint32_t block = 0; block < num_blocks; ++block) {
    const uint32_t base_vec = block * kFastScanBlockSize;
    const uint8_t* codes_ptr = packed_codes.data.data() + block * bytes_per_block;
    
    // Accumulators for 32 vectors (split into 4 groups of 8 for AVX2)
    __m256i accum0 = _mm256_setzero_si256();
    __m256i accum1 = _mm256_setzero_si256();
    __m256i accum2 = _mm256_setzero_si256();
    __m256i accum3 = _mm256_setzero_si256();
    
    // Process segments in pairs (4-bit: 2 codes per byte)
    for (uint32_t seg = 0; seg < num_segments; seg += 2) {
      // Load LUT for this segment pair (32 bytes = 2 × 16 entries)
      const __m256i lut = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(packed_lut.data.data() + seg * 16));
      
      // Load 32 packed codes (32 bytes)
      const __m256i codes = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(codes_ptr + (seg / 2) * kFastScanBlockSize));
      
      // Extract low and high nibbles
      const __m256i mask = _mm256_set1_epi8(0x0F);
      const __m256i lo = _mm256_and_si256(codes, mask);
      const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), mask);
      
      // Shuffle lookup for low nibble (segment seg)
      // vpshufb does 16-way parallel lookup within each 128-bit lane
      __m256i lut_lo = _mm256_permute2x128_si256(lut, lut, 0x00);  // Broadcast low 128 bits
      __m256i result_lo = _mm256_shuffle_epi8(lut_lo, lo);
      
      // Shuffle lookup for high nibble (segment seg+1)
      __m256i lut_hi = _mm256_permute2x128_si256(lut, lut, 0x11);  // Broadcast high 128 bits
      __m256i result_hi = _mm256_shuffle_epi8(lut_hi, hi);
      
      // Accumulate as 16-bit to avoid overflow
      // Split into 4 groups, widen to 16-bit, accumulate
      __m256i sum = _mm256_add_epi8(result_lo, result_hi);
      
      // Unpack and accumulate to 16-bit accumulators
      __m256i zero = _mm256_setzero_si256();
      __m256i sum_lo = _mm256_unpacklo_epi8(sum, zero);
      __m256i sum_hi = _mm256_unpackhi_epi8(sum, zero);
      
      // Add to accumulators (treating as 16-bit integers)
      accum0 = _mm256_add_epi16(accum0, _mm256_unpacklo_epi16(sum_lo, zero));
      accum1 = _mm256_add_epi16(accum1, _mm256_unpackhi_epi16(sum_lo, zero));
      accum2 = _mm256_add_epi16(accum2, _mm256_unpacklo_epi16(sum_hi, zero));
      accum3 = _mm256_add_epi16(accum3, _mm256_unpackhi_epi16(sum_hi, zero));
    }
    
    // Convert accumulated uint16 sums to float distances
    // Store results for vectors in this block
    alignas(32) int32_t sums[32];
    _mm256_store_si256(reinterpret_cast<__m256i*>(sums + 0), accum0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(sums + 8), accum1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(sums + 16), accum2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(sums + 24), accum3);
    
    for (uint32_t v = 0; v < kFastScanBlockSize; ++v) {
      const uint32_t vec_idx = base_vec + v;
      if (vec_idx < packed_codes.num_vectors) {
        distances[vec_idx] = static_cast<float>(sums[v]) * scale + bias;
      }
    }
  }
}

#endif  // __AVX2__

// Scalar fallback
void FastScanEstimate4bit_Scalar(const FastScanCodes& packed_codes,
                                   const FastScanLUT& packed_lut,
                                   float* distances) {
  const uint32_t num_blocks = packed_codes.NumBlocks();
  const uint32_t num_segments = packed_codes.num_segments;
  const size_t bytes_per_block = packed_codes.BytesPerBlock();
  
  const float scale = packed_lut.scale;
  const float bias = packed_lut.bias * static_cast<float>(num_segments);
  
  for (uint32_t block = 0; block < num_blocks; ++block) {
    const uint32_t base_vec = block * kFastScanBlockSize;
    const uint8_t* codes_ptr = packed_codes.data.data() + block * bytes_per_block;
    
    for (uint32_t v = 0; v < kFastScanBlockSize; ++v) {
      const uint32_t vec_idx = base_vec + v;
      if (vec_idx >= packed_codes.num_vectors) continue;
      
      uint32_t sum = 0;
      for (uint32_t seg = 0; seg < num_segments; seg += 2) {
        const size_t offset = static_cast<size_t>(seg / 2) * kFastScanBlockSize + v;
        const uint8_t byte = codes_ptr[offset];
        const uint8_t code_lo = byte & 0x0F;
        const uint8_t code_hi = byte >> 4;
        
        sum += packed_lut.data[seg * 16 + code_lo];
        if (seg + 1 < num_segments) {
          sum += packed_lut.data[(seg + 1) * 16 + code_hi];
        }
      }
      
      distances[vec_idx] = static_cast<float>(sum) * scale + bias;
    }
  }
}

void FastScanEstimate4bit(const FastScanCodes& packed_codes,
                           const FastScanLUT& packed_lut,
                           float* distances) {
  // Use scalar implementation for correctness
  // AVX2 shuffle-based implementation has lane-crossing issues that need more work
  FastScanEstimate4bit_Scalar(packed_codes, packed_lut, distances);
}

// ============================================================================
// FastScan Distance Estimation (8-bit)
// ============================================================================

void FastScanEstimate8bit(const FastScanCodes& packed_codes,
                           const FastScanLUT& packed_lut,
                           float* distances) {
  // For 8-bit codes, we use gather instructions or scalar
  // 8-bit lookup is less amenable to shuffle-based optimization
  const uint32_t num_blocks = packed_codes.NumBlocks();
  const uint32_t num_segments = packed_codes.num_segments;
  const size_t bytes_per_block = packed_codes.BytesPerBlock();
  
  const float scale = packed_lut.scale;
  const float bias = packed_lut.bias * static_cast<float>(num_segments);
  
  for (uint32_t block = 0; block < num_blocks; ++block) {
    const uint32_t base_vec = block * kFastScanBlockSize;
    const uint8_t* codes_ptr = packed_codes.data.data() + block * bytes_per_block;
    
    for (uint32_t v = 0; v < kFastScanBlockSize; ++v) {
      const uint32_t vec_idx = base_vec + v;
      if (vec_idx >= packed_codes.num_vectors) continue;
      
      uint32_t sum = 0;
      for (uint32_t seg = 0; seg < num_segments; ++seg) {
        const uint8_t code = codes_ptr[seg * kFastScanBlockSize + v];
        sum += packed_lut.data[seg * 256 + code];
      }
      
      distances[vec_idx] = static_cast<float>(sum) * scale + bias;
    }
  }
}

// ============================================================================
// FastScan with Selection Mask
// ============================================================================

void FastScanEstimateSelected4bit(const FastScanCodes& packed_codes,
                                   const FastScanLUT& packed_lut,
                                   const uint8_t* mask,
                                   float* distances) {
  // Simple implementation - skip masked vectors
  const uint32_t num_blocks = packed_codes.NumBlocks();
  const uint32_t num_segments = packed_codes.num_segments;
  const size_t bytes_per_block = packed_codes.BytesPerBlock();
  
  const float scale = packed_lut.scale;
  const float bias = packed_lut.bias * static_cast<float>(num_segments);
  
  for (uint32_t block = 0; block < num_blocks; ++block) {
    const uint32_t base_vec = block * kFastScanBlockSize;
    const uint8_t* codes_ptr = packed_codes.data.data() + block * bytes_per_block;
    
    for (uint32_t v = 0; v < kFastScanBlockSize; ++v) {
      const uint32_t vec_idx = base_vec + v;
      if (vec_idx >= packed_codes.num_vectors) continue;
      if (!mask[vec_idx]) {
        distances[vec_idx] = std::numeric_limits<float>::max();
        continue;
      }
      
      uint32_t sum = 0;
      for (uint32_t seg = 0; seg < num_segments; seg += 2) {
        const size_t offset = static_cast<size_t>(seg / 2) * kFastScanBlockSize + v;
        const uint8_t byte = codes_ptr[offset];
        
        sum += packed_lut.data[seg * 16 + (byte & 0x0F)];
        if (seg + 1 < num_segments) {
          sum += packed_lut.data[(seg + 1) * 16 + (byte >> 4)];
        }
      }
      
      distances[vec_idx] = static_cast<float>(sum) * scale + bias;
    }
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

bool CanUseFastScan(const uint32_t* centroids_per_segment,
                     uint32_t num_segments) {
  if (centroids_per_segment == nullptr || num_segments == 0) {
    return false;
  }
  
  // Check if all segments fit within FastScan constraints
  // For 4-bit FastScan: all segments must have ≤16 centroids
  // For 8-bit FastScan: all segments must have ≤256 centroids
  uint32_t max_centroids = 0;
  for (uint32_t i = 0; i < num_segments; ++i) {
    max_centroids = std::max(max_centroids, centroids_per_segment[i]);
    // Must have at least 2 centroids (1 bit)
    if (centroids_per_segment[i] < 2) {
      return false;
    }
  }
  
  // Check if compatible with 4-bit or 8-bit FastScan
  return max_centroids <= 256;
}

uint32_t RecommendedFastScanBits(uint32_t max_centroids) {
  if (max_centroids <= 16) return 4;
  if (max_centroids <= 256) return 8;
  return 0;  // Not supported
}

}  // namespace saq
