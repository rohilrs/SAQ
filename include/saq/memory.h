#pragma once

#include <memory>
#include <stdint.h>

#include <cstdlib>
#include <cstring>

#include "saq/tools.h"

#ifdef _MSC_VER
#include <malloc.h>
#include <intrin.h>
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#else
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#endif

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace saq {

/// @brief Portable free for memory allocated by align_mm or _aligned_malloc.
/// On MSVC, _aligned_malloc must be paired with _aligned_free (not std::free).
inline void portable_aligned_free(void *p) {
    if (!p) return;
#ifdef _MSC_VER
    _aligned_free(p);
#else
    std::free(p);
#endif
}

template <size_t alignment, class T, bool HUGE_PAGE = false>
inline T *align_mm(size_t size) {
    size_t nbytes = rd_up_to_multiple_of(size * sizeof(T), alignment);
#ifdef _MSC_VER
    void *p = _aligned_malloc(nbytes, alignment);
#else
    void *p = std::aligned_alloc(alignment, nbytes);
#endif
#ifdef __linux__
    if (HUGE_PAGE) {
        madvise(p, nbytes, MADV_HUGEPAGE);
    }
#endif
    std::memset(p, 0, size * sizeof(T));
    return static_cast<T *>(p);
}

template <typename T, size_t alignment = 64>
struct AlignedAllocator {
    T *ptr = nullptr;
    size_t alignment_ = alignment;
    using value_type = T;
    T *allocate(size_t n) {
        size_t nbytes = rd_up_to_multiple_of(n * sizeof(T), alignment_);
#ifdef _MSC_VER
        return ptr = (T *)_aligned_malloc(nbytes, alignment_);
#else
        return ptr = (T *)std::aligned_alloc(alignment_, nbytes);
#endif
    }
    void deallocate(T *p, size_t) {
#ifdef _MSC_VER
        _aligned_free(p);
#else
        std::free(p);
#endif
        p = nullptr;
    }
    template <typename U>
    struct rebind {
        typedef AlignedAllocator<U, alignment> other;
    };
    // Converting constructor required by MSVC's std::vector internals
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, alignment> &) noexcept : alignment_(alignment) {}
    AlignedAllocator() noexcept : alignment_(alignment) {}

    bool operator!=(const AlignedAllocator &rhs) const { return alignment_ != rhs.alignment_; }
    bool operator==(const AlignedAllocator &rhs) const { return alignment_ == rhs.alignment_; }
};

template <typename T>
using UniqueArray = std::unique_ptr<T[], void (*)(void *)>;

/**
 * @brief Create a unique array with the specified size and alignment. The returned UniqueArray manages the
 *        lifetime of the allocated memory, freeing it using std::free when it goes out of scope.
 * @tparam T The type of the elements in the array.
 * @param size The number of elements in the array.
 * @param alignment The alignment requirement for the array (in bytes). 64 for AVX512.
 *                  If 0, thedefault alignment of std::malloc is used.
 */
template <typename T>
UniqueArray<T> make_unique_array(std::size_t size, std::size_t alignment = 0) {
    if (size == 0) {
        return UniqueArray<T>(nullptr, [](void *) {});
    }
    void *raw_ptr;
#ifdef _MSC_VER
    // MSVC: always use _aligned_malloc/_aligned_free pair to avoid heap mismatch.
    // When alignment is 0, use a default alignment of sizeof(void*).
    size_t actual_align = alignment ? alignment : sizeof(void *);
    raw_ptr = _aligned_malloc(rd_up_to_multiple_of(size * sizeof(T), actual_align), actual_align);
#else
    if (alignment) {
        raw_ptr = std::aligned_alloc(alignment, rd_up_to_multiple_of(size * sizeof(T), alignment));
    } else {
        raw_ptr = std::malloc(size * sizeof(T));
    }
#endif

    if (!raw_ptr) {
        throw std::bad_alloc();
    }
    return std::unique_ptr<T[], void (*)(void *)>(static_cast<T *>(raw_ptr), [](void *ptr) {
#ifdef _MSC_VER
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    });
}

static inline void prefetch_l1(const void *addr) {
#if defined(__SSE2__) || defined(_MSC_VER)
    _mm_prefetch((const char *)addr, _MM_HINT_T0);
#elif defined(__GNUC__)
    __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void *addr) {
#if defined(__SSE2__) || defined(_MSC_VER)
    _mm_prefetch((const char *)addr, _MM_HINT_T1);
#elif defined(__GNUC__)
    __builtin_prefetch(addr, 0, 2);
#endif
}

inline void mem_prefetch_l1(const char *ptr, size_t num_lines) {
    switch (num_lines) {
    default:
        [[fallthrough]];
    case 20:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 19:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 18:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 17:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 16:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 15:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 14:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 13:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 12:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 11:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 10:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 9:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 8:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 7:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 6:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 5:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 4:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 3:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 2:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 1:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 0:
        break;
    }
}

inline void mem_prefetch_l2(const char *ptr, size_t num_lines) {
    switch (num_lines) {
    default:
        [[fallthrough]];
    case 20:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 19:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 18:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 17:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 16:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 15:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 14:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 13:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 12:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 11:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 10:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 9:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 8:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 7:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 6:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 5:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 4:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 3:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 2:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 1:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 0:
        break;
    }
}
} // namespace saq
