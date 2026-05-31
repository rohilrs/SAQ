# tests/CMakeLists.txt
# SAQ C++ tests — plain standalone executables, no gtest dependency.
#
# Note: bit_allocation_dp_test.cpp, pca_projection_test.cpp, and
# saq_quantizer_test.cpp reference an obsolete API (SAQQuantizer,
# TransformQuery, WorkingDim) that was replaced when the codebase was
# ported to the reference IVF / SaqData pipeline. They are left in-tree
# for historical reference but intentionally not wired into the build.
#
# bit_allocation_dp_test.cpp additionally references a pre-scaffolded
# BitAllocatorDP::Allocate / ComputeDistortion / ApplyAllocation interface
# that is out of scope for Sub-Project 2 (only AllocateJoint exists today).
# Re-enable when those static helpers + non-joint Allocate path land.
# add_executable(bit_allocation_dp_test bit_allocation_dp_test.cpp)
# target_link_libraries(bit_allocation_dp_test PRIVATE saq)

add_executable(test_fit_decompress test_fit_decompress.cpp)
target_link_libraries(test_fit_decompress PRIVATE saq)

# Faiss from conda needs a newer libstdc++ than spack GCC 12 ships.
# Put conda's lib dir BEFORE the linker's default search so both link-time
# and run-time resolution find the newer libstdc++. Also switch to RUNPATH
# so LD_LIBRARY_PATH can override if needed.
if(DEFINED ENV{CONDA_PREFIX})
  target_link_directories(test_fit_decompress BEFORE PRIVATE $ENV{CONDA_PREFIX}/lib)
  target_link_options(test_fit_decompress PRIVATE "LINKER:--enable-new-dtags")
endif()

# The saq static library applies AVX-512 flags PRIVATE-ly, but consumers
# that transitively include fast_scan.h (via index/ivf_index.h) must also
# compile with AVX-512 enabled, otherwise the static_assert in
# transfer_lut_hacc fires. Mirror the root-level flag selection here.
if(SAQ_REQUIRE_AVX512)
  if(MSVC)
    target_compile_options(test_fit_decompress PRIVATE /arch:AVX512)
  else()
    target_compile_options(test_fit_decompress PRIVATE
      -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma)
  endif()
endif()

add_executable(test_codebook_builder codebook_builder_test.cpp)
target_link_libraries(test_codebook_builder PRIVATE saq)
if(DEFINED ENV{CONDA_PREFIX})
  target_link_directories(test_codebook_builder BEFORE PRIVATE $ENV{CONDA_PREFIX}/lib)
  target_link_options(test_codebook_builder PRIVATE "LINKER:--enable-new-dtags")
endif()

# test_codebook_builder now includes ivf_index.h (via TestNativeDerivationEndToEnd),
# which transitively pulls in fast_scan.h. Mirror the same AVX-512 flag block as
# test_fit_decompress so the static_assert in transfer_lut_hacc doesn't fire.
if(SAQ_REQUIRE_AVX512)
  if(MSVC)
    target_compile_options(test_codebook_builder PRIVATE /arch:AVX512)
  else()
    target_compile_options(test_codebook_builder PRIVATE
      -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma)
  endif()
endif()

add_executable(bit_allocator_greedy_test bit_allocator_greedy_test.cpp)
target_link_libraries(bit_allocator_greedy_test PRIVATE saq)
if(DEFINED ENV{CONDA_PREFIX})
  target_link_directories(bit_allocator_greedy_test BEFORE PRIVATE $ENV{CONDA_PREFIX}/lib)
  target_link_options(bit_allocator_greedy_test PRIVATE "LINKER:--enable-new-dtags")
endif()

# Manual-run benchmark: codebook init strategy speed + quality comparison.
add_executable(codebook_init_benchmark codebook_init_benchmark.cpp)
target_link_libraries(codebook_init_benchmark PRIVATE saq)
if(DEFINED ENV{CONDA_PREFIX})
  target_link_directories(codebook_init_benchmark BEFORE PRIVATE $ENV{CONDA_PREFIX}/lib)
  target_link_options(codebook_init_benchmark PRIVATE "LINKER:--enable-new-dtags")
endif()

# P4-T7/T8 benchmark: DP vs Greedy bit-allocators on dbpedia-100K.
add_executable(allocator_comparison_benchmark allocator_comparison_benchmark.cpp)
target_link_libraries(allocator_comparison_benchmark PRIVATE saq)
if(DEFINED ENV{CONDA_PREFIX})
  target_link_directories(allocator_comparison_benchmark BEFORE PRIVATE $ENV{CONDA_PREFIX}/lib)
  target_link_options(allocator_comparison_benchmark PRIVATE "LINKER:--enable-new-dtags")
endif()
if(SAQ_REQUIRE_AVX512)
  if(MSVC)
    target_compile_options(allocator_comparison_benchmark PRIVATE /arch:AVX512)
  else()
    target_compile_options(allocator_comparison_benchmark PRIVATE
      -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma)
  endif()
endif()
