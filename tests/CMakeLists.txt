# tests/CMakeLists.txt
# SAQ C++ tests — plain standalone executables, no gtest dependency.
#
# Note: bit_allocation_dp_test.cpp, pca_projection_test.cpp, and
# saq_quantizer_test.cpp reference an obsolete API (SAQQuantizer,
# TransformQuery, WorkingDim) that was replaced when the codebase was
# ported to the reference IVF / SaqData pipeline. They are left in-tree
# for historical reference but intentionally not wired into the build.

add_executable(test_fit_decompress test_fit_decompress.cpp)
target_link_libraries(test_fit_decompress PRIVATE saq)

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
