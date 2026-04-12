cmake_minimum_required(VERSION 3.16)
# Allow older FetchContent deps (gflags) that use cmake_minimum_required < 3.5
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

project(saq LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SAQ_BUILD_SAMPLES "Build SAQ sample programs" ON)
option(SAQ_REQUIRE_AVX512 "Require AVX-512 support" ON)
option(SAQ_BUILD_TESTS "Build SAQ test programs" OFF)
option(SAQ_USE_OPENMP "Enable OpenMP parallelization" ON)
option(SAQ_BUILD_PYTHON "Build Python bindings via pybind11" OFF)
option(SAQ_BUILD_CUDA "Build CUDA GPU acceleration" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)

# gflags (needed by glog)
set(GFLAGS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(GFLAGS_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GFLAGS_BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(GFLAGS_BUILD_gflags_LIB ON CACHE BOOL "" FORCE)
set(GFLAGS_BUILD_gflags_nothreads_LIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG        v2.2.2
  GIT_SHALLOW    TRUE
)

# glog
set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
set(WITH_GTEST OFF CACHE BOOL "" FORCE)
set(WITH_UNWIND OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG        v0.7.1
  GIT_SHALLOW    TRUE
)

# fmt
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        10.2.1
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(eigen gflags glog fmt)

if(SAQ_BUILD_PYTHON)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

add_library(saq STATIC)
set_target_properties(saq PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(saq
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(saq
  PUBLIC
    Eigen3::Eigen
    glog::glog
    gflags_static
    fmt::fmt
)

if(SAQ_REQUIRE_AVX512)
  if(MSVC)
    target_compile_options(saq PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
  else()
    # GCC/Clang need -mfma alongside -mavx512f (Eigen requirement)
    target_compile_options(saq PRIVATE -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma)
  endif()
  target_compile_definitions(saq PUBLIC SAQ_REQUIRE_AVX512=1)
endif()

if(SAQ_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(saq PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(saq PUBLIC SAQ_USE_OPENMP=1)
    if(MSVC)
      target_compile_options(saq PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>)
    endif()
    message(STATUS "OpenMP enabled (version: ${OpenMP_CXX_VERSION})")
  else()
    message(WARNING "OpenMP not found, building without parallelization")
  endif()
endif()

if(SAQ_BUILD_CUDA)
  # Allow unsupported host compiler (VS 2025 / Build Tools 18)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # A100=80, RTX3090=86, RTX4090=89, H100=90, B200=100, RTX5090=120
  # Use -real to produce CUBIN only (avoids PTX "unsupported toolchain" rejection)
  set(CMAKE_CUDA_ARCHITECTURES "80-real;86-real;89-real;90-real;100-real;120-real" CACHE STRING "CUDA architectures")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(saq PUBLIC CUDA::cublas CUDA::cudart)
  target_compile_definitions(saq PUBLIC SAQ_USE_CUDA=1)
  message(STATUS "CUDA enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

# ---- Faiss / Eigen preprocessing dispatch ----
if(WIN32)
    set(SAQ_USE_FAISS_DEFAULT OFF)
else()
    set(SAQ_USE_FAISS_DEFAULT ON)
endif()
option(SAQ_USE_FAISS "Use Faiss for PCA and k-means (Linux default ON, Windows default OFF)"
       ${SAQ_USE_FAISS_DEFAULT})

if(SAQ_USE_FAISS)
    find_package(faiss REQUIRED)
    target_link_libraries(saq PRIVATE faiss)
    target_compile_definitions(saq PRIVATE SAQ_USE_FAISS)
    message(STATUS "SAQ_USE_FAISS=ON: using Faiss for PCA and k-means")
else()
    message(STATUS "SAQ_USE_FAISS=OFF: using Eigen BDCSVD fallback for PCA and k-means")
endif()

# gpu-codebook branch: codebook API surface is present but not yet wired
target_compile_definitions(saq PRIVATE SAQ_ENABLE_CODEBOOK)

if(SAQ_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(SAQ_BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

add_subdirectory(src)

if(SAQ_BUILD_PYTHON)
  add_subdirectory(python/bindings)
endif()
