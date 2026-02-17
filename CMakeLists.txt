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
FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG        v2.2.2
  GIT_SHALLOW    TRUE
)

# glog
set(WITH_GFLAGS ON CACHE BOOL "" FORCE)
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
    target_compile_options(saq PRIVATE /arch:AVX512)
  else()
    target_compile_options(saq PRIVATE -mavx512f -mavx512bw -mavx512dq -mavx512vl)
  endif()
  target_compile_definitions(saq PUBLIC SAQ_REQUIRE_AVX512=1)
endif()

if(SAQ_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(saq PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(saq PUBLIC SAQ_USE_OPENMP=1)
    if(MSVC)
      target_compile_options(saq PRIVATE /openmp:llvm)
    endif()
    message(STATUS "OpenMP enabled (version: ${OpenMP_CXX_VERSION})")
  else()
    message(WARNING "OpenMP not found, building without parallelization")
  endif()
endif()

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
