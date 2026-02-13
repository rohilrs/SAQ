cmake_minimum_required(VERSION 3.16)

project(saq LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SAQ_BUILD_SAMPLES "Build SAQ sample programs" ON)
option(SAQ_REQUIRE_AVX512 "Require AVX-512 support" ON)
option(SAQ_BUILD_TESTS "Build SAQ test programs" OFF)
option(SAQ_USE_OPENMP "Enable OpenMP parallelization" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
  taocpp-json
  GIT_REPOSITORY https://github.com/taocpp/json.git
  GIT_TAG        1.0.0-beta.14
)
FetchContent_MakeAvailable(taocpp-json)

add_library(saq STATIC)
target_include_directories(saq
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(saq
  PUBLIC
    taocpp::json
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
