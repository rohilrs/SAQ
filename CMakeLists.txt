cmake_minimum_required(VERSION 3.16)

project(saq LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SAQ_BUILD_SAMPLES "Build SAQ sample programs" ON)

add_library(saq STATIC)
target_include_directories(saq
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

if(SAQ_BUILD_SAMPLES)
  add_executable(saq_sample sample/saq_sample.cpp)
  target_link_libraries(saq_sample PRIVATE saq)
endif()
