# SAQ Sample Programs
#
# These samples demonstrate end-to-end usage of the SAQ library.

add_executable(saq_dbpedia_sample
  ${CMAKE_CURRENT_SOURCE_DIR}/saq_dbpedia_sample.cpp
)

target_link_libraries(saq_dbpedia_sample
  PRIVATE
    saq
)

target_include_directories(saq_dbpedia_sample
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Set working directory to project root for data paths
set_target_properties(saq_dbpedia_sample PROPERTIES
  VS_DEBUGGER_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

if(MSVC)
  target_compile_options(saq_dbpedia_sample PRIVATE /arch:AVX512)
else()
  target_compile_options(saq_dbpedia_sample PRIVATE -mavx512f -mavx512bw -mavx512dq -mavx512vl)
endif()
