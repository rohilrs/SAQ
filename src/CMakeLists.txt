target_sources(saq
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rotator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/caq_encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fast_scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lut.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cluster_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cluster_packer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/single_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/saq_quantizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/caq_estimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/saq_estimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/saq_searcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ivf_index.cpp
)

if(SAQ_BUILD_CUDA)
  # CUDA kernel sources (compiled by nvcc)
  target_sources(saq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_encoder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_packer.cu
  )
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_encoder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_packer.cu
    PROPERTIES LANGUAGE CUDA
  )
  # GPU orchestrator (compiled by host CXX — includes Eigen which is not CUDA-compatible)
  target_sources(saq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_ivf_construct.cpp
  )
  target_include_directories(saq PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
