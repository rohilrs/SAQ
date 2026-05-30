target_sources(saq
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bit_allocator_dp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bit_allocator_greedy.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_scatter.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_search.cu
  )
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_encoder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_scatter.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_search.cu
    PROPERTIES LANGUAGE CUDA
  )
  # GPU orchestrators (compiled by host CXX — include Eigen which is not CUDA-compatible)
  target_sources(saq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_ivf_construct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_ivf_search.cpp
  )
  target_include_directories(saq PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Preprocessing module (shared across all variants)
target_sources(saq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/preprocessing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/codebook_builder.cpp
)

if(SAQ_USE_FAISS)
    target_sources(saq PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/pca_faiss.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/kmeans_faiss.cpp
    )
else()
    target_sources(saq PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/pca_eigen.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/kmeans_eigen.cpp
    )
endif()
