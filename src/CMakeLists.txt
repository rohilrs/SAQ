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

# Preprocessing module (shared across all variants)
target_sources(saq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/preprocessing.cpp
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
