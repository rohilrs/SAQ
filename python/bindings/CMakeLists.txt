pybind11_add_module(_saq_core saq_bindings.cpp)
target_link_libraries(_saq_core PRIVATE saq)
target_include_directories(_saq_core PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

# Propagate AVX-512 flags (saq target uses PRIVATE, so we need them here too)
if(SAQ_REQUIRE_AVX512)
  if(MSVC)
    target_compile_options(_saq_core PRIVATE /arch:AVX512)
  else()
    target_compile_options(_saq_core PRIVATE -mavx512f -mavx512bw -mavx512dq -mavx512vl)
  endif()
endif()

# Install to python/saq/ for development
set_target_properties(_saq_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/saq"
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/saq"
)

# Copy dependent DLLs next to the .pyd on Windows
if(WIN32)
  add_custom_command(TARGET _saq_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:glog::glog> "${PROJECT_SOURCE_DIR}/python/saq/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:fmt::fmt> "${PROJECT_SOURCE_DIR}/python/saq/"
    COMMENT "Copying glog and fmt DLLs to python/saq/"
  )
endif()
