pybind11_add_module(_saq_core saq_bindings.cpp)
target_link_libraries(_saq_core PRIVATE saq)
target_include_directories(_saq_core PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

# Install to python/saq/ for development
set_target_properties(_saq_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/saq"
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/saq"
)
