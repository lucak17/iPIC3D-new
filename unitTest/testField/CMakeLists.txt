cmake_minimum_required(VERSION 3.18)
project(CudaTestField LANGUAGES CXX CUDA)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# first, try the new CUDA Toolkit package
find_package(CUDAToolkit QUIET)

# Build test executable
add_executable(testField
  testField.cu
)

# include your headers
target_include_directories(testField PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/buffers
)

# link CUDA runtime, trying both possible imported targets
if (TARGET CUDAToolkit::cudart)
  target_link_libraries(testField PRIVATE
    CUDAToolkit::cudart
  )
else()
  # fallback to the classic FindCUDA
  find_package(CUDA REQUIRED)
  target_link_libraries(testField PRIVATE
    CUDA::cudart
  )
endif()

enable_testing()
add_test(NAME FieldTests COMMAND testField)
