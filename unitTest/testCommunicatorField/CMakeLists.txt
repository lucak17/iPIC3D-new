cmake_minimum_required(VERSION 3.22)  # Minimum CMake version
project(testCommunicatorField LANGUAGES CXX CUDA)


# Check if compilers are not already set and provide defaults
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER cc)
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER CC)
endif()

# first, try the new CUDA Toolkit package
find_package(CUDAToolkit QUIET)


# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find MPI
find_package(MPI REQUIRED)

# Default to Release build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Optimization flags for each build type
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")                 # Debug: No optimization, debug info
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")         # Release: High optimization

# Build test executable
add_executable(${PROJECT_NAME}
  testCommunicatorField.cpp
)

# include your headers
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/buffers
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/mpiCommunication
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

# manually set MPI 
set(MPI_mpi_cray_LIBRARY /opt/cray/pe/mpich/8.1.28/ofi/cray/17.0/lib/libmpi_cray.so)
set(Cray_xpmem_LIBRARY /opt/cray/xpmem/2.8.2-1.0_3.9__g84a27a5.shasta/lib64/libxpmem.so)

include_directories(/opt/cray/pe/mpich/8.1.28/ofi/cray/17.0/include)
target_link_libraries(${PROJECT_NAME} PUBLIC ${MPI_mpi_cray_LIBRARY} ${Cray_xpmem_LIBRARY})



# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC include)