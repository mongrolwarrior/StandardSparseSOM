cmake_minimum_required(VERSION 3.18)
project(StandardSparseSOM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to Ada (RTX 4090, sm_89); override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(standardsparsesom src/main.cu)
target_include_directories(standardsparsesom PRIVATE include)
target_link_libraries(standardsparsesom PRIVATE CUDA::cudart CUDA::cusparse)
