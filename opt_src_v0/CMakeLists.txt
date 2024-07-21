cmake_minimum_required(VERSION 3.8)
project(sph)
 
include_directories("./header/")
find_package(CUDA REQUIRED)
file (GLOB_RECURSE files  *.cu */*.cu)
set(CUDA_NVCC_FLAGS -rdc=true)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
#include_directories("/home/zzp/workspace/temp/sphio/src")
#link_directories("/home/zzp/workspace/temp/sphio/build")
set(CUDA_SEPARABLE_COMPILATION ON)
 
cuda_add_executable(sph ${files})
target_link_libraries(sph ${CUDA_LIBRARIES})
