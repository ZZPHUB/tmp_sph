cmake_minimum_required(VERSION 3.8)
project(sph)
 
include_directories("./header/")
find_package(CUDA REQUIRED)
file (GLOB_RECURSE files  *.cu */*.cu)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
 
cuda_add_executable(sph ${files})
target_link_libraries(sph ${CUDA_LIBRARIES})
