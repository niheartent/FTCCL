cmake_minimum_required(VERSION 3.18)
project(ftccl_midware LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(FTCCL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(FTCCl_NCCL_HOME "${FTCCL_REPO_ROOT}/ftccl_lib/build" CACHE PATH "FTCCL/NCCL install prefix")
set(CUDA_HOME "/usr/local/cuda" CACHE PATH "CUDA install prefix")

add_library(ftccl_midware SHARED
  src/ftccl_midware.cu
)
target_include_directories(ftccl_midware PRIVATE
  ${FTCCl_NCCL_HOME}/include
  ${CUDA_HOME}/include
)
target_link_directories(ftccl_midware PRIVATE
  ${FTCCl_NCCL_HOME}/lib
  ${FTCCl_NCCL_HOME}/lib64
  ${CUDA_HOME}/lib64
)
target_link_libraries(ftccl_midware PRIVATE nccl cudart dl pthread)
set_target_properties(ftccl_midware PROPERTIES
  OUTPUT_NAME "ftccl_midware"
  BUILD_RPATH "${FTCCl_NCCL_HOME}/lib;${FTCCl_NCCL_HOME}/lib64;${CUDA_HOME}/lib64"
)

add_executable(ftccl_allreduce_smoke
  tests/ftccl_allreduce_smoke.cu
)
target_include_directories(ftccl_allreduce_smoke PRIVATE
  ${FTCCl_NCCL_HOME}/include
  ${CUDA_HOME}/include
)
target_link_directories(ftccl_allreduce_smoke PRIVATE
  ${FTCCl_NCCL_HOME}/lib
  ${FTCCl_NCCL_HOME}/lib64
  ${CUDA_HOME}/lib64
)
target_link_libraries(ftccl_allreduce_smoke PRIVATE nccl cudart)
set_target_properties(ftccl_allreduce_smoke PROPERTIES
  BUILD_RPATH "${FTCCl_NCCL_HOME}/lib;${FTCCl_NCCL_HOME}/lib64;${CUDA_HOME}/lib64"
)
