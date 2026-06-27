cmake_minimum_required(VERSION 3.18)
get_filename_component(FTCCL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(HIP_PATH "$ENV{HIP_PATH}" CACHE PATH "HIP install prefix")
if(NOT HIP_PATH)
  set(HIP_PATH "/public/software/compiler/dtk-26.04/hip")
endif()
find_program(HIPCC_EXECUTABLE hipcc HINTS "${HIP_PATH}/bin")
if(NOT HIPCC_EXECUTABLE)
  message(FATAL_ERROR "hipcc not found. Load DTK first, e.g. module load compiler/dtk/26.04")
endif()
set(CMAKE_CXX_COMPILER "${HIPCC_EXECUTABLE}")

project(ftccl_midware LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RCCL_ROOT "${FTCCL_REPO_ROOT}/custom-rccl-dtk-26.04/build" CACHE PATH "Custom RCCL/FTCCL build prefix")

find_library(RCCL_LIBRARY
  NAMES rccl nccl
  HINTS "${RCCL_ROOT}" "${RCCL_ROOT}/lib" "${RCCL_ROOT}/lib64"
  NO_DEFAULT_PATH
)
if(NOT RCCL_LIBRARY)
  message(FATAL_ERROR "Could not find librccl/libnccl under RCCL_ROOT=${RCCL_ROOT}")
endif()

find_library(HIP_RUNTIME_LIBRARY
  NAMES amdhip64
  HINTS "${HIP_PATH}/lib" "${HIP_PATH}/lib64" "${HIP_PATH}/../lib" "${HIP_PATH}/../lib64"
  NO_DEFAULT_PATH
)
if(NOT HIP_RUNTIME_LIBRARY)
  message(FATAL_ERROR "Could not find libamdhip64 under HIP_PATH=${HIP_PATH}")
endif()

add_library(ftccl_midware SHARED
  src/ftccl_midware.cpp
)
target_include_directories(ftccl_midware PRIVATE
  ${RCCL_ROOT}/include
  ${HIP_PATH}/include
)
target_link_directories(ftccl_midware PRIVATE
  ${RCCL_ROOT}
  ${RCCL_ROOT}/lib
  ${RCCL_ROOT}/lib64
  ${HIP_PATH}/lib
  ${HIP_PATH}/lib64
)
target_link_libraries(ftccl_midware PRIVATE ${RCCL_LIBRARY} ${HIP_RUNTIME_LIBRARY} dl pthread)
set_target_properties(ftccl_midware PROPERTIES
  OUTPUT_NAME "ftccl_midware"
  BUILD_RPATH "${RCCL_ROOT};${RCCL_ROOT}/lib;${RCCL_ROOT}/lib64;${HIP_PATH}/lib;${HIP_PATH}/lib64"
)

add_executable(ftccl_allreduce_smoke
  tests/ftccl_allreduce_smoke.cpp
)
target_include_directories(ftccl_allreduce_smoke PRIVATE
  ${RCCL_ROOT}/include
  ${HIP_PATH}/include
)
target_link_directories(ftccl_allreduce_smoke PRIVATE
  ${RCCL_ROOT}
  ${RCCL_ROOT}/lib
  ${RCCL_ROOT}/lib64
  ${HIP_PATH}/lib
  ${HIP_PATH}/lib64
)
target_link_libraries(ftccl_allreduce_smoke PRIVATE ${RCCL_LIBRARY} ${HIP_RUNTIME_LIBRARY})
set_target_properties(ftccl_allreduce_smoke PROPERTIES
  BUILD_RPATH "${RCCL_ROOT};${RCCL_ROOT}/lib;${RCCL_ROOT}/lib64;${HIP_PATH}/lib;${HIP_PATH}/lib64"
)
