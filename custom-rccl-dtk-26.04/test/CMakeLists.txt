# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.

cmake_minimum_required(VERSION 2.8.12)

if(BUILD_TESTS)

  option(OPENMP_TESTS_ENABLED "Enable OpenMP for unit tests" OFF)

  message("Building rccl unit tests (Installed in /test/rccl-UnitTests)")
  if (GTEST_PATH)
    set(GTEST_ROOT ${GTEST_PATH})
    find_package(GTest 1.11)
  else()
    # 单独给 googletest 一个构建目录，并传递编译器
    set(GTEST_BINARY_DIR ${CMAKE_BINARY_DIR}/googletest-build)
    set(GTEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/googletest)

    add_subdirectory(${GTEST_SOURCE_DIR} ${GTEST_BINARY_DIR} EXCLUDE_FROM_ALL)

    enable_testing()
    include(GoogleTest)
  endif()
  if (BUILD_TESTS_ONLY)
    add_compile_options(-std=c++14)
    find_package(hip REQUIRED)
    find_package(ROCM REQUIRED)
    include(ROCMInstallTargets)
  endif()

  find_package(hsa-runtime64 PATHS /opt/rocm )
  if(${hsa-runtime64_FOUND})
    message("hsa-runtime64 found @  ${hsa-runtime64_DIR} ")
  else()
    message("find_package did NOT find hsa-runtime64, finding it the OLD Way")
    message("Looking for header files in ${ROCR_INC_DIR}")
    message("Looking for library files in ${ROCR_LIB_DIR}")

    # Search for ROCr header file in user defined locations
    find_path(ROCR_HDR hsa/hsa.h PATHS ${ROCR_INC_DIR} "/opt/rocm" PATH_SUFFIXES include REQUIRED)
    INCLUDE_DIRECTORIES(${ROCR_HDR})

    # Search for ROCr library file in user defined locations
    find_library(ROCR_LIB ${CORE_RUNTIME_TARGET} PATHS ${ROCR_LIB_DIR} "/opt/rocm" PATH_SUFFIXES lib lib64 REQUIRED)
  endif()

  if(OPENMP_TESTS_ENABLED)
    find_package(OpenMP REQUIRED)
  endif()

  include_directories(${GTEST_INCLUDE_DIRS} ./common)

  option(FULL_STRESS_TEST "Enable full stress test" OFF)

  if(FULL_STRESS_TEST)
    add_compile_definitions(FULL_STRESS_TEST)
  endif()

  # Collect testing framework source files
  set(TEST_SOURCE_FILES
    AllGatherTests.cpp
    AllReduceTests.cpp
    AllToAllTests.cpp
    AllToAllVTests.cpp
    BroadcastTests.cpp
    GatherTests.cpp
    GroupCallTests.cpp
    NonBlockingTests.cpp
    ReduceScatterTests.cpp
    ReduceTests.cpp
    ScatterTests.cpp
    SendRecvTests.cpp
    StandaloneTests.cpp
    PerformanceTests.cpp
    common/main.cpp
    common/CallCollectiveForked.cpp
    common/CollectiveArgs.cpp
    common/EnvVars.cpp
    common/PrepDataFuncs.cpp
    common/PtrUnion.cpp
    common/TestBed.cpp
    common/TestBedChild.cpp
    common/StandaloneUtils.cpp
    )

  add_executable(rccl-UnitTests ${TEST_SOURCE_FILES})

  ## Set rccl-UnitTests include directories
  target_include_directories(rccl-UnitTests PRIVATE ${ROCM_PATH} ${GTEST_INCLUDE_DIRS})
  target_compile_options(rccl-UnitTests PRIVATE -Wno-return-type)
  if (BUILD_TESTS_ONLY)
    target_include_directories(rccl-UnitTests PRIVATE ${PROJECT_BINARY_DIR}/../../build/include)        # for generated rccl.h header
    target_include_directories(rccl-UnitTests PRIVATE ${PROJECT_BINARY_DIR}/../../build/hipify/src/include) # for rccl_bfloat16.h
  else()
    target_include_directories(rccl-UnitTests PRIVATE ${PROJECT_BINARY_DIR}/include)        # for generated rccl.h header
    target_include_directories(rccl-UnitTests PRIVATE ${PROJECT_BINARY_DIR}/hipify/src/include) # for rccl_bfloat16.h
  endif()

  ## Set rccl-UnitTests compile definitions
  if(LL128_ENABLED)
    target_compile_definitions(rccl-UnitTests PRIVATE ENABLE_LL128)
  endif()
  if(OPENMP_TESTS_ENABLED)
    target_compile_definitions(rccl-UnitTests PRIVATE ENABLE_OPENMP)
  endif()
  target_compile_definitions(rccl-UnitTests PRIVATE ROCM_PATH="${ROCM_PATH}")

  ## Set rccl-UnitTests compile definitions
  if(OPENMP_TESTS_ENABLED)
    target_compile_options(rccl-UnitTests PRIVATE "${OpenMP_CXX_FLAGS}")
  endif()

  ## Set rccl-UnitTests linked libraries
  target_link_libraries(rccl-UnitTests PRIVATE GTest::gtest GTest::gtest_main)
  target_link_libraries(rccl-UnitTests PRIVATE hip::host hip::device hsa-runtime64::hsa-runtime64)
  target_link_libraries(rccl-UnitTests PRIVATE Threads::Threads)
  target_link_libraries(rccl-UnitTests PRIVATE dl) 
  if(OPENMP_TESTS_ENABLED)
    target_link_libraries(rccl-UnitTests PRIVATE "${OpenMP_CXX_FLAGS}")
  endif()

  # rccl-UnitTests using static library of rccl requires passing rccl
  # through -l and -L instead of command line input.
  if(BUILD_SHARED_LIBS)
    if (BUILD_TESTS_ONLY)
      target_link_libraries(rccl-UnitTests PRIVATE ${PROJECT_BINARY_DIR}/../../build/librccl.so)
    else()
      target_link_libraries(rccl-UnitTests PRIVATE rccl)
    endif()
    set_property(TARGET rccl-UnitTests PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${ROCM_PATH}/lib;${CMAKE_BINARY_DIR}")
  else()
    add_dependencies(rccl-UnitTests rccl)
    target_link_libraries(rccl-UnitTests PRIVATE dl rt numa -lrccl -L${CMAKE_BINARY_DIR} -lrocm_smi64 -L${ROCM_PATH}/lib -L${ROCM_PATH}/rocm_smi/lib)
  endif()
  set_property(TARGET rccl-UnitTests PROPERTY BUILD_RPATH "${CMAKE_BINARY_DIR};${ROCM_PATH}/lib")
  rocm_install(TARGETS rccl-UnitTests COMPONENT tests)
else()
  message("Not building rccl unit tests")
endif()
