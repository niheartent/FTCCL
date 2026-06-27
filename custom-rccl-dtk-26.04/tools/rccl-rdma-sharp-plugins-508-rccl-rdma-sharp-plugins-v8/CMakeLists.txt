cmake_minimum_required(VERSION 3.5)

project(rccl_plugins CXX)

set(SHARP_INSTALL_DIR "${CMAKE_SOURCE_DIR}/install" CACHE PATH "SHARP plugins install directory")

include(ExternalProject)

  ExternalProject_Add(
    rccl_rdma_sharp_plugins_508                  
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}  
    CONFIGURE_COMMAND             
        chmod 777 ${CMAKE_CURRENT_LIST_DIR}/autogen.sh &&
        ${CMAKE_CURRENT_LIST_DIR}/autogen.sh || true && 
        ${CMAKE_CURRENT_LIST_DIR}/autogen.sh &&
        ${CMAKE_CURRENT_LIST_DIR}/configure
        --prefix=${CMAKE_SOURCE_DIR}/install/lib-v8
        --with-hip=${ROCM_PATH}/hip
        --with-ucx=/usr 
        --without-sharp
    BUILD_COMMAND make -j 32      
    INSTALL_COMMAND make install  
    BUILD_IN_SOURCE TRUE          
  )

  #安装预设rccl.conf到用户主目录下
  set(RCCL_NET_PLUGIN_VALUE "shca" CACHE STRING "Default NCCL network plugin")

  configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/rccl.conf.in
    ${CMAKE_SOURCE_DIR}/rccl.conf
    @ONLY
  )

  rocm_install(
    FILES ${CMAKE_SOURCE_DIR}/rccl.conf
    DESTINATION ${ROCM_PATH}/rccl/lib
  )

  rocm_install(DIRECTORY ${CMAKE_SOURCE_DIR}/install/lib-v8/lib/ DESTINATION ${ROCM_PATH}/lib)












