/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ROCMWRAP_H_
#define NCCL_ROCMWRAP_H_

#include <hsa/hsa.h>
#include "hsa/hsa_ext_amd.h"
#include <mutex>
#include <dlfcn.h>

typedef hsa_status_t (*PFN_hsa_init)();
typedef hsa_status_t (*PFN_hsa_system_get_info)(hsa_system_info_t attribute, void* value);
typedef hsa_status_t (*PFN_hsa_status_string)(hsa_status_t status, const char ** status_string);
typedef hsa_status_t (*PFN_hsa_amd_portable_export_dmabuf)(const void* ptr, size_t size, int* dmabuf, uint64_t* offset);

typedef hsa_status_t (*PFN_hsa_agent_get_info)(hsa_agent_t agent,hsa_agent_info_t attribute,void* value);
typedef hsa_status_t (*PFN_hsa_iterate_agents)(hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data);
typedef hsa_status_t (*PFN_hsa_ext_get_xhcl_link_count)(hsa_agent_t src_agent, hsa_agent_t dst_agent, uint32_t *link_count);
#ifdef HCU_SDMA_FEATURE
typedef hsa_status_t (*PFN_hsa_ext_create_sdma_group_queue)(hsa_agent_t src_agent, hsa_agent_t dst_agent, uint32_t size, uint32_t flag, hsa_sdma_group_queue_t *group_queue);
typedef hsa_status_t (*PFN_hsa_ext_destroy_sdma_group_queue)(hsa_agent_t agent);
#endif
typedef hsa_status_t (*PFN_hsa_amd_ipc_memory_create)(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle);
typedef hsa_status_t (*PFN_hsa_amd_ipc_memory_attach)(const hsa_amd_ipc_memory_t* handle, size_t len,uint32_t num_agents,const hsa_agent_t* mapping_agents,void** mapped_ptr);
typedef hsa_status_t (*PFN_hsa_amd_ipc_memory_detach)(void* mapped_ptr);
typedef hsa_status_t (*PFN_hsa_amd_memory_pool_get_info)(hsa_amd_memory_pool_t memory_pool, hsa_amd_memory_pool_info_t attribute, void* value);
typedef hsa_status_t (*PFN_hsa_amd_agent_iterate_memory_pools)(hsa_agent_t agent,hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),void* data);
typedef hsa_status_t (*PFN_hsa_amd_memory_pool_allocate)(hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void** ptr);
typedef hsa_status_t (*PFN_hsa_amd_memory_pool_free)(void* ptr);

struct hsaAgentInfo {
  int rank;
  bool validHsaAgent;
  int64_t busId;
  hsa_agent_t hsaAgent;
};


#define CUPFN(symbol) pfn_##symbol

// Check CUDA PFN driver calls
#define CUCHECK(cmd) do {				      \
    hsa_status_t err = pfn_##cmd;				      \
    if( err != HSA_STATUS_SUCCESS ) {				      \
      const char *errStr;				      \
      pfn_hsa_status_string(err, &errStr);	      \
      WARN("ROCr failure '%s'", errStr);		      \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {		      \
    hsa_status_t err = pfn_##cmd;				      \
    if( err != HSA_STATUS_SUCCESS ) {				      \
      const char *errStr;				      \
      pfn_hsa_status_string(err, &errStr);	      \
      WARN("ROCr failure '%s'", errStr);		      \
      res = ncclUnhandledCudaError;			      \
      goto label;					      \
    }							      \
} while(false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) do {						\
    hsa_status_t err = pfn_##cmd;						\
    if( err != HSA_STATUS_SUCCESS ) {						\
      const char *errStr;						\
      pfn_hsa_status_string(err, &errStr);			\
      INFO(NCCL_ALL,"%s:%d ROCr failure '%s'", __FILE__, __LINE__, errStr);	\
    }									\
} while(false)

#define CUCHECKTHREAD(cmd, args) do {					\
    hsa_status_t err = pfn_##cmd;						\
    if (err != HSA_STATUS_SUCCESS) {						\
      INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
      args->ret = ncclUnhandledCudaError;				\
      return args;							\
    }									\
} while(0)

static void *hsaLib = nullptr;
static std::mutex hsaMtx;

static void* loadHsaLib() {
  char path[1024];
  char *ncclRocrPath = getenv("RCCL_ROCR_PATH");

  if (ncclRocrPath == NULL) {
    snprintf(path, sizeof(path), "libhsa-runtime64.so");
  } else {
    snprintf(path, sizeof(path), "%s/%s", ncclRocrPath, "libhsa-runtime64.so");
  }

  void* hsaLib = dlopen(path, RTLD_LAZY);
  if (hsaLib == nullptr) {
    const char *error = dlerror();  
    fprintf(stderr, "Failed to find ROCm runtime library in %s (RCCL_ROCR_PATH=%s): %s\n",
      path, ncclRocrPath, error ? error : "Unknown error");
    return nullptr;
  }

  return hsaLib;
}

static void* getHsaLib() {
  if (hsaLib == nullptr) {    
    std::lock_guard<std::mutex> lock(hsaMtx);
    if (hsaLib == nullptr) {
      hsaLib = loadHsaLib();
    }
  }

  return hsaLib;
}

#define DECLARE_ROCM_PFN_EXTERN(symbol) extern PFN_##symbol pfn_##symbol

DECLARE_ROCM_PFN_EXTERN(hsa_amd_portable_export_dmabuf); // DMA-BUF support

/* ROCr Driver functions loaded with dlsym() */
DECLARE_ROCM_PFN_EXTERN(hsa_init);
DECLARE_ROCM_PFN_EXTERN(hsa_system_get_info);
DECLARE_ROCM_PFN_EXTERN(hsa_status_string);

DECLARE_ROCM_PFN_EXTERN(hsa_agent_get_info);
DECLARE_ROCM_PFN_EXTERN(hsa_iterate_agents);
DECLARE_ROCM_PFN_EXTERN(hsa_ext_get_xhcl_link_count);
#ifdef HCU_SDMA_FEATURE
DECLARE_ROCM_PFN_EXTERN(hsa_ext_create_sdma_group_queue);
DECLARE_ROCM_PFN_EXTERN(hsa_ext_destroy_sdma_group_queue);
#endif
DECLARE_ROCM_PFN_EXTERN(hsa_amd_ipc_memory_create);
DECLARE_ROCM_PFN_EXTERN(hsa_amd_ipc_memory_attach);
DECLARE_ROCM_PFN_EXTERN(hsa_amd_ipc_memory_detach);
DECLARE_ROCM_PFN_EXTERN(hsa_amd_memory_pool_get_info);
DECLARE_ROCM_PFN_EXTERN(hsa_amd_agent_iterate_memory_pools);
DECLARE_ROCM_PFN_EXTERN(hsa_amd_memory_pool_allocate);
DECLARE_ROCM_PFN_EXTERN(hsa_amd_memory_pool_free);

ncclResult_t rocmLibraryInit(void);

extern bool ncclCudaLaunchBlocking; // initialized by ncclCudaLibraryInit()

ncclResult_t rocm_hsa_agent_get_info(hsa_agent_t agent,hsa_agent_info_t attribute,void* value);
ncclResult_t rocm_hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data);
ncclResult_t rocm_hsa_ext_get_xhcl_link_count(hsa_agent_t src_agent, hsa_agent_t dst_agent, uint32_t *link_count);
#ifdef HCU_SDMA_FEATURE
ncclResult_t rocm_hsa_ext_create_sdma_group_queue(hsa_agent_t src_agent, hsa_agent_t dst_agent, uint32_t size, uint32_t flag, hsa_sdma_group_queue_t *group_queue);
ncclResult_t rocm_hsa_ext_destroy_sdma_group_queue(hsa_agent_t agent);
#endif
ncclResult_t rocm_hsa_amd_ipc_memory_create(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle);
ncclResult_t rocm_hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* handle, size_t len,uint32_t num_agents,const hsa_agent_t* mapping_agents,void** mapped_ptr);
ncclResult_t rocm_hsa_amd_ipc_memory_detach(void* mapped_ptr);
ncclResult_t rocm_hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool, hsa_amd_memory_pool_info_t attribute, void* value);
ncclResult_t rocm_hsa_amd_agent_iterate_memory_pools(hsa_agent_t agent,hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),void* data);
ncclResult_t rocm_hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void** ptr);
ncclResult_t rocm_hsa_amd_memory_pool_free(void* ptr);

ncclResult_t rocmGetHsaAgentFromBdf(struct hsaAgentInfo *agentInfo);
ncclResult_t rocmAllocPcieLinkMem(hsa_agent_t agent_handle, size_t size, void** ptr);

#endif
