/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "debug.h"
#include "rocmwrap.h"
#include "hsa/hsa.h"
#include "param.h"

#include <dlfcn.h>
#include <sys/utsname.h>
#include <fstream>

#define DECLARE_ROCM_PFN(symbol) PFN_##symbol pfn_##symbol = nullptr

DECLARE_ROCM_PFN(hsa_amd_portable_export_dmabuf); // DMA-BUF support
NCCL_PARAM(DmaBufEnable, "DMABUF_ENABLE", 0);
/* ROCr Driver functions loaded with dlsym() */
DECLARE_ROCM_PFN(hsa_init);
DECLARE_ROCM_PFN(hsa_system_get_info);
DECLARE_ROCM_PFN(hsa_status_string);

DECLARE_ROCM_PFN(hsa_agent_get_info);
DECLARE_ROCM_PFN(hsa_iterate_agents);
DECLARE_ROCM_PFN(hsa_ext_get_xhcl_link_count);
DECLARE_ROCM_PFN(hsa_amd_ipc_memory_create);
DECLARE_ROCM_PFN(hsa_amd_ipc_memory_attach);
DECLARE_ROCM_PFN(hsa_amd_ipc_memory_detach);
DECLARE_ROCM_PFN(hsa_amd_memory_pool_get_info);
DECLARE_ROCM_PFN(hsa_amd_agent_iterate_memory_pools);
DECLARE_ROCM_PFN(hsa_amd_memory_pool_allocate);
DECLARE_ROCM_PFN(hsa_amd_memory_pool_free);

static uint16_t version_major, version_minor;
bool ncclCudaLaunchBlocking = false;

static pthread_once_t initOnceControl = PTHREAD_ONCE_INIT;
static ncclResult_t initResult;

#define LOAD_HSA_SYM(handle, symbol, funcptr) do {         \
  void** cast = (void**)&funcptr;                          \
  void* tmp = dlsym(handle, symbol);                       \
  if (tmp == NULL) {                                       \
    WARN("dlvsym failed on %s - %s ", symbol, dlerror());  \
    goto error;                                            \
  }                                                        \
  *cast = tmp;                                             \
} while (0)

#define CALL_FUNC(func, ...)                               \
  do {                                                     \
    if (func == NULL) {                                    \
        WARN("Fail to call null func " #func);             \
        return ncclInternalError;                          \
    }                                                      \
    hsa_status_t status = func(__VA_ARGS__);               \
    if (status != HSA_STATUS_SUCCESS) {                    \
      WARN("Fail to call " #func " status %d", status);    \
      return ncclInternalError;                            \
    }                                                      \
    return ncclSuccess;                                    \
  } while (0)

bool isEnvEnabled(const char* envName) {
  char* val = getenv(envName);
  return val != nullptr && val[0] != '\0' && !(val[0] == '0' && val[1] == '\0');
}

static void initOnceFunc() {
  do {
    ncclCudaLaunchBlocking = (isEnvEnabled("GPU_FLUSH_ON_EXECUTION") | isEnvEnabled("CUDA_LAUNCH_BLOCKING") | isEnvEnabled("HIP_LAUNCH_BLOCKING"));
  } while (0);

  bool dmaBufSupport = false;
  hsa_status_t res;

  pfn_hsa_system_get_info = (PFN_hsa_system_get_info) dlsym(getHsaLib(), "hsa_system_get_info");
  if (pfn_hsa_system_get_info == NULL) {
    WARN("Failed to load ROCr missing symbol hsa_system_get_info");
    goto error;
  }

  pfn_hsa_status_string = (PFN_hsa_status_string) dlsym(getHsaLib(), "hsa_status_string");
  if (pfn_hsa_status_string == NULL) {
    WARN("Failed to load ROCr missing symbol hsa_status_string");
    goto error;
  }

  res = pfn_hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &version_major);
  if (res != 0) {
    WARN("pfn_hsa_system_get_info failed with %d", res);
    goto error;
  }
  res = pfn_hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &version_minor);
  if (res != 0) {
    WARN("pfn_hsa_system_get_info failed with %d", res);
    goto error;
  }
  
  LOAD_HSA_SYM(getHsaLib(), "hsa_agent_get_info", pfn_hsa_agent_get_info);
  LOAD_HSA_SYM(getHsaLib(), "hsa_iterate_agents", pfn_hsa_iterate_agents);
  LOAD_HSA_SYM(getHsaLib(), "hsa_ext_get_xhcl_link_count", pfn_hsa_ext_get_xhcl_link_count);
  LOAD_HSA_SYM(getHsaLib(), "hsa_amd_ipc_memory_create", pfn_hsa_amd_ipc_memory_create);
  LOAD_HSA_SYM(getHsaLib(), "hsa_amd_ipc_memory_attach", pfn_hsa_amd_ipc_memory_attach);
  LOAD_HSA_SYM(getHsaLib(), "hsa_amd_ipc_memory_detach", pfn_hsa_amd_ipc_memory_detach);
  LOAD_HSA_SYM(getHsaLib(), "hsa_amd_memory_pool_get_info", pfn_hsa_amd_memory_pool_get_info);
  LOAD_HSA_SYM(getHsaLib(), "hsa_amd_agent_iterate_memory_pools", pfn_hsa_amd_agent_iterate_memory_pools);
  LOAD_HSA_SYM(getHsaLib(), "hsa_amd_memory_pool_allocate", pfn_hsa_amd_memory_pool_allocate);
  LOAD_HSA_SYM(getHsaLib(), "hsa_amd_memory_pool_free", pfn_hsa_amd_memory_pool_free);


  INFO(NCCL_INIT, "ROCr version %d.%d", version_major, version_minor);

  //if (hsaDriverVersion < ROCR_DRIVER_MIN_VERSION) {
    // WARN("ROCr Driver version found is %d. Minimum requirement is %d", hsaDriverVersion, ROCR_DRIVER_MIN_VERSION);
    // Silently ignore version check mismatch for backwards compatibility
    //goto error;
  //}

  /* DMA-BUF support */
  //ROCm support
  if (ncclParamDmaBufEnable() == 0 ) {
    INFO(NCCL_INIT, "Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1");
    goto error;
  }
  res = pfn_hsa_system_get_info((hsa_system_info_t) 0x204, &dmaBufSupport);
  if (res != HSA_STATUS_SUCCESS || !dmaBufSupport) {
    INFO(NCCL_INIT, "Current version of ROCm does not support dmabuf feature.");
    goto error;
  }
  else {
    pfn_hsa_amd_portable_export_dmabuf = (PFN_hsa_amd_portable_export_dmabuf) dlsym(getHsaLib(), "hsa_amd_portable_export_dmabuf");
    if (pfn_hsa_amd_portable_export_dmabuf == NULL) {
      WARN("Failed to load ROCr missing symbol hsa_amd_portable_export_dmabuf");
      goto error;
    }
    else {
      //check OS kernel support
      struct utsname utsname;
      FILE *fp = NULL;
      char kernel_opt1[28] = "CONFIG_DMABUF_MOVE_NOTIFY=y";
      char kernel_opt2[20] = "CONFIG_PCI_P2PDMA=y";
      char kernel_conf_file[128];
      char buf[256];
      int found_opt1 = 0;
      int found_opt2 = 0;

      //check for kernel name exists
      if (uname(&utsname) == -1) INFO(NCCL_INIT,"Could not get kernel name");
      //format and store the kernel conf file location
      snprintf(kernel_conf_file, sizeof(kernel_conf_file), "/boot/config-%s", utsname.release);
      fp = fopen(kernel_conf_file, "r");
      if (fp == NULL) INFO(NCCL_INIT,"Could not open kernel conf file");
      //look for kernel_opt1 and kernel_opt2 in the conf file and check
      while (fgets(buf, sizeof(buf), fp) != NULL) {
        if (strstr(buf, kernel_opt1) != NULL) {
          found_opt1 = 1;
          INFO(NCCL_INIT,"CONFIG_DMABUF_MOVE_NOTIFY=y in /boot/config-%s", utsname.release);
        }
        if (strstr(buf, kernel_opt2) != NULL) {
          found_opt2 = 1;
          INFO(NCCL_INIT,"CONFIG_PCI_P2PDMA=y in /boot/config-%s", utsname.release);
        }
      }
      if (!found_opt1 || !found_opt2) {
        dmaBufSupport = 0;
        INFO(NCCL_INIT, "CONFIG_DMABUF_MOVE_NOTIFY and CONFIG_PCI_P2PDMA should be set for DMA_BUF in /boot/config-%s", utsname.release);
        INFO(NCCL_INIT, "DMA_BUF_SUPPORT Failed due to OS kernel support");
      }

      if(dmaBufSupport) INFO(NCCL_INIT, "DMA_BUF Support Enabled");
      else goto error;
    }
  }

  /*
   * Required to initialize the ROCr Driver.
   * Multiple calls of hsa_init() will return immediately
   * without making any relevant change
   */
  pfn_hsa_init();

  initResult = ncclSuccess;
  return;

error:
  initResult = ncclSystemError;
}

int ncclCuMemEnable() {
  return 0;
}

ncclResult_t rocmLibraryInit() {
  pthread_once(&initOnceControl, initOnceFunc);
  return initResult;
}

ncclResult_t rocm_hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void* value)
{
  CALL_FUNC(pfn_hsa_agent_get_info, agent, attribute, value);
}

ncclResult_t rocm_hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data)
{
  CALL_FUNC(pfn_hsa_iterate_agents, callback, data);
}

ncclResult_t rocm_hsa_ext_get_xhcl_link_count(hsa_agent_t src_agent, hsa_agent_t dst_agent, uint32_t *link_count)
{
  CALL_FUNC(pfn_hsa_ext_get_xhcl_link_count, src_agent, dst_agent, link_count);
}

#ifdef HCU_SDMA_FEATURE
ncclResult_t rocm_hsa_ext_create_sdma_group_queue(hsa_agent_t src_agent, hsa_agent_t dst_agent, uint32_t size, uint32_t flag, hsa_sdma_group_queue_t *group_queue)
{
  CALL_FUNC(pfn_hsa_ext_create_sdma_group_queue, src_agent, dst_agent, size, flag, group_queue);
}

ncclResult_t rocm_hsa_ext_destroy_sdma_group_queue(hsa_agent_t agent)
{
  CALL_FUNC(pfn_hsa_ext_destroy_sdma_group_queue, agent);
}
#endif

ncclResult_t rocm_hsa_amd_ipc_memory_create(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle)
{
  CALL_FUNC(pfn_hsa_amd_ipc_memory_create, ptr, len, handle);
}

ncclResult_t rocm_hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* handle, size_t len, uint32_t num_agents, const hsa_agent_t* mapping_agents, void** mapped_ptr)
{
  CALL_FUNC(pfn_hsa_amd_ipc_memory_attach, handle, len, num_agents, mapping_agents, mapped_ptr);
}

ncclResult_t rocm_hsa_amd_ipc_memory_detach(void* mapped_ptr)
{
  CALL_FUNC(pfn_hsa_amd_ipc_memory_detach, mapped_ptr);
}

ncclResult_t rocm_hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool, hsa_amd_memory_pool_info_t attribute, void* value)
{
  CALL_FUNC(pfn_hsa_amd_memory_pool_get_info, memory_pool, attribute, value);
}

ncclResult_t rocm_hsa_amd_agent_iterate_memory_pools(hsa_agent_t agent,hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data), void* data)
{
  CALL_FUNC(pfn_hsa_amd_agent_iterate_memory_pools, agent, callback, data);
}

ncclResult_t rocm_hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void** ptr)
{
  CALL_FUNC(pfn_hsa_amd_memory_pool_allocate, memory_pool, size, flags, ptr);
}

ncclResult_t rocm_hsa_amd_memory_pool_free(void* ptr)
{
  CALL_FUNC(pfn_hsa_amd_memory_pool_free, ptr);
}

static hsa_status_t iterateGpuMemoryPoolCallback(hsa_amd_memory_pool_t pool, void* data)
{
#ifdef SUPPORT_PCIE_CHANNEL
  hsa_status_t status;

  if (data == nullptr) {
    WARN("Error, invalid data ptr");
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (pfn_hsa_amd_memory_pool_get_info == nullptr) {
    WARN("Error, invalid hsa pool get info ptr");
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_segment_t segment_type = (hsa_region_segment_t)0;
  status = pfn_hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (status != HSA_STATUS_SUCCESS) {
    WARN("Fail to get info POOL_INFO_SEGMENT");
    return status;
  }

  hsa_amd_memory_pool_t* gpu_ext_fine_grained_segment = reinterpret_cast<hsa_amd_memory_pool_t*>(data);
  switch (segment_type) {
    case HSA_REGION_SEGMENT_GLOBAL: {
      uint32_t global_flag = 0;
      status = pfn_hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag);
      if (status != HSA_STATUS_SUCCESS) {
        WARN("Fail to get info POOL_INFO_GLOBAL_FLAGS");
        return status;
      }

      // If the flag set is ext scoped fine grain, break the loop
      if ((global_flag & HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED) != 0) {
        *gpu_ext_fine_grained_segment = pool;
        break;
      }
    }
    default:
      break;
  }
  return HSA_STATUS_SUCCESS;
#else
  return HSA_STATUS_ERROR;
#endif
}

ncclResult_t rocmAllocPcieLinkMem(hsa_agent_t agent_handle, size_t size, void** ptr)
{
#ifdef SUPPORT_PCIE_CHANNEL
  hsa_status_t status;
  hsa_amd_memory_pool_t pool;
  pool.handle = 0;

  if (agent_handle.handle == 0) {
    WARN("Error, invalid agent handle");
    return ncclInternalError;
  }

  if (pfn_hsa_amd_agent_iterate_memory_pools == NULL) {
    WARN("Error, invalid hsa iterate memory pools funcptr");
    return ncclInternalError;
  }

  status = pfn_hsa_amd_agent_iterate_memory_pools(agent_handle, iterateGpuMemoryPoolCallback, &pool);
  if (status != HSA_STATUS_SUCCESS) {
    WARN("Error, fail to get memory pool status %d", status);
    return ncclInternalError;
  }

  if (pool.handle == 0) {
    WARN("Error, invalid pool handle");
    return ncclInternalError;
  }

  CALL_FUNC(pfn_hsa_amd_memory_pool_allocate, pool, size, HSA_AMD_MEMORY_POOL_PCIE_P2P_FLAG, ptr);

  INFO(NCCL_INIT, "rocm alloc pcie mem agent 0x%lx pool mem ptr %p size %ld", agent_handle.handle, *ptr, size);
  return ncclSuccess;
#else
  return ncclInternalError;
#endif  
}

static hsa_status_t iterateAgentCallback(hsa_agent_t agent, void* data)
{
  struct hsaAgentInfo *agentInfo = (struct hsaAgentInfo *)data;
  hsa_status_t status;
  uint32_t hsaPciBdfId = 0;
  uint32_t hsaPciDomainId = 0;

  if (data == NULL) {
    WARN("Invalid input data");
    return HSA_STATUS_ERROR;
  }

  if (pfn_hsa_agent_get_info == NULL) {
    WARN("Failed to load ROCr missing symbol hsa_agent_get_info");
    return HSA_STATUS_ERROR;
  }
  status = pfn_hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &hsaPciBdfId);
  if (status != HSA_STATUS_SUCCESS) {
    WARN("Fail to get info BDFID status %d", status);
    return status;
  }
  status = pfn_hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN), &hsaPciDomainId);
  if (status != HSA_STATUS_SUCCESS) {
    WARN("Fail to get info DOMAIN status %d", status);
    return status;
  }

  uint32_t hsaBus = (hsaPciBdfId >> 8) & 0xFF;
  uint32_t hsaDevice = (hsaPciBdfId >> 3) & 0x1F;
  uint32_t hsaFunction = hsaPciBdfId & 0x07;

  uint32_t rankDomainId = agentInfo->busId >> 20;
  uint32_t rankBus = (agentInfo->busId >> 12) & 0xFF;
  uint32_t rankDevice = (agentInfo->busId >> 4) & 0x1F;
  uint32_t rankFunction = agentInfo->busId & 0x07;

  if (hsaPciDomainId == rankDomainId && hsaBus == rankBus && hsaDevice == rankDevice && hsaFunction == rankFunction) {
    agentInfo->hsaAgent = agent;
    agentInfo->validHsaAgent = true;
    INFO(NCCL_INIT, "rank:%d get match hsa dev domain:0x%x bdf:0x%x rank busId:0x%ld",
      agentInfo->rank, hsaPciDomainId, hsaPciBdfId, agentInfo->busId);
  }

  return HSA_STATUS_SUCCESS;
}

ncclResult_t rocmGetHsaAgentFromBdf(struct hsaAgentInfo *agentInfo)
{
  hsa_status_t status;

  if (agentInfo == NULL) {
    WARN("Error, invalid agentInfo ptr");
    return ncclInternalError;
  }

  if (pfn_hsa_iterate_agents == NULL) {
    WARN("Error, invalid hsa iterate agents funcptr");
    return ncclInternalError;
  }

  agentInfo->validHsaAgent = false;
  status = pfn_hsa_iterate_agents(iterateAgentCallback, agentInfo);
  if (status != HSA_STATUS_SUCCESS) {
    WARN("Error, tail to get hsa agent");
    return ncclInternalError;
  }

  if (!agentInfo->validHsaAgent) {
    WARN("rank:%d bdf:0x%lx fail to get valid hsa agent", agentInfo->rank, agentInfo->busId);
    return ncclInternalError;
  }

  return ncclSuccess;
}