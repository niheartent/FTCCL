/* Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */


 #ifndef HIP_SRC_HIP_PROF_RCCL_API_H
 #define HIP_SRC_HIP_PROF_RCCL_API_H
 
 #include <atomic>
 #include <cassert>
 #include <iostream>
 #include <shared_mutex>
 #include <utility>
 #include "info.h"
 #include "debug.h"
 #include <dlfcn.h>
 #include <sys/utsname.h>
 #include <fstream>
 #include <sys/shm.h>
 #include "hipprof/hip_prof_rccl.h"
 #include "hipprof/hip_prof_rccl_param.h"

 const int type2size[ncclNumTypes] = {1,1,4,4,8,8,2,4,8,2,1,1};
 
 #define RCCL_CB_SPAWNER_OBJECT(operation_id, info_ptr) rccl_cb_spawner_object<RCCL_API_ID_##operation_id> __api_tracer(info_ptr); 
 template <rccl_api_id_t operation_id> class rccl_cb_spawner_object {  
 public:  
    rccl_cb_spawner_object(ncclInfo* info_ptr): info(info_ptr), entry(nullptr), stat(STATUS_ERROR), correlation_id(0) {
         if (ncclParamHipProf() == 1) {
             static_assert(operation_id >= RCCL_API_ID_FIRST && operation_id <= RCCL_API_ID_LAST, "invalid RCCL_API operation id"); 
             init_rccl_prof_fns();   
             entry = std::make_unique<hip_prof_rccl_entry>();  
             entry->kind = RCCL_KIND_ID_API;
             entry->ret_stat = 0;
             entry->cid = operation_id;
             entry->sendbuff = info->sendbuff;
             entry->recvbuff = info->recvbuff;
             entry->count = info->count;
             entry->datatype = info->datatype;
             entry->op = info->op;
             entry->rid = info->root; 
             if (pfn_rccl_prof_api_enter != nullptr) {
                 stat = pfn_rccl_prof_api_enter(operation_id, entry.get());  
                 if (stat != STATUS_SUCCESS) {
                     INFO(NCCL_INIT, "stat: %d, Failed to add rccl_prof_api_enter.", stat);
                     entry.reset();
                 } else {
                     correlation_id = entry->correlation_id;
                 }      
             }        
         } 
     }
     
     activity_correlation_id_t getCorrelationId() const {  
         return correlation_id;  
     }
 
     ~rccl_cb_spawner_object() { 
         if (stat == STATUS_SUCCESS && entry) {        
             entry->nBytes = info->count * type2size[info->datatype];
             if (pfn_rccl_prof_api_exit != nullptr) {
                 stat = pfn_rccl_prof_api_exit(operation_id, entry.get());
                 if (stat != STATUS_SUCCESS) {
                     INFO(NCCL_INIT, "Failed to add rccl_prof_api_exit.");
                 }
             }
             correlation_id = 0;
         }
     }  
   
 private:
     prof_error_t stat;
     std::unique_ptr<hip_prof_rccl_entry> entry;  
     activity_correlation_id_t correlation_id;
     ncclInfo* info;
 };
 
 
 #endif  // HIP_SRC_HIP_PROF_RCCL_API_H
 