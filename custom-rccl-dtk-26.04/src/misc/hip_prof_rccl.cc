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

 #include "hipprof/hip_prof_rccl.h"  
 #include <dlfcn.h> 
 #include "debug.h"
 
 PFN_rccl_prof_api_enter pfn_rccl_prof_api_enter = nullptr;  
 PFN_rccl_prof_api_exit pfn_rccl_prof_api_exit = nullptr;  
 
 void init_rccl_prof_fns() {  
     static void *galaxyLib = nullptr;  
     if (galaxyLib == nullptr) {  
         char path[1024];  
         const char *ncclGalaxyPath = getenv("HIP_PATH");  
         if (ncclGalaxyPath == NULL) {  
             snprintf(path, sizeof(path), "libgalaxyhip.so");  
         } else {  
             snprintf(path, sizeof(path), "%s/lib/libgalaxyhip.so", ncclGalaxyPath);  
         }  
  
         galaxyLib = dlopen(path, RTLD_LAZY);  
         if (galaxyLib == NULL) {  
             WARN("Failed to find galaxy runtime library in %s", ncclGalaxyPath);  
             return;  
         }  
  
         pfn_rccl_prof_api_enter = (PFN_rccl_prof_api_enter) dlsym(galaxyLib, "rccl_prof_api_enter");  
         if (pfn_rccl_prof_api_enter == NULL) {  
             WARN("Failed to load Galaxy missing symbol rccl_prof_api_enter");  
         }  
  
         pfn_rccl_prof_api_exit = (PFN_rccl_prof_api_exit) dlsym(galaxyLib, "rccl_prof_api_exit");  
         if (pfn_rccl_prof_api_exit == NULL) {  
             WARN("Failed to load Galaxy missing symbol rccl_prof_api_exit");  
         }  
     }  
 }
 