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

 #ifndef HIP_SRC_HIP_PROF_RCCL_H
 #define HIP_SRC_HIP_PROF_RCCL_H
 
 #include "hipprof/hip_prof_rccl_str.h"
 #include "hipprof/hip_profile_common.h"
 
 
 
 typedef prof_error_t (*PFN_rccl_prof_api_enter)(uint32_t cid, void *api_entry);
 typedef prof_error_t (*PFN_rccl_prof_api_exit)(uint32_t cid, void *api_entry);
 
 extern PFN_rccl_prof_api_enter pfn_rccl_prof_api_enter;  
 extern PFN_rccl_prof_api_exit pfn_rccl_prof_api_exit;  
 
 void init_rccl_prof_fns(); 
 
 #endif  // HIP_SRC_HIP_PROF_RCCL_H
 