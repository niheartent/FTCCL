#ifndef HIP_SRC_HIP_PROF_RCCL_PARAM_H
#define HIP_SRC_HIP_PROF_RCCL_PARAM_H

#include <stdint.h>
void ncclLoadHipProfParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache);
int64_t ncclParamHipProf();

#endif // HIP_SRC_HIP_PROF_RCCL_PARAM_H
