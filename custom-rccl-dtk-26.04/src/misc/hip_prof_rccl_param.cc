#include "debug.h"
#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <pwd.h>
#include "hipprof/hip_prof_rccl_param.h"

const int deftVal = 0;

void ncclLoadHipProfParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&mutex);
  if (__atomic_load_n(cache, __ATOMIC_RELAXED) == uninitialized) {
    char* str = getenv(env);
    int64_t value = deftVal;
    if (str && strlen(str) > 0) {
      errno = 0;
      value = strtoll(str, nullptr, 0);
      if (errno) {
        value = deftVal;
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
      } else {
        INFO(NCCL_ENV,"%s set by environment to %lld.", env, (long long)value);
      }
    }
    __atomic_store_n(cache, value, __ATOMIC_RELAXED);
  }
  pthread_mutex_unlock(&mutex);
}

int64_t ncclParamHipProf() {  
    constexpr int64_t uninitialized = INT64_MIN;  
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value.");  
    static int64_t cache = uninitialized;
  
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false)) {  
        ncclLoadHipProfParam("NCCL_HIP_PROFILE", deftVal, uninitialized, &cache);  
    }  
    return cache;
}  
