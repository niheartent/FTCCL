#include <hip/hip_runtime.h>
#include <nccl.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define HIPC_CHECK(cmd) do { \
  hipError_t e = (cmd); \
  if (e != hipSuccess) { \
    fprintf(stderr, "rank %d HIP error %s:%d %s\n", gRank, __FILE__, __LINE__, hipGetErrorString(e)); \
    return 1; \
  } \
} while (0)

#define NCCLCHECK(cmd) do { \
  ncclResult_t r = (cmd); \
  if (r != ncclSuccess) { \
    fprintf(stderr, "rank %d NCCL error %s:%d %s\n", gRank, __FILE__, __LINE__, ncclGetErrorString(r)); \
    return 1; \
  } \
} while (0)

static int gRank = -1;

static bool exists(const char* path) {
  struct stat st;
  return stat(path, &st) == 0;
}

static double nowSec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int waitFile(const char* path, int timeoutSec) {
  double deadline = nowSec() + timeoutSec;
  while (nowSec() < deadline) {
    if (exists(path)) return 0;
    usleep(10000);
  }
  fprintf(stderr, "rank %d timeout waiting for %s\n", gRank, path);
  return 1;
}

static void writeUniqueId(const char* path, const ncclUniqueId* id) {
  char tmp[4096];
  snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
  FILE* f = fopen(tmp, "wb");
  if (!f) {
    fprintf(stderr, "rank %d open %s failed: %s\n", gRank, tmp, strerror(errno));
    exit(1);
  }
  fwrite(id, sizeof(*id), 1, f);
  fclose(f);
  rename(tmp, path);
}

static void readUniqueId(const char* path, ncclUniqueId* id) {
  if (waitFile(path, 30) != 0) exit(1);
  FILE* f = fopen(path, "rb");
  if (!f || fread(id, sizeof(*id), 1, f) != 1) {
    fprintf(stderr, "rank %d read unique id failed\n", gRank);
    exit(1);
  }
  fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s rank nranks id_file count\n", argv[0]);
    return 1;
  }
  int rank = atoi(argv[1]);
  int nranks = atoi(argv[2]);
  const char* idFile = argv[3];
  int count = atoi(argv[4]);
  gRank = rank;

  int devCount = 0;
  HIPC_CHECK(hipGetDeviceCount(&devCount));
  HIPC_CHECK(hipSetDevice(rank % devCount));

  ncclUniqueId id;
  if (rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&id));
    writeUniqueId(idFile, &id);
  } else {
    readUniqueId(idFile, &id);
  }

  ncclComm_t comm = nullptr;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));

  hipStream_t stream;
  HIPC_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  float* dSend = nullptr;
  float* dRecv = nullptr;
  HIPC_CHECK(hipMalloc(&dSend, count * sizeof(float)));
  HIPC_CHECK(hipMalloc(&dRecv, count * sizeof(float)));

  std::string bypassDir = getenv("FTCCL_BYPASS_DIR") ? getenv("FTCCL_BYPASS_DIR") : "/tmp/ftccl_bypass";
  int deadRank = getenv("FTCCL_BYPASS_DEAD_RANK") ? atoi(getenv("FTCCL_BYPASS_DEAD_RANK")) : nranks - 1;
  int survivorCount = nranks - 1;
  bool survivor = rank != deadRank;

  for (int iter = 0; iter < 3; ++iter) {
    float input = rank + 1.0f + iter;
    HIPC_CHECK(hipMemcpy(dSend, &input, sizeof(float), hipMemcpyHostToDevice));
    HIPC_CHECK(hipMemset(dRecv, 0, count * sizeof(float)));
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllReduce(dSend, dRecv, count, ncclFloat32, ncclSum, comm, stream));
    NCCLCHECK(ncclGroupEnd());
    HIPC_CHECK(hipStreamSynchronize(stream));
    if (survivor) {
      float observed = 0.0f;
      HIPC_CHECK(hipMemcpy(&observed, dRecv, sizeof(float), hipMemcpyDeviceToHost));
      float expected = 0.0f;
      bool bypassActive = exists((bypassDir + "/activate." + std::to_string(rank)).c_str());
      if (bypassActive) {
        for (int r = 0; r < nranks; ++r) if (r != deadRank) expected += r + 1.0f + iter;
      } else {
        for (int r = 0; r < nranks; ++r) expected += r + 1.0f + iter;
      }
      if (observed != expected) {
        fprintf(stderr, "rank %d iter %d mismatch observed=%f expected=%f\n", rank, iter, observed, expected);
        return 2;
      }
      printf("SMOKE_PASS rank=%d iter=%d observed=%f survivor_count=%d\n",
             rank, iter, observed, bypassActive ? survivorCount : nranks);
      fflush(stdout);
    }
  }

  hipFree(dSend);
  hipFree(dRecv);
  ncclCommAbort(comm);
  hipStreamDestroy(stream);
  return 0;
}
