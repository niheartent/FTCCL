/*************************************************************************
 * Native RCCL survivor collective experiments on an old parent communicator.
 *
 * All original ranks create parent_comm. The selected failed rank then
 * SIGKILLs itself and never enters NCCL again. Survivor ranks call
 * ncclCommPrepareSurvivorTopo() and ncclCommActivateSurvivorTopo(), then run
 * the requested collective on the old parent_comm. RCCL keeps NCCL-compatible
 * public API names, so nccl* symbols are expected here.
 ************************************************************************/

#include <hip/hip_runtime.h>
#include <nccl.h>

#include <dlfcn.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int g_rank = -1;

#define HIPCHECK(cmd)                                                          \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "rank %d HIP error %s:%d '%s' op=%s\n", g_rank,          \
              __FILE__, __LINE__, hipGetErrorString(e), #cmd);                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = (cmd);                                                    \
    if (r != ncclSuccess) {                                                    \
      fprintf(stderr, "rank %d NCCL error %s:%d '%s' op=%s\n", g_rank,         \
              __FILE__, __LINE__, ncclGetErrorString(r), #cmd);                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

enum Collective {
  COLLECTIVE_ALLREDUCE,
  COLLECTIVE_REDUCESCATTER,
  COLLECTIVE_ALLGATHER,
  COLLECTIVE_BROADCAST,
  COLLECTIVE_REDUCE,
  COLLECTIVE_INVALID
};

enum RunMode {
  MODE_BYPASS,
  MODE_NORMAL,
  MODE_INVALID
};

struct Options {
  Collective collective = COLLECTIVE_INVALID;
  RunMode mode = MODE_BYPASS;
  int nranks = 8;
  int rank = -1;
  int kill_rank = 2;
  int root_rank = -1;
  int count = 1024;
  int iters = 10;
  int timeout_sec = 30;
  const char* id_file = NULL;
  const char* sync_dir = NULL;
  const char* ccld_dump_dir = NULL;
};

struct LocalCcldChannelMetrics {
  uint64_t sendCount;
  uint64_t recvCount;
  uint64_t sendBytes;
  uint64_t recvBytes;
  uint64_t sendPostCount;
  uint64_t recvPostCount;
  uint64_t sendWaitSpins;
  uint64_t recvWaitSpins;
  uint64_t sendMaxWaitSpins;
  uint64_t recvMaxWaitSpins;
  uint64_t lastSendTimestamp;
  uint64_t lastRecvTimestamp;
  uint64_t lastAnyTimestamp;
};

typedef ncclResult_t (*SurvivorTopoFn)(ncclComm_t, int, const int*);
typedef ncclResult_t (*CcldGetMetricsFn)(const ncclComm_t, int, void*, size_t);

static void* resolve_rccl_symbol(const char* name) {
  dlerror();
  void* sym = dlsym(RTLD_DEFAULT, name);
  if (sym != NULL)
    return sym;

  const char* lib_dir = getenv("FTCCl_RCCL_LIB");
  if (lib_dir != NULL && lib_dir[0] != '\0') {
    char path[4096];
    snprintf(path, sizeof(path), "%s/librccl.so", lib_dir);
    void* handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (handle != NULL) {
      dlerror();
      sym = dlsym(handle, name);
      if (sym != NULL)
        return sym;
    }
  }

  const char* err = dlerror();
  fprintf(stderr, "rank %d failed to resolve RCCL symbol %s: %s\n", g_rank,
          name, err != NULL ? err : "symbol not found");
  return NULL;
}

static ncclResult_t call_prepare_survivor_topo(ncclComm_t comm,
                                               int survivor_count,
                                               const int* survivors) {
  static SurvivorTopoFn fn = NULL;
  if (fn == NULL)
    fn = (SurvivorTopoFn)resolve_rccl_symbol("ncclCommPrepareSurvivorTopo");
  return fn != NULL ? fn(comm, survivor_count, survivors) : ncclInternalError;
}

static ncclResult_t call_activate_survivor_topo(ncclComm_t comm,
                                                int survivor_count,
                                                const int* survivors) {
  static SurvivorTopoFn fn = NULL;
  if (fn == NULL)
    fn = (SurvivorTopoFn)resolve_rccl_symbol("ncclCommActivateSurvivorTopo");
  return fn != NULL ? fn(comm, survivor_count, survivors) : ncclInternalError;
}

static ncclResult_t call_ccld_get_metrics(const ncclComm_t comm,
                                          int max_channels, void* out_metrics,
                                          size_t out_bytes) {
  static CcldGetMetricsFn fn = NULL;
  if (fn == NULL)
    fn = (CcldGetMetricsFn)resolve_rccl_symbol("ncclCommCcldGetMetrics");
  return fn != NULL ? fn(comm, max_channels, out_metrics, out_bytes)
                    : ncclInternalError;
}

static double now_sec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static int file_exists(const char* path) {
  struct stat st;
  return stat(path, &st) == 0;
}

static int survivor_index(int rank, int kill_rank) {
  return rank < kill_rank ? rank : rank - 1;
}

static int survivor_rank_at(int index, int kill_rank) {
  return index < kill_rank ? index : index + 1;
}

static int first_survivor_rank(int kill_rank) {
  return kill_rank == 0 ? 1 : 0;
}

static const char* collective_name(Collective collective) {
  switch (collective) {
  case COLLECTIVE_ALLREDUCE: return "allreduce";
  case COLLECTIVE_REDUCESCATTER: return "reducescatter";
  case COLLECTIVE_ALLGATHER: return "allgather";
  case COLLECTIVE_BROADCAST: return "broadcast";
  case COLLECTIVE_REDUCE: return "reduce";
  default: return "invalid";
  }
}

static const char* mode_name(RunMode mode) {
  switch (mode) {
  case MODE_BYPASS: return "bypass";
  case MODE_NORMAL: return "normal";
  default: return "invalid";
  }
}

static RunMode parse_mode(const char* name) {
  if (strcmp(name, "bypass") == 0) return MODE_BYPASS;
  if (strcmp(name, "normal") == 0) return MODE_NORMAL;
  return MODE_INVALID;
}

static Collective parse_collective(const char* name) {
  if (strcmp(name, "allreduce") == 0) return COLLECTIVE_ALLREDUCE;
  if (strcmp(name, "reducescatter") == 0) return COLLECTIVE_REDUCESCATTER;
  if (strcmp(name, "allgather") == 0) return COLLECTIVE_ALLGATHER;
  if (strcmp(name, "broadcast") == 0) return COLLECTIVE_BROADCAST;
  if (strcmp(name, "reduce") == 0) return COLLECTIVE_REDUCE;
  return COLLECTIVE_INVALID;
}

static int is_root_collective(Collective collective) {
  return collective == COLLECTIVE_BROADCAST || collective == COLLECTIVE_REDUCE;
}

static void usage(const char* prog) {
  fprintf(stderr,
          "Usage: %s --collective NAME --rank R --nranks N --id-file PATH "
          "--sync-dir DIR [OPTIONS]\n"
          "\n"
          "Options:\n"
          "  --collective NAME allreduce, reducescatter, allgather, "
          "broadcast, or reduce\n"
          "  --mode NAME       bypass or normal. Default: bypass\n"
          "  --kill-rank R     Rank to SIGKILL before prepare. Default: 2\n"
          "  --root-rank R     Broadcast/Reduce root as parent rank. "
          "Default: first survivor\n"
          "  --count N         Number of int elements per rank operation. "
          "Default: 1024\n"
          "  --iters N         Survivor collective iterations. Default: 10\n"
          "  --timeout-sec N   Setup/communication timeout. Default: 30\n"
          "  --ccld-dump-dir D Write CCL-D metrics for this rank into D.\n"
          "  -h, --help        Show this help.\n",
          prog);
  return;
}

static Options parse_options(int argc, char** argv) {
  Options opts;
  static struct option long_options[] = {
      {"collective", required_argument, 0, 1},
      {"nranks", required_argument, 0, 2},
      {"rank", required_argument, 0, 3},
      {"kill-rank", required_argument, 0, 4},
      {"root-rank", required_argument, 0, 5},
      {"count", required_argument, 0, 6},
      {"iters", required_argument, 0, 7},
      {"timeout-sec", required_argument, 0, 8},
      {"id-file", required_argument, 0, 9},
      {"sync-dir", required_argument, 0, 10},
      {"ccld-dump-dir", required_argument, 0, 11},
      {"mode", required_argument, 0, 12},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0},
  };

  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "h", long_options, &option_index);
    if (c == -1)
      break;
    switch (c) {
    case 1: opts.collective = parse_collective(optarg); break;
    case 2: opts.nranks = atoi(optarg); break;
    case 3: opts.rank = atoi(optarg); break;
    case 4: opts.kill_rank = atoi(optarg); break;
    case 5: opts.root_rank = atoi(optarg); break;
    case 6: opts.count = atoi(optarg); break;
    case 7: opts.iters = atoi(optarg); break;
    case 8: opts.timeout_sec = atoi(optarg); break;
    case 9: opts.id_file = optarg; break;
    case 10: opts.sync_dir = optarg; break;
    case 11: opts.ccld_dump_dir = optarg; break;
    case 12: opts.mode = parse_mode(optarg); break;
    case 'h':
      usage(argv[0]);
      exit(EXIT_SUCCESS);
    default:
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if (opts.root_rank < 0 && opts.kill_rank >= 0)
    opts.root_rank = first_survivor_rank(opts.kill_rank);

  if (opts.collective == COLLECTIVE_INVALID || opts.mode == MODE_INVALID ||
      opts.rank < 0 || opts.rank >= opts.nranks || opts.nranks < 2 ||
      opts.kill_rank < 0 || opts.kill_rank >= opts.nranks ||
      opts.root_rank < 0 || opts.root_rank >= opts.nranks ||
      (opts.mode == MODE_BYPASS && opts.root_rank == opts.kill_rank) ||
      opts.count <= 0 || opts.iters <= 0 || opts.id_file == NULL ||
      opts.sync_dir == NULL) {
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }
  return opts;
}

static void touch_rank_file(const char* sync_dir, const char* prefix, int rank) {
  char path[4096];
  char tmp[8192];
  snprintf(path, sizeof(path), "%s/%s.%d", sync_dir, prefix, rank);
  snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
  FILE* f = fopen(tmp, "wb");
  if (f == NULL) {
    fprintf(stderr, "rank %d failed to open %s: %s\n", g_rank, tmp,
            strerror(errno));
    exit(EXIT_FAILURE);
  }
  fprintf(f, "%d\n", rank);
  fflush(f);
  fsync(fileno(f));
  fclose(f);
  if (rename(tmp, path) != 0) {
    fprintf(stderr, "rank %d failed to publish %s: %s\n", g_rank, path,
            strerror(errno));
    exit(EXIT_FAILURE);
  }
  int dir_fd = open(sync_dir, O_RDONLY | O_DIRECTORY);
  if (dir_fd >= 0) {
    fsync(dir_fd);
    close(dir_fd);
  }
}

static int wait_prefix(const char* sync_dir, const char* prefix, int nranks,
                       int skip_rank, int timeout_sec) {
  double deadline = now_sec() + timeout_sec;
  int* seen_ranks = (int*)calloc((size_t)nranks, sizeof(int));
  if (seen_ranks == NULL) {
    fprintf(stderr, "rank %d failed to allocate wait bitmap\n", g_rank);
    return 1;
  }
  char expected_prefix[256];
  snprintf(expected_prefix, sizeof(expected_prefix), "%s.", prefix);
  size_t expected_prefix_len = strlen(expected_prefix);

  while (now_sec() < deadline) {
    int seen = 0;
    memset(seen_ranks, 0, (size_t)nranks * sizeof(int));

    DIR* dir = opendir(sync_dir);
    if (dir != NULL) {
      struct dirent* ent = NULL;
      while ((ent = readdir(dir)) != NULL) {
        if (strncmp(ent->d_name, expected_prefix, expected_prefix_len) != 0)
          continue;
        char* end = NULL;
        long rank = strtol(ent->d_name + expected_prefix_len, &end, 10);
        if (end == ent->d_name + expected_prefix_len || *end != '\0')
          continue;
        if (rank < 0 || rank >= nranks || rank == skip_rank)
          continue;
        seen_ranks[rank] = 1;
      }
      closedir(dir);
    }

    for (int r = 0; r < nranks; ++r) {
      if (r == skip_rank)
        continue;
      seen += seen_ranks[r];
    }
    if (seen == nranks - (skip_rank >= 0 ? 1 : 0)) {
      free(seen_ranks);
      return 0;
    }
    usleep(10000);
  }

  fprintf(stderr, "rank %d timed out waiting for %s files; missing ranks:",
          g_rank, prefix);
  for (int r = 0; r < nranks; ++r) {
    if (r == skip_rank)
      continue;
    if (!seen_ranks[r])
      fprintf(stderr, " %d", r);
  }
  fprintf(stderr, "\n");
  free(seen_ranks);
  return 1;
}

static int wait_file(const char* path, int timeout_sec) {
  double deadline = now_sec() + timeout_sec;
  while (now_sec() < deadline) {
    if (file_exists(path))
      return 0;
    usleep(10000);
  }
  fprintf(stderr, "rank %d timed out waiting for %s\n", g_rank, path);
  return 1;
}

static void write_unique_id(const char* path, const ncclUniqueId* id) {
  char tmp[8192];
  snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
  FILE* f = fopen(tmp, "wb");
  if (f == NULL) {
    fprintf(stderr, "rank %d failed to open %s: %s\n", g_rank, tmp,
            strerror(errno));
    exit(EXIT_FAILURE);
  }
  if (fwrite(id, sizeof(*id), 1, f) != 1) {
    fprintf(stderr, "rank %d failed to write %s\n", g_rank, tmp);
    exit(EXIT_FAILURE);
  }
  fclose(f);
  if (rename(tmp, path) != 0) {
    fprintf(stderr, "rank %d failed to publish %s: %s\n", g_rank, path,
            strerror(errno));
    exit(EXIT_FAILURE);
  }
}

static void read_unique_id(const char* path, ncclUniqueId* id, int timeout_sec) {
  if (wait_file(path, timeout_sec) != 0)
    exit(EXIT_FAILURE);
  FILE* f = fopen(path, "rb");
  if (f == NULL) {
    fprintf(stderr, "rank %d failed to open %s: %s\n", g_rank, path,
            strerror(errno));
    exit(EXIT_FAILURE);
  }
  if (fread(id, sizeof(*id), 1, f) != 1) {
    fprintf(stderr, "rank %d failed to read ncclUniqueId\n", g_rank);
    exit(EXIT_FAILURE);
  }
  fclose(f);
}

static int wait_stream_with_timeout(hipStream_t stream, ncclComm_t comm,
                                    const char* label, int timeout_sec) {
  hipEvent_t done;
  HIPCHECK(hipEventCreateWithFlags(&done, hipEventDisableTiming));
  HIPCHECK(hipEventRecord(done, stream));
  double deadline = now_sec() + timeout_sec;
  while (now_sec() < deadline) {
    hipError_t q = hipEventQuery(done);
    if (q == hipSuccess) {
      HIPCHECK(hipEventDestroy(done));
      return 0;
    }
    if (q != hipErrorNotReady) {
      fprintf(stderr, "rank %d hipEventQuery failed during %s: %s\n", g_rank,
              label, hipGetErrorString(q));
      HIPCHECK(hipEventDestroy(done));
      return 1;
    }
    ncclResult_t async = ncclSuccess;
    ncclResult_t nr = ncclCommGetAsyncError(comm, &async);
    if (nr != ncclSuccess || async != ncclSuccess) {
      fprintf(stderr, "rank %d NCCL async error during %s: call=%s async=%s\n",
              g_rank, label, ncclGetErrorString(nr), ncclGetErrorString(async));
      HIPCHECK(hipEventDestroy(done));
      return 1;
    }
    usleep(1000);
  }
  fprintf(stderr, "rank %d timed out during %s\n", g_rank, label);
  HIPCHECK(hipEventDestroy(done));
  return 1;
}

static int run_parent_barrier(ncclComm_t comm, hipStream_t stream,
                              int timeout_sec) {
  int h_send = 1;
  int h_recv = 0;
  int* d_send = NULL;
  int* d_recv = NULL;
  HIPCHECK(hipMalloc(&d_send, sizeof(int)));
  HIPCHECK(hipMalloc(&d_recv, sizeof(int)));
  HIPCHECK(hipMemcpy(d_send, &h_send, sizeof(int), hipMemcpyHostToDevice));
  HIPCHECK(hipMemset(d_recv, 0, sizeof(int)));
  NCCLCHECK(ncclAllReduce(d_send, d_recv, 1, ncclInt, ncclSum, comm, stream));
  int ret = wait_stream_with_timeout(stream, comm, "parent_barrier",
                                     timeout_sec);
  if (ret == 0)
    HIPCHECK(hipMemcpy(&h_recv, d_recv, sizeof(int), hipMemcpyDeviceToHost));
  HIPCHECK(hipFree(d_send));
  HIPCHECK(hipFree(d_recv));
  if (ret != 0)
    return ret;
  return h_recv > 0 ? 0 : 1;
}

static int ensure_dir(const char* path) {
  if (mkdir(path, 0775) == 0 || errno == EEXIST)
    return 0;
  fprintf(stderr, "rank %d failed to create %s: %s\n", g_rank, path,
          strerror(errno));
  return 1;
}

static int dump_ccld_metrics(const Options& opts, ncclComm_t comm) {
  if (opts.ccld_dump_dir == NULL)
    return 0;
  if (ensure_dir(opts.ccld_dump_dir) != 0)
    return 1;

  const int max_channels = 32;
  LocalCcldChannelMetrics metrics[max_channels];
  memset(metrics, 0, sizeof(metrics));
  NCCLCHECK(call_ccld_get_metrics(comm, max_channels, metrics,
                                  sizeof(metrics)));

  char host[256] = {0};
  if (gethostname(host, sizeof(host) - 1) != 0)
    snprintf(host, sizeof(host), "unknown");

  char path[4096];
  snprintf(path, sizeof(path), "%s/ccld.rank%d.%s.%d.tsv", opts.ccld_dump_dir,
           opts.rank, host, getpid());
  FILE* f = fopen(path, "wb");
  if (f == NULL) {
    fprintf(stderr, "rank %d failed to open CCL-D dump %s: %s\n", opts.rank,
            path, strerror(errno));
    return 1;
  }

  fprintf(f, "rank\tchannel\tsendCount\trecvCount\tsendBytes\trecvBytes\t"
             "sendPostCount\trecvPostCount\tsendWaitSpins\trecvWaitSpins\t"
             "sendMaxWaitSpins\trecvMaxWaitSpins\tlastSendTimestamp\t"
             "lastRecvTimestamp\tlastAnyTimestamp\n");
  uint64_t nonzero = 0;
  for (int c = 0; c < max_channels; ++c) {
    const LocalCcldChannelMetrics* m = metrics + c;
    nonzero += m->sendCount + m->recvCount + m->sendBytes + m->recvBytes +
               m->sendPostCount + m->recvPostCount + m->sendWaitSpins +
               m->recvWaitSpins + m->sendMaxWaitSpins + m->recvMaxWaitSpins +
               m->lastSendTimestamp + m->lastRecvTimestamp +
               m->lastAnyTimestamp;
    fprintf(f, "%d\t%d\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t"
               "%lu\t%lu\t%lu\n",
            opts.rank, c, (unsigned long)m->sendCount,
            (unsigned long)m->recvCount, (unsigned long)m->sendBytes,
            (unsigned long)m->recvBytes, (unsigned long)m->sendPostCount,
            (unsigned long)m->recvPostCount, (unsigned long)m->sendWaitSpins,
            (unsigned long)m->recvWaitSpins,
            (unsigned long)m->sendMaxWaitSpins,
            (unsigned long)m->recvMaxWaitSpins,
            (unsigned long)m->lastSendTimestamp,
            (unsigned long)m->lastRecvTimestamp,
            (unsigned long)m->lastAnyTimestamp);
  }
  fclose(f);
  printf("CCLD_DUMP rank=%d path=%s nonzero=%lu channels=%d\n", opts.rank,
         path, (unsigned long)nonzero, max_channels);
  fflush(stdout);
  return 0;
}

static void fill_send_host(int* data, int count, int rank) {
  for (int i = 0; i < count; ++i)
    data[i] = rank * 100000 + i;
  return;
}

static int expected_sum_value(int index, int nranks, int kill_rank) {
  int sum = 0;
  for (int r = 0; r < nranks; ++r) {
    if (r != kill_rank)
      sum += r * 100000 + index;
  }
  return sum;
}

static int verify_result(const Options& opts, int sidx, int survivor_count,
                         const int* observed) {
  if (opts.collective == COLLECTIVE_ALLGATHER) {
    for (int s = 0; s < survivor_count; ++s) {
      for (int i = 0; i < opts.count; ++i) {
        int expected_rank = opts.mode == MODE_BYPASS ? survivor_rank_at(s, opts.kill_rank) : s;
        int expected = expected_rank * 100000 + i;
        int actual = observed[s * opts.count + i];
        if (actual != expected) {
          fprintf(stderr,
                  "rank %d %s mismatch segment=%d index=%d expected=%d "
                  "observed=%d\n",
                  opts.rank, collective_name(opts.collective), s, i, expected,
                  actual);
          return 2;
        }
      }
    }
    return 0;
  }
  int expected_root_index = opts.mode == MODE_BYPASS
                                ? survivor_index(opts.root_rank, opts.kill_rank)
                                : opts.root_rank;
  if (opts.collective == COLLECTIVE_REDUCE && sidx != expected_root_index)
    return 0;

  for (int i = 0; i < opts.count; ++i) {
    int expected;
    if (opts.collective == COLLECTIVE_BROADCAST) {
      expected = opts.root_rank * 100000 + i;
    } else if (opts.collective == COLLECTIVE_REDUCESCATTER) {
      expected = expected_sum_value(sidx * opts.count + i, opts.nranks,
                                    opts.mode == MODE_BYPASS ? opts.kill_rank : -1);
    } else {
      expected = expected_sum_value(i, opts.nranks,
                                    opts.mode == MODE_BYPASS ? opts.kill_rank : -1);
    }
    if (observed[i] != expected) {
      fprintf(stderr,
              "rank %d %s mismatch index=%d expected=%d observed=%d "
              "survivor_index=%d\n",
              opts.rank, collective_name(opts.collective), i, expected,
              observed[i], sidx);
      return 2;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  Options opts = parse_options(argc, argv);
  g_rank = opts.rank;

  int device_count = 0;
  HIPCHECK(hipGetDeviceCount(&device_count));
  HIPCHECK(hipSetDevice(opts.rank % device_count));

  hipStream_t stream;
  HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  ncclUniqueId id;
  if (opts.rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&id));
    write_unique_id(opts.id_file, &id);
  } else {
    read_unique_id(opts.id_file, &id, opts.timeout_sec);
  }

  ncclComm_t parent = NULL;
  NCCLCHECK(ncclCommInitRank(&parent, opts.nranks, id, opts.rank));
  printf("PARENT_SETUP rank=%d nranks=%d collective=%s mode=%s kill_rank=%d root_rank=%d\n",
         opts.rank, opts.nranks, collective_name(opts.collective),
         mode_name(opts.mode), opts.kill_rank, opts.root_rank);
  fflush(stdout);
  touch_rank_file(opts.sync_dir, "parent_setup", opts.rank);
  if (run_parent_barrier(parent, stream, opts.timeout_sec) != 0) {
    fprintf(stderr, "rank %d parent communicator barrier failed\n", opts.rank);
    return EXIT_FAILURE;
  }

  int bypass_mode = opts.mode == MODE_BYPASS;
  int survivor_count = bypass_mode ? opts.nranks - 1 : opts.nranks;
  int* survivors = (int*)malloc((size_t)survivor_count * sizeof(int));
  if (survivors == NULL) {
    fprintf(stderr, "rank %d survivor allocation failed\n", opts.rank);
    return EXIT_FAILURE;
  }
  for (int i = 0; i < survivor_count; ++i)
    survivors[i] = bypass_mode ? survivor_rank_at(i, opts.kill_rank) : i;

  int is_survivor = !bypass_mode || opts.rank != opts.kill_rank;
  int sidx = bypass_mode ? (is_survivor ? survivor_index(opts.rank, opts.kill_rank) : -1)
                         : opts.rank;
  int root_sidx = bypass_mode ? survivor_index(opts.root_rank, opts.kill_rank)
                              : opts.root_rank;
  int prev = is_survivor
                 ? (bypass_mode ? survivor_rank_at((sidx + survivor_count - 1) % survivor_count,
                                                   opts.kill_rank)
                                : (opts.rank + opts.nranks - 1) % opts.nranks)
                 : -1;
  int next = is_survivor
                 ? (bypass_mode ? survivor_rank_at((sidx + 1) % survivor_count, opts.kill_rank)
                                : (opts.rank + 1) % opts.nranks)
                 : -1;
  int reporter = is_root_collective(opts.collective) && opts.collective == COLLECTIVE_REDUCE
                     ? opts.root_rank
                     : (bypass_mode ? first_survivor_rank(opts.kill_rank) : 0);

  int send_count = opts.count;
  int recv_count = opts.count;
  if (opts.collective == COLLECTIVE_REDUCESCATTER)
    send_count *= survivor_count;
  if (opts.collective == COLLECTIVE_ALLGATHER)
    recv_count *= survivor_count;
  size_t send_bytes = (size_t)send_count * sizeof(int);
  size_t recv_bytes = (size_t)recv_count * sizeof(int);
  int* h_send = (int*)malloc(send_bytes);
  int* h_observed = (int*)malloc(recv_bytes);
  if (h_send == NULL || h_observed == NULL) {
    fprintf(stderr, "rank %d host allocation failed\n", opts.rank);
    return EXIT_FAILURE;
  }

  int* d_send = NULL;
  int* d_recv = NULL;
  HIPCHECK(hipMalloc(&d_send, send_bytes));
  HIPCHECK(hipMalloc(&d_recv, recv_bytes));

  if (bypass_mode && !is_survivor) {
    printf("FAILED_RANK_SIGKILL rank=%d before_prepare=1\n", opts.rank);
    fflush(stdout);
    touch_rank_file(opts.sync_dir, "killed", opts.rank);
    raise(SIGKILL);
  }

  if (bypass_mode) {
    usleep(1000000);

    NCCLCHECK(call_prepare_survivor_topo(parent, survivor_count, survivors));
    printf("SURVIVOR_NATIVE_PREPARE rank=%d collective=%s survivor_index=%d "
           "prev=%d next=%d survivor_count=%d root_parent=%d root_survivor=%d "
           "after_kill=1\n",
           opts.rank, collective_name(opts.collective), sidx, prev, next,
           survivor_count, opts.root_rank, root_sidx);
    fflush(stdout);
    touch_rank_file(opts.sync_dir, "prepare", opts.rank);

    NCCLCHECK(call_activate_survivor_topo(parent, survivor_count, survivors));
    printf("SURVIVOR_NATIVE_ACTIVATE rank=%d collective=%s survivor_index=%d "
           "prev=%d next=%d survivor_count=%d root_parent=%d root_survivor=%d "
           "after_kill=1\n",
           opts.rank, collective_name(opts.collective), sidx, prev, next,
           survivor_count, opts.root_rank, root_sidx);
    fflush(stdout);
    touch_rank_file(opts.sync_dir, "activate", opts.rank);
  } else {
    printf("NORMAL_NATIVE_START rank=%d collective=%s nranks=%d root_rank=%d\n",
           opts.rank, collective_name(opts.collective), opts.nranks,
           opts.root_rank);
    fflush(stdout);
  }

  int local_result = 0;
  for (int iter = 0; iter < opts.iters && local_result == 0; ++iter) {
    fill_send_host(h_send, send_count, opts.rank);
    HIPCHECK(hipMemcpy(d_send, h_send, send_bytes, hipMemcpyHostToDevice));
    HIPCHECK(hipMemset(d_recv, 0, recv_bytes));
    switch (opts.collective) {
    case COLLECTIVE_ALLREDUCE:
      NCCLCHECK(ncclAllReduce(d_send, d_recv, opts.count, ncclInt, ncclSum,
                              parent, stream));
      break;
    case COLLECTIVE_REDUCESCATTER:
      NCCLCHECK(ncclReduceScatter(d_send, d_recv, opts.count, ncclInt, ncclSum,
                                  parent, stream));
      break;
    case COLLECTIVE_ALLGATHER:
      NCCLCHECK(ncclAllGather(d_send, d_recv, opts.count, ncclInt, parent,
                              stream));
      break;
    case COLLECTIVE_BROADCAST:
      NCCLCHECK(ncclBroadcast(d_send, d_recv, opts.count, ncclInt, root_sidx,
                              parent, stream));
      break;
    case COLLECTIVE_REDUCE:
      NCCLCHECK(ncclReduce(d_send, d_recv, opts.count, ncclInt, ncclSum,
                           root_sidx, parent, stream));
      break;
    default:
      return EXIT_FAILURE;
    }
    local_result = wait_stream_with_timeout(stream, parent,
                                            collective_name(opts.collective),
                                            opts.timeout_sec);
    if (local_result == 0) {
      HIPCHECK(hipMemcpy(h_observed, d_recv, recv_bytes,
                           hipMemcpyDeviceToHost));
      local_result = verify_result(opts, sidx, survivor_count, h_observed);
    }
  }

  printf("%s_NATIVE_DONE rank=%d collective=%s local_result=%d prev=%d "
         "next=%d failed_rank=%d root_parent=%d root_survivor=%d iters=%d "
         "count=%d survivor_index=%d\n",
         bypass_mode ? "SURVIVOR" : "NORMAL", opts.rank,
         collective_name(opts.collective), local_result, prev, next,
         bypass_mode ? opts.kill_rank : -1, opts.root_rank, root_sidx,
         opts.iters, opts.count, sidx);
  fflush(stdout);
  touch_rank_file(opts.sync_dir, local_result == 0 ? "done" : "fail",
                  opts.rank);

  if (dump_ccld_metrics(opts, parent) != 0)
    local_result = local_result == 0 ? 3 : local_result;

  HIPCHECK(hipFree(d_send));
  HIPCHECK(hipFree(d_recv));
  free(h_send);
  free(h_observed);
  free(survivors);

  ncclCommAbort(parent);
  HIPCHECK(hipStreamDestroy(stream));

  if (opts.rank == reporter) {
    printf("RESULT_FRAGMENT rank=%d collective=%s mode=%s result=%s participants=%d "
           "kill_rank=%d root_parent=%d root_survivor=%d iters=%d count=%d\n",
           opts.rank, collective_name(opts.collective), mode_name(opts.mode),
           local_result == 0 ? "LOCAL_PASS" : "LOCAL_FAIL", survivor_count,
           bypass_mode ? opts.kill_rank : -1, opts.root_rank, root_sidx,
           opts.iters, opts.count);
    fflush(stdout);
  }

  return local_result == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
