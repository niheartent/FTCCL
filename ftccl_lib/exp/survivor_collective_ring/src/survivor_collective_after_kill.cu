/*************************************************************************
 * Native NCCL survivor collective experiments on an old parent communicator.
 *
 * All original ranks create parent_comm. The selected failed rank then
 * SIGKILLs itself and never enters NCCL again. Survivor ranks call
 * ncclCommPrepareSurvivorTopo() and ncclCommActivateSurvivorTopo(), then run
 * the requested collective on the old parent_comm.
 ************************************************************************/

#include <cuda_runtime.h>
#include <nccl.h>

#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int g_rank = -1;

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "rank %d CUDA error %s:%d '%s' op=%s\n", g_rank,         \
              __FILE__, __LINE__, cudaGetErrorString(e), #cmd);                \
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

struct Options {
  Collective collective = COLLECTIVE_INVALID;
  int nranks = 8;
  int rank = -1;
  int kill_rank = 2;
  int root_rank = -1;
  int count = 1024;
  int iters = 10;
  int timeout_sec = 30;
  const char* id_file = NULL;
  const char* sync_dir = NULL;
};

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
          "  --kill-rank R     Rank to SIGKILL before prepare. Default: 2\n"
          "  --root-rank R     Broadcast/Reduce root as parent rank. "
          "Default: first survivor\n"
          "  --count N         Number of int elements per rank operation. "
          "Default: 1024\n"
          "  --iters N         Survivor collective iterations. Default: 10\n"
          "  --timeout-sec N   Setup/communication timeout. Default: 30\n"
          "  -h, --help        Show this help.\n",
          prog);
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

  if (opts.collective == COLLECTIVE_INVALID || opts.rank < 0 ||
      opts.rank >= opts.nranks || opts.nranks < 2 || opts.kill_rank < 0 ||
      opts.kill_rank >= opts.nranks || opts.root_rank < 0 ||
      opts.root_rank >= opts.nranks || opts.root_rank == opts.kill_rank ||
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
  fclose(f);
  if (rename(tmp, path) != 0) {
    fprintf(stderr, "rank %d failed to publish %s: %s\n", g_rank, path,
            strerror(errno));
    exit(EXIT_FAILURE);
  }
}

static int wait_prefix(const char* sync_dir, const char* prefix, int nranks,
                       int skip_rank, int timeout_sec) {
  double deadline = now_sec() + timeout_sec;
  while (now_sec() < deadline) {
    int seen = 0;
    for (int r = 0; r < nranks; ++r) {
      if (r == skip_rank)
        continue;
      char path[4096];
      snprintf(path, sizeof(path), "%s/%s.%d", sync_dir, prefix, r);
      seen += file_exists(path);
    }
    if (seen == nranks - (skip_rank >= 0 ? 1 : 0))
      return 0;
    usleep(10000);
  }
  fprintf(stderr, "rank %d timed out waiting for %s files\n", g_rank, prefix);
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

static int wait_stream_with_timeout(cudaStream_t stream, ncclComm_t comm,
                                    const char* label, int timeout_sec) {
  cudaEvent_t done;
  CUDACHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
  CUDACHECK(cudaEventRecord(done, stream));
  double deadline = now_sec() + timeout_sec;
  while (now_sec() < deadline) {
    cudaError_t q = cudaEventQuery(done);
    if (q == cudaSuccess) {
      CUDACHECK(cudaEventDestroy(done));
      return 0;
    }
    if (q != cudaErrorNotReady) {
      fprintf(stderr, "rank %d cudaEventQuery failed during %s: %s\n", g_rank,
              label, cudaGetErrorString(q));
      CUDACHECK(cudaEventDestroy(done));
      return 1;
    }
    ncclResult_t async = ncclSuccess;
    ncclResult_t nr = ncclCommGetAsyncError(comm, &async);
    if (nr != ncclSuccess || async != ncclSuccess) {
      fprintf(stderr, "rank %d NCCL async error during %s: call=%s async=%s\n",
              g_rank, label, ncclGetErrorString(nr), ncclGetErrorString(async));
      CUDACHECK(cudaEventDestroy(done));
      return 1;
    }
    usleep(1000);
  }
  fprintf(stderr, "rank %d timed out during %s\n", g_rank, label);
  CUDACHECK(cudaEventDestroy(done));
  return 1;
}

static void fill_send_host(int* data, int count, int rank) {
  for (int i = 0; i < count; ++i)
    data[i] = rank * 100000 + i;
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
        int expected = survivor_rank_at(s, opts.kill_rank) * 100000 + i;
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
  if (opts.collective == COLLECTIVE_REDUCE &&
      sidx != survivor_index(opts.root_rank, opts.kill_rank))
    return 0;

  for (int i = 0; i < opts.count; ++i) {
    int expected;
    if (opts.collective == COLLECTIVE_BROADCAST) {
      expected = opts.root_rank * 100000 + i;
    } else if (opts.collective == COLLECTIVE_REDUCESCATTER) {
      expected = expected_sum_value(sidx * opts.count + i, opts.nranks,
                                    opts.kill_rank);
    } else {
      expected = expected_sum_value(i, opts.nranks, opts.kill_rank);
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
  CUDACHECK(cudaGetDeviceCount(&device_count));
  CUDACHECK(cudaSetDevice(opts.rank % device_count));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  ncclUniqueId id;
  if (opts.rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&id));
    write_unique_id(opts.id_file, &id);
  } else {
    read_unique_id(opts.id_file, &id, opts.timeout_sec);
  }

  ncclComm_t parent = NULL;
  NCCLCHECK(ncclCommInitRank(&parent, opts.nranks, id, opts.rank));
  printf("PARENT_SETUP rank=%d nranks=%d collective=%s kill_rank=%d root_rank=%d\n",
         opts.rank, opts.nranks, collective_name(opts.collective),
         opts.kill_rank, opts.root_rank);
  fflush(stdout);
  touch_rank_file(opts.sync_dir, "parent_setup", opts.rank);
  if (wait_prefix(opts.sync_dir, "parent_setup", opts.nranks, -1,
                  opts.timeout_sec) != 0)
    return EXIT_FAILURE;

  int survivor_count = opts.nranks - 1;
  int* survivors = (int*)malloc((size_t)survivor_count * sizeof(int));
  if (survivors == NULL) {
    fprintf(stderr, "rank %d survivor allocation failed\n", opts.rank);
    return EXIT_FAILURE;
  }
  for (int i = 0; i < survivor_count; ++i)
    survivors[i] = survivor_rank_at(i, opts.kill_rank);

  int is_survivor = opts.rank != opts.kill_rank;
  int sidx = is_survivor ? survivor_index(opts.rank, opts.kill_rank) : -1;
  int root_sidx = survivor_index(opts.root_rank, opts.kill_rank);
  int prev = is_survivor
                 ? survivor_rank_at((sidx + survivor_count - 1) % survivor_count,
                                    opts.kill_rank)
                 : -1;
  int next = is_survivor
                 ? survivor_rank_at((sidx + 1) % survivor_count, opts.kill_rank)
                 : -1;
  int reporter = is_root_collective(opts.collective) && opts.collective == COLLECTIVE_REDUCE
                     ? opts.root_rank
                     : first_survivor_rank(opts.kill_rank);

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
  CUDACHECK(cudaMalloc(&d_send, send_bytes));
  CUDACHECK(cudaMalloc(&d_recv, recv_bytes));

  char killed_path[4096];
  snprintf(killed_path, sizeof(killed_path), "%s/killed.%d", opts.sync_dir,
           opts.kill_rank);
  if (!is_survivor) {
    printf("FAILED_RANK_SIGKILL rank=%d before_prepare=1\n", opts.rank);
    fflush(stdout);
    touch_rank_file(opts.sync_dir, "killed", opts.rank);
    raise(SIGKILL);
  }

  if (wait_file(killed_path, opts.timeout_sec) != 0)
    return EXIT_FAILURE;
  usleep(200000);

  NCCLCHECK(ncclCommPrepareSurvivorTopo(parent, survivor_count, survivors));
  printf("SURVIVOR_NATIVE_PREPARE rank=%d collective=%s survivor_index=%d "
         "prev=%d next=%d survivor_count=%d root_parent=%d root_survivor=%d "
         "after_kill=1\n",
         opts.rank, collective_name(opts.collective), sidx, prev, next,
         survivor_count, opts.root_rank, root_sidx);
  fflush(stdout);
  touch_rank_file(opts.sync_dir, "prepare", opts.rank);
  if (wait_prefix(opts.sync_dir, "prepare", opts.nranks, opts.kill_rank,
                  opts.timeout_sec) != 0)
    return EXIT_FAILURE;

  NCCLCHECK(ncclCommActivateSurvivorTopo(parent, survivor_count, survivors));
  printf("SURVIVOR_NATIVE_ACTIVATE rank=%d collective=%s survivor_index=%d "
         "prev=%d next=%d survivor_count=%d root_parent=%d root_survivor=%d "
         "after_kill=1\n",
         opts.rank, collective_name(opts.collective), sidx, prev, next,
         survivor_count, opts.root_rank, root_sidx);
  fflush(stdout);
  touch_rank_file(opts.sync_dir, "activate", opts.rank);
  if (wait_prefix(opts.sync_dir, "activate", opts.nranks, opts.kill_rank,
                  opts.timeout_sec) != 0)
    return EXIT_FAILURE;

  int local_result = 0;
  for (int iter = 0; iter < opts.iters && local_result == 0; ++iter) {
    fill_send_host(h_send, send_count, opts.rank);
    CUDACHECK(cudaMemcpy(d_send, h_send, send_bytes, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_recv, 0, recv_bytes));
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
      CUDACHECK(cudaMemcpy(h_observed, d_recv, recv_bytes,
                           cudaMemcpyDeviceToHost));
      local_result = verify_result(opts, sidx, survivor_count, h_observed);
    }
  }

  printf("SURVIVOR_NATIVE_DONE rank=%d collective=%s local_result=%d prev=%d "
         "next=%d failed_rank=%d root_parent=%d root_survivor=%d iters=%d "
         "count=%d survivor_index=%d\n",
         opts.rank, collective_name(opts.collective), local_result, prev, next,
         opts.kill_rank, opts.root_rank, root_sidx, opts.iters, opts.count,
         sidx);
  fflush(stdout);
  touch_rank_file(opts.sync_dir, local_result == 0 ? "done" : "fail",
                  opts.rank);

  CUDACHECK(cudaFree(d_send));
  CUDACHECK(cudaFree(d_recv));
  free(h_send);
  free(h_observed);
  free(survivors);

  ncclCommAbort(parent);
  CUDACHECK(cudaStreamDestroy(stream));

  if (opts.rank == reporter) {
    printf("RESULT_FRAGMENT rank=%d collective=%s result=%s survivors=%d "
           "kill_rank=%d root_parent=%d root_survivor=%d iters=%d count=%d\n",
           opts.rank, collective_name(opts.collective),
           local_result == 0 ? "LOCAL_PASS" : "LOCAL_FAIL", survivor_count,
           opts.kill_rank, opts.root_rank, root_sidx, opts.iters, opts.count);
    fflush(stdout);
  }

  return local_result == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
