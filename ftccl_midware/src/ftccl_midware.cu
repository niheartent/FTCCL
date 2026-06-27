#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <dlfcn.h>
#include <nccl.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

extern "C" {
ncclResult_t ncclCommPrepareSurvivorTopo(ncclComm_t comm, int survivorCount, const int* survivorRanks);
ncclResult_t ncclCommActivateSurvivorTopo(ncclComm_t comm, int survivorCount, const int* survivorRanks);
}

namespace {

using CommInitRankFn = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int);
using CommInitRankConfigFn = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
using CommAbortFn = ncclResult_t (*)(ncclComm_t);
using CommUserRankFn = ncclResult_t (*)(const ncclComm_t, int*);
using GroupStartFn = ncclResult_t (*)();
using GroupEndFn = ncclResult_t (*)();
using AllReduceFn = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
using ReduceScatterFn = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
using AllGatherFn = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
using BroadcastFn = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
using ReduceFn = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t);
using CcldGetMetricsFn = ncclResult_t (*)(const ncclComm_t, int, void*, size_t);

CommInitRankFn realCommInitRank = nullptr;
CommInitRankConfigFn realCommInitRankConfig = nullptr;
CommAbortFn realCommAbort = nullptr;
CommUserRankFn realCommUserRank = nullptr;
GroupStartFn realGroupStart = nullptr;
GroupEndFn realGroupEnd = nullptr;
AllReduceFn realAllReduce = nullptr;
ReduceScatterFn realReduceScatter = nullptr;
AllGatherFn realAllGather = nullptr;
BroadcastFn realBroadcast = nullptr;
ReduceFn realReduce = nullptr;
CcldGetMetricsFn realCcldGetMetrics = nullptr;

std::mutex gMu;
std::atomic<bool> gBypassStarted{false};
int gFullSizeCommOrdinal = 0;

struct CommState {
  int parentRank = -1;
  int parentNranks = -1;
  int fullSizeOrdinal = 0;
  int deadRank = -1;
  int survivorRank = -1;
  int survivorCount = -1;
  bool bypassed = false;
  bool victimParked = false;
  bool propagatedRecoverableError = false;
  uint64_t collectiveSeq = 0;
};

std::unordered_map<ncclComm_t, CommState> gComms;

enum class OpKind {
  AllReduce,
  ReduceScatter,
  AllGather,
  Broadcast,
  Reduce,
};

struct PendingOp {
  OpKind kind;
  const void* sendbuff = nullptr;
  void* recvbuff = nullptr;
  size_t count = 0;
  ncclDataType_t datatype = ncclFloat32;
  ncclRedOp_t op = ncclSum;
  int root = 0;
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
};

struct ThreadGroupState {
  int depth = 0;
  bool deferred = false;
  std::vector<PendingOp> ops;
};

thread_local ThreadGroupState tlsGroup;
thread_local bool tlsSkipRealOp = false;

template <typename T>
T loadSym(T& slot, const char* name) {
  if (!slot) {
    slot = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
    if (!slot) {
      fprintf(stderr, "[ftccl-midware] failed to resolve %s: %s\n", name, dlerror());
      abort();
    }
  }
  return slot;
}

template <typename T>
T loadOptionalSym(T& slot, const char* name) {
  if (!slot) {
    dlerror();
    slot = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
  }
  return slot;
}

struct CcldChannelMetrics {
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

struct CcldTotals {
  bool available = false;
  uint64_t sendCount = 0;
  uint64_t recvCount = 0;
  uint64_t sendBytes = 0;
  uint64_t recvBytes = 0;
  uint64_t sendPostCount = 0;
  uint64_t recvPostCount = 0;
  uint64_t sendWaitSpins = 0;
  uint64_t recvWaitSpins = 0;
  uint64_t sendMaxWaitSpins = 0;
  uint64_t recvMaxWaitSpins = 0;
  uint64_t lastSendTimestamp = 0;
  uint64_t lastRecvTimestamp = 0;
  uint64_t lastAnyTimestamp = 0;
};

bool envFlag(const char* name, bool def = false) {
  const char* v = getenv(name);
  if (!v) return def;
  return strcmp(v, "0") != 0 && strcasecmp(v, "false") != 0 && strcasecmp(v, "no") != 0;
}

int envInt(const char* name, int def) {
  const char* v = getenv(name);
  return v ? atoi(v) : def;
}

std::string envStr(const char* name, const char* def) {
  const char* v = getenv(name);
  return v ? std::string(v) : std::string(def);
}

bool enabled() {
  return envFlag("FTCCL_BYPASS_ENABLE", true);
}

bool detectEnabled() {
  return enabled() && envFlag("FTCCL_DETECT_ENABLE", false);
}

bool ccldKernelDetectEnabled() {
  return detectEnabled() && envFlag("FTCCL_DETECT_CCLD_ENABLE", true);
}

int logLevel() {
  return envInt("FTCCL_LOG_LEVEL", 1);
}

void logf(int level, const char* fmt, ...) {
  if (logLevel() < level) return;
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[ftccl-midware] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

std::string bypassDir() {
  return envStr("FTCCL_BYPASS_DIR", "/tmp/ftccl_bypass");
}

std::string pathFor(const char* name) {
  return bypassDir() + "/" + name;
}

std::string rankPath(const char* prefix, int rank) {
  char buf[4096];
  snprintf(buf, sizeof(buf), "%s/%s.%d", bypassDir().c_str(), prefix, rank);
  return std::string(buf);
}

std::string detectRunId() {
  std::string runId = envStr("FTCCL_DETECT_RUN_ID", "default");
  for (char& ch : runId) {
    if (ch == '/' || ch == '\\' || ch == ' ' || ch == '\t' || ch == '\n') ch = '_';
  }
  return runId.empty() ? std::string("default") : runId;
}

std::string probePath(int commOrdinal, uint64_t seq, int rank) {
  char buf[4096];
  std::string runId = detectRunId();
  snprintf(buf, sizeof(buf), "%s/probe.%s.%d.%llu.%d",
           bypassDir().c_str(), runId.c_str(), commOrdinal,
           static_cast<unsigned long long>(seq), rank);
  return std::string(buf);
}

bool exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

bool readTextFile(const std::string& path, std::string* out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  std::string data;
  char buf[1024];
  while (true) {
    size_t n = fread(buf, 1, sizeof(buf), f);
    if (n > 0) data.append(buf, n);
    if (n < sizeof(buf)) {
      if (ferror(f)) {
        fclose(f);
        return false;
      }
      break;
    }
  }
  fclose(f);
  *out = data;
  return true;
}

bool readIntFile(const std::string& path, int* value) {
  std::string text;
  if (!readTextFile(path, &text)) return false;
  char* end = nullptr;
  errno = 0;
  long parsed = strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str()) return false;
  *value = static_cast<int>(parsed);
  return true;
}

void ensureDir() {
  mkdir(bypassDir().c_str(), 0777);
}

void touchFile(const std::string& path, int rank) {
  ensureDir();
  char tmp[8192];
  snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path.c_str(), getpid());
  FILE* f = fopen(tmp, "wb");
  if (!f) {
    logf(0, "rank %d failed to create %s", rank, tmp);
    return;
  }
  fprintf(f, "%d\n", rank);
  fclose(f);
  if (rename(tmp, path.c_str()) != 0) {
    logf(0, "rank %d failed to publish %s", rank, path.c_str());
  }
}

void writeTextFile(const std::string& path, const std::string& text, int rank) {
  ensureDir();
  char tmp[8192];
  snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path.c_str(), getpid());
  FILE* f = fopen(tmp, "wb");
  if (!f) {
    logf(0, "rank %d failed to create %s", rank, tmp);
    return;
  }
  fwrite(text.data(), 1, text.size(), f);
  fclose(f);
  if (rename(tmp, path.c_str()) != 0) {
    logf(0, "rank %d failed to publish %s", rank, path.c_str());
  }
}

std::string joinRanks(const std::vector<int>& ranks) {
  std::string out;
  for (size_t i = 0; i < ranks.size(); ++i) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%s%d", i == 0 ? "" : ",", ranks[i]);
    out += buf;
  }
  out += "\n";
  return out;
}

bool waitFile(const std::string& path, int timeoutSec) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
  while (std::chrono::steady_clock::now() < deadline) {
    if (exists(path)) return true;
    usleep(10000);
  }
  return false;
}

bool waitRanks(const char* prefix, const std::vector<int>& ranks, int timeoutSec) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
  while (std::chrono::steady_clock::now() < deadline) {
    bool ok = true;
    for (int r : ranks) {
      if (!exists(rankPath(prefix, r))) {
        ok = false;
        break;
      }
    }
    if (ok) return true;
    usleep(10000);
  }
  return false;
}

bool triggerPresent() {
  if (envFlag("FTCCL_BYPASS_TRIGGER_AT_START", false)) return true;
  return exists(pathFor("trigger"));
}

int parentToSurvivor(int parentRank, int deadRank) {
  if (parentRank == deadRank) return -1;
  return parentRank < deadRank ? parentRank : parentRank - 1;
}

std::vector<int> makeSurvivors(int nranks, int deadRank) {
  std::vector<int> ranks;
  ranks.reserve(nranks > 0 ? nranks - 1 : 0);
  for (int r = 0; r < nranks; ++r) {
    if (r != deadRank) ranks.push_back(r);
  }
  return ranks;
}

uint64_t nowUsec() {
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

const char* opKindName(OpKind kind) {
  switch (kind) {
    case OpKind::AllReduce:
      return "allreduce";
    case OpKind::ReduceScatter:
      return "reducescatter";
    case OpKind::AllGather:
      return "allgather";
    case OpKind::Broadcast:
      return "broadcast";
    case OpKind::Reduce:
      return "reduce";
  }
  return "unknown";
}

std::string opSignature(const PendingOp& op) {
  std::string algo = envStr("NCCL_ALGO", "");
  std::string proto = envStr("NCCL_PROTO", "");
  char buf[512];
  snprintf(buf, sizeof(buf),
           "op=%s;count=%zu;datatype=%d;redop=%d;root=%d;algo=%s;proto=%s",
           opKindName(op.kind), op.count, static_cast<int>(op.datatype),
           static_cast<int>(op.op), op.root, algo.c_str(), proto.c_str());
  return std::string(buf);
}

CcldTotals sampleCcldTotals(ncclComm_t comm) {
  CcldTotals totals;
  if (!ccldKernelDetectEnabled()) return totals;

  auto getMetrics = loadOptionalSym(realCcldGetMetrics, "ncclCommCcldGetMetrics");
  if (!getMetrics) return totals;

  int maxChannels = envInt("FTCCL_DETECT_CCLD_MAX_CHANNELS", 64);
  if (maxChannels <= 0) maxChannels = 64;
  if (maxChannels > 256) maxChannels = 256;
  std::vector<CcldChannelMetrics> metrics(static_cast<size_t>(maxChannels));
  ncclResult_t res = getMetrics(comm, maxChannels, metrics.data(),
                                metrics.size() * sizeof(CcldChannelMetrics));
  if (res != ncclSuccess) return totals;

  totals.available = true;
  for (const CcldChannelMetrics& m : metrics) {
    totals.sendCount += m.sendCount;
    totals.recvCount += m.recvCount;
    totals.sendBytes += m.sendBytes;
    totals.recvBytes += m.recvBytes;
    totals.sendPostCount += m.sendPostCount;
    totals.recvPostCount += m.recvPostCount;
    totals.sendWaitSpins += m.sendWaitSpins;
    totals.recvWaitSpins += m.recvWaitSpins;
    totals.sendMaxWaitSpins = std::max(totals.sendMaxWaitSpins, m.sendMaxWaitSpins);
    totals.recvMaxWaitSpins = std::max(totals.recvMaxWaitSpins, m.recvMaxWaitSpins);
    totals.lastSendTimestamp = std::max(totals.lastSendTimestamp, m.lastSendTimestamp);
    totals.lastRecvTimestamp = std::max(totals.lastRecvTimestamp, m.lastRecvTimestamp);
    totals.lastAnyTimestamp = std::max(totals.lastAnyTimestamp, m.lastAnyTimestamp);
  }
  return totals;
}

bool readProbeUint(const std::string& path, const char* key, uint64_t* value) {
  std::string text;
  if (!readTextFile(path, &text)) return false;
  std::string prefix = std::string(key) + "=";
  size_t pos = text.find(prefix);
  if (pos == std::string::npos) return false;
  pos += prefix.size();
  char* end = nullptr;
  errno = 0;
  unsigned long long parsed = strtoull(text.c_str() + pos, &end, 10);
  if (errno != 0 || end == text.c_str() + pos) return false;
  *value = static_cast<uint64_t>(parsed);
  return true;
}

uint64_t readProbeProgress(const std::string& path) {
  uint64_t sendCount = 0, recvCount = 0, sendBytes = 0, recvBytes = 0;
  bool ok = readProbeUint(path, "ccld_send_count", &sendCount);
  ok = readProbeUint(path, "ccld_recv_count", &recvCount) || ok;
  ok = readProbeUint(path, "ccld_send_bytes", &sendBytes) || ok;
  ok = readProbeUint(path, "ccld_recv_bytes", &recvBytes) || ok;
  if (!ok) return 0;
  return sendCount + recvCount + sendBytes + recvBytes;
}

int chooseCcldStalledRank(const std::vector<int>& ranks, int commOrdinal, uint64_t seq) {
  if (!ccldKernelDetectEnabled()) return -1;
  if (seq <= 1) return -1;

  int zeroDeltaRank = -1;
  int zeroDeltaCount = 0;
  int positiveDeltaCount = 0;
  uint64_t minDelta = static_cast<uint64_t>(std::max(1, envInt("FTCCL_DETECT_CCLD_MIN_DELTA", 1)));

  for (int rank : ranks) {
    std::string curPath = probePath(commOrdinal, seq, rank);
    std::string prevPath = probePath(commOrdinal, seq - 1, rank);
    uint64_t curProgress = readProbeProgress(curPath);
    uint64_t prevProgress = readProbeProgress(prevPath);
    uint64_t delta = curProgress > prevProgress ? curProgress - prevProgress : 0;
    if (delta >= minDelta) {
      positiveDeltaCount++;
    } else {
      zeroDeltaCount++;
      zeroDeltaRank = rank;
    }
  }

  if (zeroDeltaCount == 1 && positiveDeltaCount >= static_cast<int>(ranks.size()) - 1) {
    return zeroDeltaRank;
  }
  return -1;
}

std::string readSignature(const std::string& path) {
  std::string text;
  if (!readTextFile(path, &text)) return "";
  const char* key = "signature=";
  size_t pos = text.find(key);
  if (pos == std::string::npos) return "";
  pos += strlen(key);
  size_t end = text.find('\n', pos);
  return text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
}

uint64_t nextCollectiveSeq(ncclComm_t comm, CommState* out) {
  std::lock_guard<std::mutex> lock(gMu);
  CommState& st = gComms[comm];
  st.collectiveSeq++;
  *out = st;
  return st.collectiveSeq;
}

std::string detectionReasonJson(const char* reason) {
  std::string out;
  for (const char* p = reason; *p; ++p) {
    if (*p == '"' || *p == '\\') out.push_back('\\');
    out.push_back(*p);
  }
  return out;
}

std::vector<int> configuredSurvivors(int nranks, int deadRank) {
  std::string text;
  if (!readTextFile(pathFor("survivors"), &text)) return makeSurvivors(nranks, deadRank);

  std::vector<int> ranks;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    char* end = nullptr;
    long parsed = strtol(item.c_str(), &end, 10);
    if (end != item.c_str()) ranks.push_back(static_cast<int>(parsed));
  }
  return ranks.empty() ? makeSurvivors(nranks, deadRank) : ranks;
}

int configuredDeadRank(const CommState& st) {
  int fileRank = st.deadRank;
  if (readIntFile(pathFor("failed_rank"), &fileRank)) return fileRank;
  return envInt("FTCCL_BYPASS_DEAD_RANK", st.deadRank);
}

std::string configuredGenerationText() {
  std::string text;
  if (readTextFile(pathFor("generation"), &text) && !text.empty()) return text;
  return "1\n";
}

std::string nextGenerationText() {
  int generation = 0;
  readIntFile(pathFor("generation"), &generation);
  return std::to_string(generation + 1) + "\n";
}

void updateState(ncclComm_t comm, const CommState& st);
CommState getOrCreateState(ncclComm_t comm);

void publishDetectorBypass(const CommState& st, int failedRank, const char* reason,
                           const PendingOp* op, uint64_t seq) {
  if (failedRank < 0 || failedRank >= st.parentNranks) return;
  if (exists(pathFor("trigger"))) return;

  std::vector<int> survivors = makeSurvivors(st.parentNranks, failedRank);
  std::string generation = nextGenerationText();
  std::string reasonEscaped = detectionReasonJson(reason);
  const char* detectorName =
      strncmp(reason, "kernel-ccld-", strlen("kernel-ccld-")) == 0
          ? "ftccl-kernel-ccld"
          : "ccld-host-preflight";
  std::string signature = op ? opSignature(*op) : "";
  std::string signatureEscaped = detectionReasonJson(signature.c_str());

  writeTextFile(pathFor("failed_rank"), std::to_string(failedRank) + "\n", st.parentRank);
  writeTextFile(pathFor("survivors"), joinRanks(survivors), st.parentRank);
  writeTextFile(pathFor("generation"), generation, st.parentRank);

  char req[4096];
  snprintf(req, sizeof(req),
           "{\n"
           "  \"schema\": \"ftccl.bypass.v1\",\n"
           "  \"generation\": %d,\n"
           "  \"failed_rank\": %d,\n"
           "  \"world_size\": %d,\n"
           "  \"survivors\": [%s],\n"
           "  \"source\": \"detector\",\n"
           "  \"reason\": \"%s\",\n"
           "  \"detector\": \"%s\",\n"
           "  \"comm_ordinal\": %d,\n"
           "  \"trace_seq\": %llu,\n"
           "  \"operation\": \"%s\",\n"
           "  \"signature\": \"%s\",\n"
           "  \"mark_killed\": %s,\n"
           "  \"propagate_error_before_bypass\": true\n"
           "}\n",
           atoi(generation.c_str()), failedRank, st.parentNranks,
           joinRanks(survivors).c_str(), reasonEscaped.c_str(),
           detectorName, st.fullSizeOrdinal, static_cast<unsigned long long>(seq),
           op ? opKindName(op->kind) : "unknown", signatureEscaped.c_str(),
           envFlag("FTCCL_DETECT_MARK_KILLED", true) ? "true" : "false");

  std::string json(req);
  std::string prettySurvivors;
  for (size_t i = 0; i < survivors.size(); ++i) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%s%d", i == 0 ? "" : ", ", survivors[i]);
    prettySurvivors += buf;
  }
  size_t marker = json.find("[");
  size_t endMarker = json.find("]", marker);
  if (marker != std::string::npos && endMarker != std::string::npos) {
    json.replace(marker + 1, endMarker - marker - 1, prettySurvivors);
  }
  writeTextFile(pathFor("bypass_request.json"), json, st.parentRank);

  if (envFlag("FTCCL_DETECT_MARK_KILLED", true)) {
    touchFile(rankPath("killed", failedRank), st.parentRank);
  }
  if (envFlag("FTCCL_DETECT_PROPAGATE_ERROR", true)) {
    touchFile(pathFor("propagate_error"), st.parentRank);
  }
  touchFile(rankPath("detected_fault", failedRank), st.parentRank);
  touchFile(pathFor("trigger"), st.parentRank);
  gBypassStarted.store(true, std::memory_order_release);

  logf(0,
       "rank %d detector requested bypass failedRank=%d reason=%s commOrdinal=%d seq=%llu",
       st.parentRank, failedRank, reason, st.fullSizeOrdinal,
       static_cast<unsigned long long>(seq));
}

void markRecoverableErrorReturned(ncclComm_t comm, CommState st) {
  st.propagatedRecoverableError = true;
  updateState(comm, st);
  touchFile(rankPath("error_propagated", st.parentRank), st.parentRank);
  touchFile(rankPath("comm_error", st.parentRank), st.parentRank);
}

int chooseMismatchedRank(const std::vector<int>& ranks, int currentRank,
                         const std::unordered_map<int, std::string>& signatures) {
  std::unordered_map<std::string, int> counts;
  for (const auto& item : signatures) counts[item.second]++;
  if (counts.size() <= 1) return -1;

  std::string majority;
  int majorityCount = -1;
  for (const auto& item : counts) {
    if (item.second > majorityCount) {
      majority = item.first;
      majorityCount = item.second;
    }
  }

  for (int rank : ranks) {
    auto it = signatures.find(rank);
    if (it != signatures.end() && it->second != majority && rank != currentRank) return rank;
  }
  for (int rank : ranks) {
    auto it = signatures.find(rank);
    if (it != signatures.end() && it->second != majority) return rank;
  }
  return -1;
}

ncclResult_t runCollectiveProbe(ncclComm_t comm, const PendingOp& op, const char* where) {
  if (!detectEnabled()) return ncclSuccess;
  if (triggerPresent() || gBypassStarted.load(std::memory_order_acquire)) return ncclSuccess;

  CommState st = getOrCreateState(comm);
  int expectedWorld = envInt("FTCCL_BYPASS_WORLD_SIZE", envInt("WORLD_SIZE", st.parentNranks));
  if (st.bypassed || st.parentNranks != expectedWorld || st.parentNranks <= 1) return ncclSuccess;
  if (st.fullSizeOrdinal <= 0) return ncclSuccess;

  int minOrdinal = envInt("FTCCL_DETECT_MIN_COMM_ORDINAL",
                          envInt("FTCCL_BYPASS_MIN_COMM_ORDINAL", 1));
  if (st.fullSizeOrdinal < minOrdinal) return ncclSuccess;

  uint64_t seq = nextCollectiveSeq(comm, &st);
  std::string signature = opSignature(op);
  std::vector<int> expected = makeSurvivors(st.parentNranks, -1);
  CcldTotals ccld = sampleCcldTotals(comm);

  std::string probe;
  char header[4096];
  snprintf(header, sizeof(header),
           "signature=%s\n"
           "schema=ftccl.detect.v1\n"
           "rank=%d\n"
           "comm_ordinal=%d\n"
           "seq=%llu\n"
           "where=%s\n"
           "op=%s\n"
           "count=%zu\n"
           "datatype=%d\n"
           "redop=%d\n"
           "root=%d\n"
           "enter_usec=%llu\n"
           "ccld_available=%d\n"
           "ccld_send_count=%llu\n"
           "ccld_recv_count=%llu\n"
           "ccld_send_bytes=%llu\n"
           "ccld_recv_bytes=%llu\n"
           "ccld_send_post_count=%llu\n"
           "ccld_recv_post_count=%llu\n"
           "ccld_send_wait_spins=%llu\n"
           "ccld_recv_wait_spins=%llu\n"
           "ccld_send_max_wait_spins=%llu\n"
           "ccld_recv_max_wait_spins=%llu\n"
           "ccld_last_send_timestamp=%llu\n"
           "ccld_last_recv_timestamp=%llu\n"
           "ccld_last_any_timestamp=%llu\n",
           signature.c_str(), st.parentRank, st.fullSizeOrdinal,
           static_cast<unsigned long long>(seq), where, opKindName(op.kind),
           op.count, static_cast<int>(op.datatype), static_cast<int>(op.op), op.root,
           static_cast<unsigned long long>(nowUsec()),
           ccld.available ? 1 : 0,
           static_cast<unsigned long long>(ccld.sendCount),
           static_cast<unsigned long long>(ccld.recvCount),
           static_cast<unsigned long long>(ccld.sendBytes),
           static_cast<unsigned long long>(ccld.recvBytes),
           static_cast<unsigned long long>(ccld.sendPostCount),
           static_cast<unsigned long long>(ccld.recvPostCount),
           static_cast<unsigned long long>(ccld.sendWaitSpins),
           static_cast<unsigned long long>(ccld.recvWaitSpins),
           static_cast<unsigned long long>(ccld.sendMaxWaitSpins),
           static_cast<unsigned long long>(ccld.recvMaxWaitSpins),
           static_cast<unsigned long long>(ccld.lastSendTimestamp),
           static_cast<unsigned long long>(ccld.lastRecvTimestamp),
           static_cast<unsigned long long>(ccld.lastAnyTimestamp));
  probe = header;
  writeTextFile(probePath(st.fullSizeOrdinal, seq, st.parentRank), probe, st.parentRank);

  int timeoutMs = envInt("FTCCL_DETECT_TIMEOUT_MS", 5000);
  int pollUsec = envInt("FTCCL_DETECT_POLL_USEC", 1000);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

  while (std::chrono::steady_clock::now() < deadline) {
    if (triggerPresent() || gBypassStarted.load(std::memory_order_acquire)) {
      return envFlag("FTCCL_DETECT_PROPAGATE_ERROR", true) ? ncclSystemError : ncclSuccess;
    }

    int present = 0;
    std::vector<int> missing;
    std::unordered_map<int, std::string> signatures;
    for (int rank : expected) {
      std::string path = probePath(st.fullSizeOrdinal, seq, rank);
      if (!exists(path)) {
        missing.push_back(rank);
        continue;
      }
      present++;
      signatures[rank] = readSignature(path);
    }

    if (present == static_cast<int>(expected.size())) {
      int mismatched = chooseMismatchedRank(expected, st.parentRank, signatures);
      if (mismatched >= 0) {
        publishDetectorBypass(st, mismatched, "operation-signature-mismatch", &op, seq);
        if (envFlag("FTCCL_DETECT_PROPAGATE_ERROR", true)) {
          markRecoverableErrorReturned(comm, st);
          return ncclSystemError;
        }
        return ncclSuccess;
      }
      int ccldStalled = chooseCcldStalledRank(expected, st.fullSizeOrdinal, seq);
      if (ccldStalled >= 0) {
        publishDetectorBypass(st, ccldStalled, "kernel-ccld-progress-stall", &op, seq);
        if (envFlag("FTCCL_DETECT_PROPAGATE_ERROR", true)) {
          markRecoverableErrorReturned(comm, st);
          return ncclSystemError;
        }
        return ncclSuccess;
      }
      return ncclSuccess;
    }
    usleep(pollUsec > 0 ? pollUsec : 1000);
  }

  int failedRank = -1;
  for (int rank : expected) {
    if (!exists(probePath(st.fullSizeOrdinal, seq, rank))) {
      failedRank = rank;
      break;
    }
  }
  if (failedRank >= 0) {
    publishDetectorBypass(st, failedRank, "not-entered-or-slow-rank-timeout", &op, seq);
    if (envFlag("FTCCL_DETECT_PROPAGATE_ERROR", true)) {
      markRecoverableErrorReturned(comm, st);
      return ncclSystemError;
    }
    return ncclSuccess;
  }
  return ncclSuccess;
}

bool shouldPropagateRecoverableError(ncclComm_t comm, const CommState& st, const char* where) {
  if (!envFlag("FTCCL_BYPASS_PROPAGATE_ERROR_BEFORE_BYPASS", false) &&
      !exists(pathFor("propagate_error"))) return false;
  if (st.bypassed || st.propagatedRecoverableError) return false;
  int deadRank = configuredDeadRank(st);
  if (st.parentRank == deadRank) return false;
  if (st.parentNranks <= 1 || deadRank < 0 || deadRank >= st.parentNranks) return false;

  std::string marker = rankPath("error_propagated", st.parentRank);
  if (exists(marker)) return false;

  std::vector<int> survivors = configuredSurvivors(st.parentNranks, deadRank);
  writeTextFile(pathFor("failed_rank"), std::to_string(deadRank) + "\n", st.parentRank);
  writeTextFile(pathFor("survivors"), joinRanks(survivors), st.parentRank);
  writeTextFile(pathFor("generation"), configuredGenerationText(), st.parentRank);
  touchFile(marker, st.parentRank);
  touchFile(rankPath("comm_error", st.parentRank), st.parentRank);

  CommState updated = st;
  updated.deadRank = deadRank;
  updated.propagatedRecoverableError = true;
  updateState(comm, updated);
  gBypassStarted.store(true, std::memory_order_release);

  if (envFlag("FTCCL_BYPASS_ABORT_ON_PROPAGATED_ERROR", false)) {
    auto abortFn = loadSym(realCommAbort, "ncclCommAbort");
    abortFn(comm);
  }

  logf(0,
       "rank %d propagating recoverable ncclSystemError at %s comm=%p deadRank=%d survivors=%zu",
       st.parentRank, where, comm, deadRank, survivors.size());
  return true;
}

CommState getOrCreateState(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(gMu);
  auto it = gComms.find(comm);
  if (it != gComms.end()) return it->second;

  CommState st;
  int rank = -1;
  if (realCommUserRank || dlsym(RTLD_NEXT, "ncclCommUserRank")) {
    auto fn = loadSym(realCommUserRank, "ncclCommUserRank");
    if (fn(comm, &rank) == ncclSuccess) st.parentRank = rank;
  }
  st.parentNranks = envInt("FTCCL_BYPASS_WORLD_SIZE", envInt("WORLD_SIZE", 8));
  st.deadRank = envInt("FTCCL_BYPASS_DEAD_RANK", st.parentNranks - 1);
  gComms[comm] = st;
  return st;
}

void updateState(ncclComm_t comm, const CommState& st) {
  std::lock_guard<std::mutex> lock(gMu);
  gComms[comm] = st;
}

void recordComm(ncclComm_t comm, int nranks, int rank) {
  CommState st;
  st.parentRank = rank;
  st.parentNranks = nranks;
  st.deadRank = envInt("FTCCL_BYPASS_DEAD_RANK", nranks - 1);
  std::lock_guard<std::mutex> lock(gMu);
  int expectedWorld = envInt("FTCCL_BYPASS_WORLD_SIZE", envInt("WORLD_SIZE", nranks));
  if (nranks == expectedWorld) st.fullSizeOrdinal = ++gFullSizeCommOrdinal;
  gComms[comm] = st;
  logf(1, "record comm=%p parentRank=%d parentNranks=%d fullSizeOrdinal=%d deadRank=%d",
       comm, rank, nranks, st.fullSizeOrdinal, st.deadRank);
}

[[noreturn]] void parkVictim(ncclComm_t comm, CommState st) {
  st.victimParked = true;
  updateState(comm, st);
  touchFile(rankPath("killed", st.parentRank), st.parentRank);
  logf(0, "rank %d entering logical victim mode for comm=%p", st.parentRank, comm);

  if (envFlag("FTCCL_BYPASS_ABORT_VICTIM_COMM", true)) {
    auto abortFn = loadSym(realCommAbort, "ncclCommAbort");
    abortFn(comm);
  }

  std::string mode = envStr("FTCCL_BYPASS_VICTIM_MODE", "park");
  if (mode == "exit0") {
    _exit(0);
  }
  while (true) {
    if (exists(pathFor("release_victim"))) _exit(0);
    sleep(1);
  }
}

ncclResult_t performBypassIfNeeded(ncclComm_t comm, const char* where) {
  if (!enabled()) return ncclSuccess;

  tlsSkipRealOp = false;
  CommState st = getOrCreateState(comm);
  st.deadRank = configuredDeadRank(st);
  updateState(comm, st);
  std::string victimMode = envStr("FTCCL_BYPASS_VICTIM_MODE", "park");
  if (gBypassStarted.load(std::memory_order_acquire) && st.parentRank == st.deadRank &&
      victimMode == "noop") {
    tlsSkipRealOp = true;
    return ncclSuccess;
  }
  if (st.bypassed) return ncclSuccess;
  if (!triggerPresent()) return ncclSuccess;

  if (st.parentNranks <= 1 || st.deadRank < 0 || st.deadRank >= st.parentNranks) {
    if (st.parentNranks <= 1) return ncclSuccess;
    logf(0, "invalid bypass config at %s: nranks=%d deadRank=%d", where, st.parentNranks, st.deadRank);
    return ncclInvalidArgument;
  }

  int expectedWorld = envInt("FTCCL_BYPASS_WORLD_SIZE", envInt("WORLD_SIZE", st.parentNranks));
  if (st.parentNranks != expectedWorld) return ncclSuccess;

  int minOrdinal = envInt("FTCCL_BYPASS_MIN_COMM_ORDINAL", 1);
  if (!gBypassStarted.load(std::memory_order_acquire) && st.fullSizeOrdinal > 0 &&
      st.fullSizeOrdinal < minOrdinal) {
    logf(2, "rank %d skip bypass at %s comm=%p ordinal=%d minOrdinal=%d",
         st.parentRank, where, comm, st.fullSizeOrdinal, minOrdinal);
    return ncclSuccess;
  }

  if (shouldPropagateRecoverableError(comm, st, where)) {
    return ncclSystemError;
  }

  if (st.parentRank == st.deadRank) {
    gBypassStarted.store(true, std::memory_order_release);
    st.victimParked = true;
    st.bypassed = true;
    updateState(comm, st);
    touchFile(rankPath("killed", st.parentRank), st.parentRank);
    logf(0, "rank %d entering logical victim mode=%s for comm=%p", st.parentRank, victimMode.c_str(), comm);
    if (victimMode == "noop") {
      tlsSkipRealOp = true;
      return ncclSuccess;
    }
    parkVictim(comm, st);
  }

  std::vector<int> survivors = configuredSurvivors(st.parentNranks, st.deadRank);
  int timeout = envInt("FTCCL_BYPASS_TIMEOUT_SEC", 30);
  logf(0, "rank %d starting bypass at %s comm=%p deadRank=%d survivors=%zu",
       st.parentRank, where, comm, st.deadRank, survivors.size());
  gBypassStarted.store(true, std::memory_order_release);
  writeTextFile(pathFor("failed_rank"), std::to_string(st.deadRank) + "\n", st.parentRank);
  writeTextFile(pathFor("survivors"), joinRanks(survivors), st.parentRank);
  writeTextFile(pathFor("generation"), configuredGenerationText(), st.parentRank);

  if (!waitFile(rankPath("killed", st.deadRank), timeout)) {
    logf(0, "rank %d timed out waiting for victim rank %d", st.parentRank, st.deadRank);
    return ncclSystemError;
  }
  usleep(envInt("FTCCL_BYPASS_SETTLE_USEC", 200000));

  ncclResult_t res = ncclCommPrepareSurvivorTopo(comm, static_cast<int>(survivors.size()), survivors.data());
  if (res != ncclSuccess) {
    logf(0, "rank %d prepare failed: %d", st.parentRank, res);
    return res;
  }
  touchFile(rankPath("prepare", st.parentRank), st.parentRank);
  if (!waitRanks("prepare", survivors, timeout)) {
    logf(0, "rank %d timed out waiting for prepare barrier", st.parentRank);
    return ncclSystemError;
  }

  res = ncclCommActivateSurvivorTopo(comm, static_cast<int>(survivors.size()), survivors.data());
  if (res != ncclSuccess) {
    logf(0, "rank %d activate failed: %d", st.parentRank, res);
    return res;
  }
  touchFile(rankPath("activate", st.parentRank), st.parentRank);
  if (!waitRanks("activate", survivors, timeout)) {
    logf(0, "rank %d timed out waiting for activate barrier", st.parentRank);
    return ncclSystemError;
  }

  st.bypassed = true;
  st.survivorCount = static_cast<int>(survivors.size());
  st.survivorRank = parentToSurvivor(st.parentRank, st.deadRank);
  updateState(comm, st);
  logf(0, "rank %d bypass complete: survivorRank=%d survivorCount=%d",
       st.parentRank, st.survivorRank, st.survivorCount);
  return ncclSuccess;
}

int mapRoot(ncclComm_t comm, int root) {
  CommState st = getOrCreateState(comm);
  if (!st.bypassed) return root;
  int mapped = parentToSurvivor(root, st.deadRank);
  return mapped < 0 ? root : mapped;
}

ncclResult_t replayOp(const PendingOp& op) {
  ncclResult_t probeRes = runCollectiveProbe(op.comm, op, "group-replay");
  if (probeRes != ncclSuccess) return probeRes;
  ncclResult_t res = performBypassIfNeeded(op.comm, "group-replay");
  if (res != ncclSuccess) return res;
  if (tlsSkipRealOp) return ncclSuccess;

  switch (op.kind) {
    case OpKind::AllReduce:
      return loadSym(realAllReduce, "ncclAllReduce")(op.sendbuff, op.recvbuff, op.count, op.datatype, op.op, op.comm, op.stream);
    case OpKind::ReduceScatter:
      return loadSym(realReduceScatter, "ncclReduceScatter")(op.sendbuff, op.recvbuff, op.count, op.datatype, op.op, op.comm, op.stream);
    case OpKind::AllGather:
      return loadSym(realAllGather, "ncclAllGather")(op.sendbuff, op.recvbuff, op.count, op.datatype, op.comm, op.stream);
    case OpKind::Broadcast:
      return loadSym(realBroadcast, "ncclBroadcast")(op.sendbuff, op.recvbuff, op.count, op.datatype, mapRoot(op.comm, op.root), op.comm, op.stream);
    case OpKind::Reduce:
      return loadSym(realReduce, "ncclReduce")(op.sendbuff, op.recvbuff, op.count, op.datatype, op.op, mapRoot(op.comm, op.root), op.comm, op.stream);
  }
  return ncclInternalError;
}

bool deferOp(PendingOp op) {
  if (!enabled() || !tlsGroup.deferred || tlsGroup.depth <= 0) return false;
  tlsGroup.ops.push_back(op);
  return true;
}

}  // namespace

extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId id, int rank) {
  ncclResult_t res = loadSym(realCommInitRank, "ncclCommInitRank")(comm, nranks, id, rank);
  if (res == ncclSuccess && comm && *comm) recordComm(*comm, nranks, rank);
  return res;
}

extern "C" ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId id, int rank, ncclConfig_t* config) {
  ncclResult_t res = loadSym(realCommInitRankConfig, "ncclCommInitRankConfig")(comm, nranks, id, rank, config);
  if (res == ncclSuccess && comm && *comm) recordComm(*comm, nranks, rank);
  return res;
}

extern "C" ncclResult_t ncclGroupStart() {
  if (!enabled()) return loadSym(realGroupStart, "ncclGroupStart")();
  if (tlsGroup.depth++ == 0) {
    tlsGroup.deferred = true;
    tlsGroup.ops.clear();
  }
  return ncclSuccess;
}

extern "C" ncclResult_t ncclGroupEnd() {
  if (!enabled()) return loadSym(realGroupEnd, "ncclGroupEnd")();
  if (tlsGroup.depth <= 0) return ncclInvalidUsage;
  tlsGroup.depth--;
  if (tlsGroup.depth > 0) return ncclSuccess;

  std::vector<PendingOp> ops;
  ops.swap(tlsGroup.ops);
  tlsGroup.deferred = false;

  if (ops.empty()) return ncclSuccess;

  ncclResult_t res = performBypassIfNeeded(ops.front().comm, "group-end");
  if (res != ncclSuccess) return res;

  res = loadSym(realGroupStart, "ncclGroupStart")();
  if (res != ncclSuccess) return res;
  for (const PendingOp& op : ops) {
    res = replayOp(op);
    if (res != ncclSuccess) break;
  }
  ncclResult_t endRes = loadSym(realGroupEnd, "ncclGroupEnd")();
  return res == ncclSuccess ? endRes : res;
}

extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op,
                                      ncclComm_t comm, cudaStream_t stream) {
  PendingOp pending{OpKind::AllReduce, sendbuff, recvbuff, count, datatype, op, 0, comm, stream};
  if (deferOp(pending)) return ncclSuccess;
  ncclResult_t probeRes = runCollectiveProbe(comm, pending, "allreduce");
  if (probeRes != ncclSuccess) return probeRes;
  ncclResult_t res = performBypassIfNeeded(comm, "allreduce");
  if (res != ncclSuccess) return res;
  if (tlsSkipRealOp) return ncclSuccess;
  return loadSym(realAllReduce, "ncclAllReduce")(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

extern "C" ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                          ncclDataType_t datatype, ncclRedOp_t op,
                                          ncclComm_t comm, cudaStream_t stream) {
  PendingOp pending{OpKind::ReduceScatter, sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream};
  if (deferOp(pending)) return ncclSuccess;
  ncclResult_t probeRes = runCollectiveProbe(comm, pending, "reducescatter");
  if (probeRes != ncclSuccess) return probeRes;
  ncclResult_t res = performBypassIfNeeded(comm, "reducescatter");
  if (res != ncclSuccess) return res;
  if (tlsSkipRealOp) return ncclSuccess;
  return loadSym(realReduceScatter, "ncclReduceScatter")(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

extern "C" ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                                      ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  PendingOp pending{OpKind::AllGather, sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream};
  if (deferOp(pending)) return ncclSuccess;
  ncclResult_t probeRes = runCollectiveProbe(comm, pending, "allgather");
  if (probeRes != ncclSuccess) return probeRes;
  ncclResult_t res = performBypassIfNeeded(comm, "allgather");
  if (res != ncclSuccess) return res;
  if (tlsSkipRealOp) return ncclSuccess;
  return loadSym(realAllGather, "ncclAllGather")(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, int root,
                                      ncclComm_t comm, cudaStream_t stream) {
  PendingOp pending{OpKind::Broadcast, sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream};
  if (deferOp(pending)) return ncclSuccess;
  ncclResult_t probeRes = runCollectiveProbe(comm, pending, "broadcast");
  if (probeRes != ncclSuccess) return probeRes;
  ncclResult_t res = performBypassIfNeeded(comm, "broadcast");
  if (res != ncclSuccess) return res;
  if (tlsSkipRealOp) return ncclSuccess;
  return loadSym(realBroadcast, "ncclBroadcast")(sendbuff, recvbuff, count, datatype, mapRoot(comm, root), comm, stream);
}

extern "C" ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op, int root,
                                   ncclComm_t comm, cudaStream_t stream) {
  PendingOp pending{OpKind::Reduce, sendbuff, recvbuff, count, datatype, op, root, comm, stream};
  if (deferOp(pending)) return ncclSuccess;
  ncclResult_t probeRes = runCollectiveProbe(comm, pending, "reduce");
  if (probeRes != ncclSuccess) return probeRes;
  ncclResult_t res = performBypassIfNeeded(comm, "reduce");
  if (res != ncclSuccess) return res;
  if (tlsSkipRealOp) return ncclSuccess;
  return loadSym(realReduce, "ncclReduce")(sendbuff, recvbuff, count, datatype, op, mapRoot(comm, root), comm, stream);
}
