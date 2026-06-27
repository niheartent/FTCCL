# FTCCl RCCL/DCU Migration Report

## Scope

This report records the FTCCl changes ported from the NV NCCL branch into the
DCU RCCL branch at:

- Source reference: `/public/home/scnethpc26107/sdr/ftccl/ftccl_lib`
- DCU target: `/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04`

The migration does not copy the generated DCU conversion tree directly. The
implementation extracts the FTCCl behavior from the NV branch and applies it to
the RCCL codebase that matches the current DCU/DTK platform.

## Modified Areas

| Area | Files | Purpose |
| --- | --- | --- |
| Public API | `src/nccl.h.in` | Add survivor topology and CCL-D metric APIs while preserving RCCL's NCCL-compatible ABI names. |
| Communicator setup | `src/init.cc`, `src/include/comm.h` | Add survivor topology prepare/activate logic and host-side CCL-D metric allocation/access. |
| Device communicator | `src/include/device.h` | Add device-visible CCL-D enable flag and per-channel metric buffer pointer. |
| Simple protocol probes | `src/device/prims_simple.h` | Record send/recv progress, byte counts, post counts, wait spins, and timestamps. |
| Bypass experiment | `exp/survivor_collective_ring/src/`, `test/` | Add HIP/RCCL test source, a plain 16-rank `mpirun` demo, and CCL-D dump-file check. |

## DP Bypass / Survivor Topology

### Public APIs

The DCU branch adds the following NCCL-compatible RCCL APIs in `src/nccl.h.in`:

```c
ncclResult_t ncclCommPrepareSurvivorTopo(ncclComm_t comm, int survivorCount,
                                         const int* survivorRanks);
ncclResult_t ncclCommActivateSurvivorTopo(ncclComm_t comm, int survivorCount,
                                          const int* survivorRanks);
```

The API contract is:

1. All ranks first create the original parent communicator.
2. A planned failed rank exits before it enters RCCL again.
3. Surviving parent ranks pass the surviving parent-rank list to
   `ncclCommPrepareSurvivorTopo`.
4. Surviving ranks call `ncclCommActivateSurvivorTopo`.
5. Subsequent collectives run on the old parent communicator, but the effective
   ring view uses only survivor ranks.

### Validation

`ncclCommValidateSurvivors` in `src/init.cc` checks:

- communicator pointer validity through `CommCheck`;
- `survivorRanks` pointer validity through `PtrCheck((void*)survivorRanks, ...)`
  to match this RCCL branch's non-const pointer checker;
- survivor count range;
- parent-rank bounds;
- duplicate survivor rank entries;
- current rank's survivor index.

Ranks that are not in `survivorRanks` are logged as excluded and do not receive
a survivor index.

### Prepare Phase

`ncclCommPrepareSurvivorTopo` computes each survivor's previous and next parent
rank in the survivor ring:

```c
prev = survivorRanks[(survivorIndex + survivorCount - 1) % survivorCount];
next = survivorRanks[(survivorIndex + 1) % survivorCount];
```

For each channel it calls `ncclTransportP2pConnect` to connect the survivor ring
neighbors, then runs `ncclTransportP2pSetup` with the ring graph and marks
`comm->initAlgoChannels[NCCL_ALGO_RING] = true`.

### Activate Phase

`ncclCommActivateSurvivorTopo` switches the existing communicator view from
parent ranks to survivor ranks:

- updates `comm->nRanks` to `survivorCount`;
- rewrites each channel's `ring.prev`, `ring.next`, and `ring.index`;
- rewrites `ring.userRanks` so logical ring rank order starts from the local
  survivor index;
- copies the new `nRanks`, rank index, ring metadata, and `devRingUserRanks` to
  the device communicator;
- uses this branch's `ncclStrongStreamAcquireUncaptured`,
  `ncclCudaMemcpyAsync`, `ncclStrongStreamSynchronize`, and
  `ncclStrongStreamRelease` flow for device-side updates.

This is the actual bypass behavior: collectives continue using the parent
communicator handle, but the device and host ring topology no longer route
through the killed parent rank.

## CCL-D Probe

### Enable Switch

`src/init.cc` adds:

```c
NCCL_PARAM(CcldEnable, "CCLD_ENABLE", 0);
```

Set `NCCL_CCLD_ENABLE=1` before communicator creation to enable probes.

### Metric Storage

When enabled, communicator initialization allocates a host-visible metric array:

```c
ncclCudaHostCalloc(&comm->ccldMetrics, MAXCHANNELS);
```

The host communicator and device communicator both carry:

- `ccldEnabled`
- `ccldMetrics`

The per-channel metric layout is defined in `src/include/device.h` as
`ncclCcldChannelMetrics`:

- `sendCount`, `recvCount`
- `sendBytes`, `recvBytes`
- `sendPostCount`, `recvPostCount`
- `sendWaitSpins`, `recvWaitSpins`
- `sendMaxWaitSpins`, `recvMaxWaitSpins`
- `lastSendTimestamp`, `lastRecvTimestamp`, `lastAnyTimestamp`

### Read/Reset APIs

`src/nccl.h.in` and `src/init.cc` add:

```c
ncclResult_t ncclCommCcldGetMetrics(const ncclComm_t comm, int maxChannels,
                                    void* outMetrics, size_t outBytes);
ncclResult_t ncclCommCcldResetMetrics(ncclComm_t comm);
```

`GetMetrics` validates the communicator and output buffer, copies up to
`min(maxChannels, MAXCHANNELS)` channel metrics, and returns zeroed metrics when
CCL-D is disabled. `ResetMetrics` clears all channel counters when enabled.

### Device Probe Points

`src/device/prims_simple.h` records Simple protocol events:

- `ccldRecordRecv`: receive completion, receive bytes, receive wait spins,
  maximum receive wait spins, and receive timestamp;
- `ccldRecordSend`: send completion, send bytes, send post count, and send
  timestamp;
- `ccldRecordRecvPost`: receive post count;
- send-side wait loops also update `sendWaitSpins` and `sendMaxWaitSpins`.

The DCU implementation uses `wall_clock64()` for device timestamps instead of
the NV `globaltimer()` mechanism.

## DCU Adaptation Notes

- RCCL retains NCCL-compatible public names (`ncclComm_t`, `ncclAllReduce`,
  `ncclGetUniqueId`, and related symbols), so public call sites keep `nccl*`.
- Test and helper code uses HIP runtime APIs and `hipcc`.
- CUDA runtime headers, `cuda*` APIs, `-lcudart`, and CUDA include/library paths
  are not used in the DCU test path.
- The previous incompatible RCCL tree required platform headers not available in
  the current environment; the final target is `custom-rccl-dtk-26.04`.

## Bypass Test Assets

The DCU survivor collective test is under:

```text
custom-rccl-dtk-26.04/exp/survivor_collective_ring/
```

Files:

- `src/survivor_collective_after_kill.cpp`: HIP/RCCL test program.
- `env.sh`: environment variables for the compiled FTCCl/RCCL path and network
  settings.
- `hostfile`: two-node hostfile with 8 ranks per node.
- `scripts/run_bypass_16gpu_ccld.sh`: plain two-node, 16-rank `mpirun`
  allreduce bypass demo with CCL-D dump output.
- `scripts/run_bypass_matrix_16gpu_ccld.sh`: original-style collective matrix.
- `README.md`: build, run, log-check, and debugging notes.

The current demo covers one `allreduce` smoke case. Parent rank 3 is SIGKILLed
before survivor prepare/activate.
Survivor ranks must print:

- `SURVIVOR_NATIVE_PREPARE`
- `SURVIVOR_NATIVE_ACTIVATE`
- `SURVIVOR_NATIVE_DONE ... local_result=0`
- `RESULT_FRAGMENT ... result=LOCAL_PASS`
- `CCLD_DUMP ...`

CCL-D files are written by the test program through `ncclCommCcldGetMetrics`,
not by the RCCL library automatically.

## Suggested Verification Commands

Run these on a node where building/running is allowed, not on the login node:

```bash
cd /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring

nm -D ../../build/librccl.so.1.0 | rg \
  'ncclComm(Prepare|Activate)SurvivorTopo|ncclCommCcld(Get|Reset)Metrics'

source env.sh

hipcc -std=c++17 -O2 --offload-arch=gfx936 \
  -I${FTCCl_RCCL_INCLUDE} \
  src/survivor_collective_after_kill.cpp \
  -L${FTCCl_RCCL_LIB} -lrccl \
  -ldl \
  -Wl,-rpath,${FTCCl_RCCL_LIB} \
  -o bin/survivor_collective_after_kill

bash scripts/run_bypass_16gpu_ccld.sh
```

The assistant did not run these build or MPI commands on the login node.
