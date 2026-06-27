# Survivor Collective Bypass + CCL-D

This experiment validates FTCCl survivor-topology bypass on DCU/RCCL and checks
that CCL-D metrics can be written to files.

## Layout

- `src/survivor_collective_after_kill.cpp`: HIP/RCCL demo source.
- `bin/`: compiled demo binary output.
- `env.sh`: FTCCl/RCCL library path and runtime environment variables.
- `hostfile`: two-node hostfile, `8` ranks per node.
- `scripts/run_ccld_16gpu.sh`: normal 16-rank allreduce, used to check CCL-D file output.
- `scripts/run_bypass_16gpu.sh`: survivor-topology bypass test, kills rank `3`.
- `scripts/run_bypass_16gpu_ccld.sh`: combined bypass + CCL-D smoke test.
- `scripts/run_bypass_matrix_16gpu_ccld.sh`: original-style collective matrix.
- `runs/`: generated logs, sync files, unique IDs, and CCL-D dumps.

The run scripts do not compile anything. Build the demo explicitly first.

## Build Demo

Run on a node where compiling is allowed:

```bash
cd /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring
source env.sh
mkdir -p bin

hipcc -std=c++17 -O2 --offload-arch=gfx936 \
  -I${FTCCl_RCCL_INCLUDE} \
  src/survivor_collective_after_kill.cpp \
  -L${FTCCl_RCCL_LIB} -lrccl \
  -ldl \
  -Wl,-rpath,${FTCCl_RCCL_LIB} \
  -o bin/survivor_collective_after_kill
```

The demo resolves FTCCl extension APIs with `dlsym` at runtime:

- `ncclCommPrepareSurvivorTopo`
- `ncclCommActivateSurvivorTopo`
- `ncclCommCcldGetMetrics`

This avoids `hipcc/dcc` link issues with custom RCCL extension symbols.

## Run CCL-D Only

This does not kill any rank. It runs a normal 16-rank `allreduce` and writes
CCL-D metric files through `ncclCommCcldGetMetrics`.

```bash
cd /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring
bash scripts/run_ccld_16gpu.sh
```

Expected markers:

- `PARENT_SETUP`: 16 lines.
- `NORMAL_NATIVE_START`: 16 lines.
- `NORMAL_NATIVE_DONE ... local_result=0`: 16 lines.
- `RESULT_FRAGMENT ... mode=normal result=LOCAL_PASS`: 1 line.
- `CCLD_DUMP`: 16 lines.

Check CCL-D files:

```bash
find runs/ccld_16gpu/allreduce_normal/ccld -name 'ccld.rank*.tsv' | wc -l
head -5 runs/ccld_16gpu/allreduce_normal/ccld/ccld.rank*.tsv
```

Expected file count is `16`.

## Run Bypass Only

This kills parent rank `3`, calls `ncclCommPrepareSurvivorTopo` and
`ncclCommActivateSurvivorTopo` on the remaining 15 ranks, then continues with a
survivor `allreduce` on the old communicator.

```bash
cd /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring
bash scripts/run_bypass_16gpu.sh
```

Expected markers:

- `PARENT_SETUP`: 16 lines.
- `FAILED_RANK_SIGKILL rank=3`: 1 line.
- `SURVIVOR_NATIVE_PREPARE`: 15 lines.
- `SURVIVOR_NATIVE_ACTIVATE`: 15 lines.
- `SURVIVOR_NATIVE_DONE ... local_result=0`: 15 lines.
- `RESULT_FRAGMENT ... mode=bypass result=LOCAL_PASS`: at least 1 line.

The script uses explicit `mpirun`:

```bash
mpirun \
  --hostfile hostfile \
  --map-by ppr:8:node \
  --bind-to none \
  --mca orte_abort_on_non_zero_status 0 \
  --mca mpi_abort_on_nonzero_status 0 \
  -np 16 \
  bash -lc 'source /etc/profile; module load ...; source env.sh; exec bin/survivor_collective_after_kill ...'
```

## Run Combined Bypass + CCL-D

Use this after the two independent tests above pass:

```bash
cd /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring
bash scripts/run_bypass_16gpu_ccld.sh
```

Expected CCL-D file count is `15` because the killed rank does not write a file:

```bash
find runs/bypass_16gpu_ccld/allreduce_kill3/ccld -name 'ccld.rank*.tsv' | wc -l
```

## Run Full Matrix

This covers every failed rank for:

- `allreduce`
- `reducescatter`
- `allgather`
- `broadcast` with every surviving root
- `reduce` with every surviving root

Run only when you want the longer full sweep:

```bash
cd /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring
bash scripts/run_bypass_matrix_16gpu_ccld.sh
```

Each case writes to:

```text
runs/bypass_matrix_16gpu_ccld/<case_name>/
```

## Check Combined Results

Combined-case log:

```bash
LOG=/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring/runs/bypass_16gpu_ccld/allreduce_kill3/run.log

grep -E 'PARENT_SETUP|FAILED_RANK_SIGKILL|SURVIVOR_NATIVE_PREPARE|SURVIVOR_NATIVE_ACTIVATE|SURVIVOR_NATIVE_DONE|RESULT_FRAGMENT|CCLD_DUMP|WARN|error' "$LOG"
```

Expected successful markers:

- `PARENT_SETUP`: 16 lines.
- `FAILED_RANK_SIGKILL rank=3`: 1 line.
- `SURVIVOR_NATIVE_PREPARE`: 15 lines.
- `SURVIVOR_NATIVE_ACTIVATE`: 15 lines.
- `SURVIVOR_NATIVE_DONE ... local_result=0`: 15 lines.
- `RESULT_FRAGMENT ... result=LOCAL_PASS`: at least 1 line.
- `CCLD_DUMP`: 15 lines.

CCL-D files:

```bash
find runs/bypass_16gpu_ccld/allreduce_kill3/ccld -name 'ccld.rank*.tsv' | wc -l
head -5 runs/bypass_16gpu_ccld/allreduce_kill3/ccld/ccld.rank*.tsv
```

Expected file count is `15` because the killed rank does not write a CCL-D file.

## Notes From Debugging

1. Build and run are separate.
   The run scripts contain only `mpirun`; they do not compile.

2. Remote ranks need their own runtime environment.
   `mpirun --hostfile` starts processes on remote nodes. Each remote shell must
   load DTK/MPI/RCCL plugin modules and then source `env.sh`:

   ```bash
   source /etc/profile
   module load compiler/dtk/26.04
   module load mpi/hpcx/2.18.0/gcc-8.5.0/shca
   module load app/rccl/shca_rdma_plugins/v8
   source env.sh
   ```

3. Use explicit rank mapping.
   Without slots and mapping, OpenMPI may place all 16 ranks on one node, causing
   RCCL `Duplicate GPU detected` errors.

   Required hostfile:

   ```text
   h07r4n06 slots=8
   h12r1n20 slots=8
   ```

   Required mpirun mapping:

   ```bash
   --map-by ppr:8:node --bind-to none
   ```

4. The failed rank is intentionally killed.
   OpenMPI must not abort the whole job just because the failed rank exits with
   SIGKILL:

   ```bash
   --mca orte_abort_on_non_zero_status 0
   --mca mpi_abort_on_nonzero_status 0
   ```

5. Clean stale run files before each case.
   Old `nccl_unique_id.bin` or sync files can break `ncclCommInitRank`. The
   scripts remove stale id/sync/CCL-D files before launching each case.

6. CCL-D file output is controlled by the demo.
   RCCL exposes in-memory metrics through `ncclCommCcldGetMetrics`; this demo
   writes them to `--ccld-dump-dir`.

7. If the run stops after only a few `SURVIVOR_NATIVE_PREPARE` lines, the
   surviving ranks are stuck inside `ncclCommPrepareSurvivorTopo`. In this DCU
   branch the survivor prepare path must clear stale parent communicator
   `connectSend/connectRecv` masks before adding survivor-ring prev/next
   connections. Otherwise `ncclTransportP2pSetup` can wait on a peer that has
   already been killed. Rebuild RCCL after changes under `custom-rccl-dtk-26.04/src/`.
