# FTCCl DCU/RCCL 修改记录

本文记录 `/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04`
中的 FTCCl DCU 版本改动。原始 RCCL 使用说明仍保留在 `README.md`，本文只说明
FTCCL 相关扩展、DCU/DTK 适配和验证流程。

## 版本基线

- 对标 RCCL 版本：`2.22.3`
- 构建日志版本标识：`2.22.3-master:07a3100`
- 当前源码目录 git HEAD：`112bb96`
- 运行平台：DTK 26.04 / HIP 6.3 / ROCm 6.3.x DCU 环境
- 主要构建输出：

```text
/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/build/librccl.so.1.0
/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/build/include/nccl.h
```

头文件版本宏：

```text
NCCL_MAJOR 2
NCCL_MINOR 22
NCCL_PATCH 3
NCCL_VERSION_CODE 22203
```

RCCL 保持 NCCL 兼容 ABI，因此公开符号仍使用 `nccl*` 命名，例如
`ncclComm_t`、`ncclAllReduce`、`ncclGetUniqueId`。

## 修改目标

本 DCU 分支把 FTCCl 在 NV/NCCL 分支中的核心能力迁移到当前 RCCL 代码基线：

1. 在旧 parent communicator 上支持 survivor topology bypass。
2. 支持 rank 下线后 survivor ranks 继续执行 collective。
3. 暴露 CCL-D 通信指标读取和清零接口。
4. 在 Simple protocol primitive 中采集 send/recv 进度与等待统计。
5. 提供 DCU/HIP 版 16-rank bypass + CCL-D 验证实验。

该迁移不是直接拷贝生成的 DCU 转换树，而是将 FTCCl 行为按当前 RCCL/DTK
代码结构重新落点。

## 主要改动区域

| 区域 | 文件 | 说明 |
| --- | --- | --- |
| Public API | `src/nccl.h.in` | 新增 survivor topology 和 CCL-D metric API。 |
| Communicator 状态 | `src/include/comm.h` | 增加 survivor/bypass 状态和 CCL-D host metrics 指针。 |
| 初始化与扩展 API | `src/init.cc` | 实现 survivor validate/prepare/activate 和 CCL-D get/reset。 |
| Device communicator | `src/include/device.h` | 增加 device 可见的 CCL-D enable flag 和 channel metrics 指针。 |
| Simple protocol probe | `src/device/prims_simple.h` | 在 send/recv/post/wait 路径记录 CCL-D 指标。 |
| 实验程序 | `exp/survivor_collective_ring/` | HIP/RCCL 版 bypass + CCL-D smoke。 |

## Survivor Topology Bypass

### 新增 API

`src/nccl.h.in` 增加以下 NCCL 兼容 RCCL API：

```c
ncclResult_t ncclCommPrepareSurvivorTopo(ncclComm_t comm, int survivorCount,
                                         const int* survivorRanks);

ncclResult_t ncclCommActivateSurvivorTopo(ncclComm_t comm, int survivorCount,
                                          const int* survivorRanks);
```

基本调用约定：

1. 所有 rank 先创建原始 parent communicator。
2. 计划下线的 rank 在进入 survivor collective 前退出。
3. survivor ranks 用 parent-rank 列表调用 `ncclCommPrepareSurvivorTopo`。
4. survivor ranks 调用 `ncclCommActivateSurvivorTopo`。
5. 后续 collective 继续使用旧 communicator handle，但 ring 视图只包含 survivor ranks。

### Prepare 阶段

`ncclCommPrepareSurvivorTopo` 会根据 survivor parent-rank 列表计算新的 ring 邻居：

```c
prev = survivorRanks[(survivorIndex + survivorCount - 1) % survivorCount];
next = survivorRanks[(survivorIndex + 1) % survivorCount];
```

每个 channel 重新连接 survivor ring 的 prev/next peer，并重新执行 P2P setup。

### Activate 阶段

`ncclCommActivateSurvivorTopo` 将现有 communicator 切换到 survivor 视图：

- `comm->nRanks` 更新为 `survivorCount`
- channel ring 的 `prev`、`next`、`index` 更新为 survivor 拓扑
- `ring.userRanks` 按 survivor ring 逻辑顺序重写
- device communicator 的 rank/nRanks/ring metadata 同步更新
- 后续 collective 仍走旧 communicator 句柄，但不再路由到 killed rank

### 修复点

DCU 验证中发现，如果 survivor prepare 复用旧 parent communicator 的连接掩码，
`ncclTransportP2pSetup` 可能等待已经 killed 的 peer。当前分支在 survivor prepare
路径中清理 stale `connectSend/connectRecv` 掩码后再添加 survivor ring 连接。

## CCL-D 指标采集

### 开关

在 communicator 创建前设置：

```bash
export NCCL_CCLD_ENABLE=1
```

内部参数来自：

```c
NCCL_PARAM(CcldEnable, "CCLD_ENABLE", 0);
```

### 新增 API

```c
ncclResult_t ncclCommCcldGetMetrics(const ncclComm_t comm, int maxChannels,
                                    void* outMetrics, size_t outBytes);

ncclResult_t ncclCommCcldResetMetrics(ncclComm_t comm);
```

`GetMetrics` 从 communicator 中拷贝每个 channel 的指标；当 CCL-D 未启用时返回清零指标。
`ResetMetrics` 清空所有 channel counter。

### Metric 字段

每个 channel 的 `ncclCcldChannelMetrics` 包含：

- `sendCount`, `recvCount`
- `sendBytes`, `recvBytes`
- `sendPostCount`, `recvPostCount`
- `sendWaitSpins`, `recvWaitSpins`
- `sendMaxWaitSpins`, `recvMaxWaitSpins`
- `lastSendTimestamp`, `lastRecvTimestamp`, `lastAnyTimestamp`

### Device 采样点

`src/device/prims_simple.h` 在 Simple protocol 中记录：

- recv 完成次数、recv bytes、recv wait spins、最大 recv wait spins
- send 完成次数、send bytes、send post count
- recv post count
- send wait spins、最大 send wait spins
- send/recv/any timestamp

DCU 版本使用 `wall_clock64()` 作为 device timestamp 来源。

当前 CCL-D 指标覆盖 `NCCL_PROTO=SIMPLE`。LL/LL128 还没有接入同等指标。

## DCU/DTK 适配点

- 使用 HIP/DTK 编译路径，测试程序使用 `hipcc`。
- 测试代码使用 HIP runtime API，不使用 CUDA runtime 头文件。
- 链接 RCCL 和 HIP runtime，不链接 `cudart`。
- 公开 API 保持 NCCL 命名，和 RCCL/PyTorch 的 NCCL 兼容层一致。
- 当前训练精度侧重点是 BF16；FP8 不作为本 DCU 分支已验证能力。

## 实验目录

DCU survivor collective 实验位于：

```text
custom-rccl-dtk-26.04/exp/survivor_collective_ring/
```

主要文件：

- `src/survivor_collective_after_kill.cpp`：HIP/RCCL demo，运行时通过 `dlsym` 解析扩展 API。
- `env.sh`：FTCCL/RCCL build 路径和网络环境变量。
- `hostfile`：两节点 hostfile，每节点 8 rank。
- `scripts/run_ccld_16gpu.sh`：正常 16-rank allreduce + CCL-D dump。
- `scripts/run_bypass_16gpu.sh`：kill rank 3 后的 survivor allreduce。
- `scripts/run_bypass_16gpu_ccld.sh`：bypass + CCL-D 组合验证。
- `scripts/run_bypass_matrix_16gpu_ccld.sh`：多 collective / 多 failed-rank 矩阵验证。
- `runs/`：运行日志、sync 文件、unique id 和 CCL-D TSV 输出。

## 构建方式

### RCCL 库

在允许编译的计算节点上执行：

```bash
cd /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04
module load compiler/dtk/26.04

mkdir -p build
cd build
CXX=hipcc cmake ..
make -j 32
```

### Survivor demo

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

扩展 API 可用性检查：

```bash
nm -D /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/build/librccl.so.1.0 | \
  grep -E 'ncclComm(Prepare|Activate)SurvivorTopo|ncclCommCcld(Get|Reset)Metrics'
```

## 验证流程

进入实验目录：

```bash
cd /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring
```

### 1. CCL-D only

```bash
bash scripts/run_ccld_16gpu.sh
```

期望：

- 16 条 `NORMAL_NATIVE_DONE ... local_result=0`
- 1 条 `RESULT_FRAGMENT ... mode=normal result=LOCAL_PASS`
- 16 个 `CCLD_DUMP`
- `runs/ccld_16gpu/allreduce_normal/ccld/` 下有 16 个 TSV

### 2. Bypass only

```bash
bash scripts/run_bypass_16gpu.sh
```

期望：

- `FAILED_RANK_SIGKILL rank=3`
- 15 条 `SURVIVOR_NATIVE_PREPARE`
- 15 条 `SURVIVOR_NATIVE_ACTIVATE`
- 15 条 `SURVIVOR_NATIVE_DONE ... local_result=0`
- `RESULT_FRAGMENT ... mode=bypass result=LOCAL_PASS`

### 3. Bypass + CCL-D

```bash
bash scripts/run_bypass_16gpu_ccld.sh
```

期望：

- rank 3 被 kill
- 15 个 survivor 完成 allreduce
- 15 个 `CCLD_DUMP`
- `runs/bypass_16gpu_ccld/allreduce_kill3/ccld/` 下有 15 个 TSV

### 4. Full matrix

```bash
bash scripts/run_bypass_matrix_16gpu_ccld.sh
```

覆盖：

- `allreduce`
- `reducescatter`
- `allgather`
- `broadcast`，遍历 survivor root
- `reduce`，遍历 survivor root
- 多个 failed rank case

该矩阵较长，建议在 smoke 全部通过后再运行。

## 已验证结果

当前目录中已有运行记录显示：

```text
RCCL version : 2.22.3-master:07a3100
HIP version  : 6.3.26113-6699a1f0
ROCm version : 6.3.3.0-0-c1d64df
```

已完成的关键验证：

- 2 节点 x 8 DCU = 16 ranks
- 正常 allreduce + CCL-D：16 个 CCL-D TSV 输出
- bypass allreduce：rank 3 kill 后 15 个 survivor 继续并 `LOCAL_PASS`
- bypass + CCL-D：rank 3 kill 后 15 个 survivor 继续并写出 15 个 CCL-D TSV

组合验证日志示例：

```text
exp/survivor_collective_ring/runs/bypass_16gpu_ccld/allreduce_kill3/run.log
```

成功标志：

```text
SURVIVOR_NATIVE_PREPARE
SURVIVOR_NATIVE_ACTIVATE
SURVIVOR_NATIVE_DONE ... local_result=0
CCLD_DUMP ...
RESULT_FRAGMENT ... result=LOCAL_PASS
```

## 运行环境注意事项

1. 远端 rank 需要加载完整环境。

   `mpirun --hostfile` 启动远端进程时，每个远端 shell 都需要：

   ```bash
   source /etc/profile
   module load compiler/dtk/26.04
   module load mpi/hpcx/2.18.0/gcc-8.5.0/shca
   module load app/rccl/shca_rdma_plugins/v8
   source env.sh
   ```

2. 必须显式指定 rank 分布。

   hostfile 示例：

   ```text
   h15r3n15 slots=8
   h15r3n16 slots=8
   ```

   mpirun 映射：

   ```bash
   --map-by ppr:8:node --bind-to none
   ```

   否则 OpenMPI 可能把所有进程放在同一节点，导致 RCCL `Duplicate GPU detected`。

3. kill rank 是预期行为。

   需要允许 OpenMPI 不因一个 rank SIGKILL 直接终止整个 job：

   ```bash
   --mca orte_abort_on_non_zero_status 0
   --mca mpi_abort_on_nonzero_status 0
   ```

4. 清理旧文件。

   每次运行前应清理旧 `nccl_unique_id.bin`、sync 文件和 CCL-D dump，避免读到 stale 状态。
   当前脚本已处理这些清理。

5. CCL-D dump 文件由 demo 写出。

   RCCL 只暴露内存指标；`ccld.rank*.tsv` 是实验程序通过
   `ncclCommCcldGetMetrics` 读取后写出的，不是 RCCL 自动落盘。

## 已知边界

- 当前 survivor bypass 首先验证 ring/SIMPLE 路径。
- CCL-D 指标目前接入 `NCCL_PROTO=SIMPLE`，LL/LL128 暂未覆盖。
- 当前 smoke 以单 rank 下线为主，多 rank 同时下线还未作为稳定能力声明。
- FTCCL/Megatron 训练容错需要上层 `ftccl_midware_dcu` 的 `LD_PRELOAD`
  和 Python patch 层配合；本 RCCL 分支只提供底层 communicator bypass 和 CCL-D 指标。
- DCU 环境主要按 BF16 训练路径验证，FP8 不作为已支持结论。

## 相关文档

- `docs/ftccl/dcu_migration_report.md`：更细的源码迁移报告。
- `exp/survivor_collective_ring/README.md`：实验构建、运行、日志检查和调试记录。
- `/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu/README.md`：
  Megatron/PyTorch LD_PRELOAD 中间件和 Python patch 层说明。
