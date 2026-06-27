# FTCCl Megatron DP Fault Tolerance Demo

本目录是当前整理后的 FTCCl 容错 demo 根目录。代码已经拆成两个并列模块：

```text
/home/ubuntu/sdr/ftccl
├── ftccl_lib/       # FTCCl/NCCL fork，提供 survivor topology 和 CCLD metrics 扩展
└── ftccl_midware/   # LD_PRELOAD + Python patch + supervisor demo
```

目标是在不修改 Megatron-LM 核心源码的前提下，为纯 DP 训练演示一个“任意 DP rank 下线后，剩余 ranks 缩小通信域并继续训练”的容错中间件闭环。

## 当前做到的功能

### 1. 底层通信域动态收缩

`ftccl_lib` 提供扩展 NCCL/FTCCl 符号：

```text
ncclCommPrepareSurvivorTopo
ncclCommActivateSurvivorTopo
ncclCommCcldGetMetrics
```

`ftccl_midware/src/ftccl_midware.cu` 通过 `LD_PRELOAD` 拦截 PyTorch/Megatron 常用 NCCL C ABI：

```text
ncclCommInitRank / ncclCommInitRankConfig
ncclGroupStart / ncclGroupEnd
ncclAllReduce / ncclReduceScatter / ncclAllGather / ncclBroadcast / ncclReduce
```

触发 bypass 后，survivor ranks 调用 FTCCL survivor topology API，把原始 8-rank DP communicator 动态收缩成 7-rank communicator。smoke 已验证：

```text
no bypass:          36, 44, 52      survivor_count=8
bypass rank 7:      28, 35, 42      survivor_count=7
bypass rank 3:      32, 39, 46      survivor_count=7
```

这证明不是固定隔离 rank 7，而是可以隔离任意 DP rank。

### 2. 统一 bypass 控制接口

主动下线、被动检测、真实进程死亡都统一通过：

```text
ftccl_midware/scripts/ftccl_bypassctl.py
```

它原子发布：

```text
bypass_request.json
failed_rank
survivors
generation
trigger
killed.<rank>            # 被动故障或 supervisor 确认 rank 已死亡时写入
propagate_error          # 需要测试 rollback/retry 时写入
```

C middleware 和 Python patch 都从同一套 canonical control files 读取状态，避免不同触发路径各写各的控制状态。

### 3. Megatron 上层 survivor DP 视角

`ftccl_midware/python/sitecustomize.py` 会在 Python 启动时自动安装 patch。主要 patch 点在：

```text
ftccl_midware/python/ftccl_megatron_patch/
```

当前实现了：

- bypass 后把 Megatron 看到的 `data_parallel_size` 从 8 改成 7。
- survivor ranks 的 DP rank 映射从 parent rank 映射到 survivor rank。
- 将 `global_batch_size` 默认从 128 重配成 126，满足 `micro_batch_size * survivor_dp_size` 整除。
- 在 iteration 边界重建 7-way survivor dataloader。
- victim rank 在逻辑隔离 demo 中保留进程但不参与后续 NCCL collective。

Megatron 逻辑 bypass demo 已验证：

```text
rank 7 entering logical victim mode=noop
rank 0 bypass complete: survivorRank=0 survivorCount=7
activated survivor training view generation=1 ... dp_size 8->7 global_batch_size 128->126
dataloader rebuilt generation=1 dp_rank=0 dp_size=7
[after training is done]
```

### 4. 真实 SIGKILL rank 容错

`ftccl_midware/scripts/run_megatron_real_kill_demo.py` 不使用 `torchrun` 管理 worker 生命周期，而是用 supervisor 启动 8 个独立 Megatron rank。流程是：

1. 所有 ranks 在受控 iteration 边界写 `pause_ready.<rank>.<iter>`。
2. supervisor 对 victim rank 发送真实 `SIGKILL`。
3. supervisor 调用 `ftccl_bypassctl.py request --mark-killed --propagate-error` 发布同一套 bypass request。
4. survivor ranks 下一次 collective 先收到一次 recoverable NCCL error。
5. Python patch 捕获 `torch.distributed.DistBackendError`，执行 iteration 级 rollback/retry。
6. survivor ranks 进入 7-rank survivor communicator 并继续训练到结束。

当前验证通过：

```text
FTCCL_REAL_KILL_DEMO_PASS killed_rank=7 victim_exit=-9
continue_after_kill received iteration=1
propagating recoverable ncclSystemError
caught recoverable communication error
rollback complete; retrying train_step
bypass complete: survivorRank=0 survivorCount=7
[after training is done]
```

同步目录中也有 survivor 完成 marker：

```text
train_step_done.0.2 ... train_step_done.6.2
rollback_replay.0.1.1 ... rollback_replay.6.1.1
```

### 5. 检测路径：host preflight + FTCCL kernel CCLD metrics

当前实现有两类检测信号：

- `ccld-host-preflight`：在 LD_PRELOAD 层、collective 提交前检查同一 trace 的 ranks 是否都进入了相同 operation signature。
- `ftccl-kernel-ccld`：从 `ncclCommCcldGetMetrics` 读取 FTCCL `ProtoSimple` primitive 中的 send/recv count、bytes、post count、wait spins 和 timestamp。

`run_smoke.sh --detect` 已验证正常路径会写 probe frame，probe 中包含：

```text
schema=ftccl.detect.v1
signature=op=allreduce;count=1024;datatype=7;redop=0;root=0;algo=RING;proto=SIMPLE
ccld_available=1
ccld_send_count=...
ccld_recv_count=...
```

注意：真实 SIGKILL supervisor demo 中默认关闭 detector，避免 host preflight 在计划 kill 前抢先判故障。检测路径和真实 kill 路径都复用同一 bypass control plane，但 demo 入口分开展示。

### 6. 梯度补偿占位策略

Python patch 支持：

```text
FTCCL_PATCH_GRAD_COMPENSATION=survivor_global_batch   # 默认，不额外缩放
FTCCL_PATCH_GRAD_COMPENSATION=old_world_size          # parent_world_size / survivor_count
FTCCL_PATCH_GRAD_COMPENSATION=zero_missing_average    # survivor_count / parent_world_size
FTCCL_PATCH_GRAD_COMPENSATION=custom
FTCCL_PATCH_GRAD_COMPENSATION_SCALE=<float>
```

缩放发生在 Megatron `train_step` 的 `optimizer.step()` 之前，对 `main_grad` / `grad` 做原地缩放，并写入：

```text
grad_compensation.<rank>.<generation>.<iteration>
```

当前这是 optimizer step 前的占位策略，不是 bucket 级或 optimizer-state 感知的生产级梯度修正。

## 技术路线

### C/CUDA 层

- `LD_PRELOAD` interposer 拦截 NCCL C ABI。
- `dlsym(RTLD_NEXT, ...)` 调真实 NCCL/FTCCL 实现。
- 记录 `ncclComm_t -> parent rank/world size/full-size ordinal/generation` 状态。
- 对 NCCL group 做延迟提交和 replay，保证 bypass 在 group collective 真正提交前完成。
- 调 FTCCL survivor topology API 完成 communicator 收缩。
- 对 victim 支持 `noop` / `park` / `exit0`。
- 写 prepare/activate/killed/comm_error/probe 等 marker 到共享控制目录。

### Python 层

- `sitecustomize.py` 自动注入，不改 Megatron 核心文件。
- monkey patch Megatron parallel state、training loop、train step、dataloader builder。
- 在 iteration 边界重建 dataloader 和 microbatch calculator。
- 捕获 recoverable NCCL/ProcessGroupNCCL error，做 survivor barrier、rerun state reset、dataloader rollback/rebuild、当前 train step retry。
- 在 step 前提供梯度补偿占位。

### Supervisor / control plane

- `ftccl_bypassctl.py` 是统一 bypass request 发布入口。
- `run_megatron_real_kill_demo.py` 用独立进程组启动各 rank，避免 torchrun 因单 worker 死亡杀掉所有 survivor。
- supervisor 负责真实 `SIGKILL`、发布 failed rank 和 survivors、放行 survivor 继续训练。

## 构建

先确认 `ftccl_lib/build` 已存在并包含 FTCCL 的 `libnccl.so.2`：

```bash
nm -D /home/ubuntu/sdr/ftccl/ftccl_lib/build/lib/libnccl.so.2 | rg 'ncclComm(Prepare|Activate)SurvivorTopo|ncclCommCcldGetMetrics'
```

构建中间件：

```bash
cd /home/ubuntu/sdr/ftccl
cmake --fresh -S ftccl_midware -B ftccl_midware/build -DFTCCl_NCCL_HOME=/home/ubuntu/sdr/ftccl/ftccl_lib/build
cmake --build ftccl_midware/build -j
```

确认链接到了新布局下的 FTCCL：

```bash
ldd ftccl_midware/build/libftccl_midware.so | rg 'nccl|cuda'
```

期望看到：

```text
libnccl.so.2 => /home/ubuntu/sdr/ftccl/ftccl_lib/build/lib/libnccl.so.2
```

## Demo 入口

### 1. 普通 8-rank all-reduce smoke

```bash
cd /home/ubuntu/sdr/ftccl
./ftccl_midware/scripts/run_smoke.sh --np 8 --sync-dir /tmp/ftccl_smoke_nobypass
```

期望三轮求和：

```text
36, 44, 52
survivor_count=8
```

### 2. bypass 默认 rank 7

```bash
cd /home/ubuntu/sdr/ftccl
./ftccl_midware/scripts/run_smoke.sh --np 8 --bypass --dead-rank 7 --sync-dir /tmp/ftccl_smoke_bypass7
```

期望三轮 survivor 求和：

```text
28, 35, 42
survivor_count=7
```

### 3. bypass 任意 rank，例如 rank 3

```bash
cd /home/ubuntu/sdr/ftccl
./ftccl_midware/scripts/run_smoke.sh --np 8 --bypass --dead-rank 3 --sync-dir /tmp/ftccl_smoke_bypass3
```

期望三轮 survivor 求和：

```text
32, 39, 46
survivor_count=7
```

### 4. 检测/probe 路径 smoke

```bash
cd /home/ubuntu/sdr/ftccl
./ftccl_midware/scripts/run_smoke.sh --np 8 --detect --sync-dir /tmp/ftccl_smoke_detect
```

检查 probe：

```bash
ls /tmp/ftccl_smoke_detect/probe.*
head -80 /tmp/ftccl_smoke_detect/probe.*
```

### 5. Megatron 逻辑 bypass demo

Megatron 启动脚本已经 source 当前路径：

```text
/home/ubuntu/sdr/ftccl/ftccl_midware/scripts/megatron_env.sh
```

运行：

```bash
cd /home/ubuntu/sdr/ftccl
./ftccl_midware/scripts/run_megatron_fault_tolerance_demo.sh /tmp/ftccl_megatron_patch.log
```

成功时输出：

```text
FTCCL_MEGATRON_DEMO_PASS log=/tmp/ftccl_megatron_patch.log
```

关键日志：

```bash
rg 'logical victim|bypass complete|activated survivor|dataloader rebuilt|after training is done' /tmp/ftccl_megatron_patch.log
```

### 6. 真实 rank SIGKILL demo

```bash
cd /home/ubuntu/sdr/ftccl
python3 ./ftccl_midware/scripts/run_megatron_real_kill_demo.py --log-dir /tmp/ftccl_real_kill_logs --sync-dir /tmp/ftccl_bypass_real
```

成功时输出：

```text
FTCCL_REAL_KILL_DEMO_PASS killed_rank=7 victim_exit=-9
```

关键日志：

```bash
rg 'continue_after_kill|propagating recoverable|caught recoverable|rollback complete|bypass complete|after training is done' /tmp/ftccl_real_kill_logs/combined.log
```

## 当前验证记录

在当前布局 `/home/ubuntu/sdr/ftccl` 下已经跑通：

```text
cmake --fresh + cmake --build                              PASS
ldd libftccl_midware.so -> ftccl_lib/build/lib/libnccl.so  PASS
run_smoke.sh --np 8                                        PASS, 36/44/52
run_smoke.sh --np 8 --bypass --dead-rank 7                 PASS, 28/35/42
run_smoke.sh --np 8 --bypass --dead-rank 3                 PASS, 32/39/46
run_smoke.sh --np 8 --detect                               PASS, probe + ccld_available=1
run_megatron_fault_tolerance_demo.sh                       PASS
run_megatron_real_kill_demo.py                             PASS
```

## 当前限制

- 当前只验证纯 DP，`TP=1`、`PP=1`、`CP=1`。
- 当前不支持 ZeRO / Megatron distributed optimizer 场景。
- 真实 kill demo 是受控 iteration 边界 SIGKILL，不是任意 CUDA kernel 中断后的精确恢复。
- detector 还不是完整论文级 CCL-D：当前 host preflight + FTCCL ProtoSimple kernel counters 可以演示故障信号和 probe 框架，但 LL/LL128 和独立常驻 analyzer 还未覆盖。
- 梯度补偿是 step 前占位缩放，不是 bucket 级生产策略。
- Broadcast/Reduce 的 failed root 复杂策略还需要单独扩展验证。

## 下一步建议

1. 将 detector 从 demo 内联 probe/analyze 演进为独立控制面服务，避免和 supervisor 触发路径竞争。
2. 将 CCLD metrics 覆盖扩展到 LL/LL128，并做更严格的 in-flight hang 判定。
3. 只对 Megatron DP communicator 启用 bypass，细分 TP/PP/CP communicator。
4. 将梯度补偿从 step 前缩放推进到 bucket 级或 optimizer-state 感知策略。
5. 把 smoke、Megatron logical bypass、real-kill 三类验证整理成一个 CI 风格的 `run_all_demos.sh`。
