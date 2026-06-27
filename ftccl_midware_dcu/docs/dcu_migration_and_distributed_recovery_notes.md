# FTCCL Midware DCU Migration and Distributed Recovery Notes

本文记录 FTCCL midware 从 NV 环境迁移到 DCU/RCCL 后，目前已经验证的路径、单节点跑通过程，以及 2 节点/分布式场景暴露的问题和修正方向。

## 目标

FTCCL 的目标不是重新实现 Megatron 训练流程，而是在不修改 Megatron-LM 主体源码的前提下，沿用 NV 版思路完成 DP 维度单 rank 故障绕过：

1. Megatron 正常执行训练。
2. 在计划的 iteration 边界触发 rank kill。
3. CCL-D/控制面发布 bypass 请求。
4. C 层 LD_PRELOAD 中间件让 survivor communicator 切换到新拓扑。
5. Python patch 调整 Megatron 的 DP 视角、dataloader、global batch size 等训练状态。
6. survivor ranks 继续训练到结束。

当前支持范围仍然是 DP-only：

- `TP=1`
- `PP=1`
- `CP=1`
- `DP=N`
- BF16 训练路径
- 不支持 FP8

## DCU 移植范围

DCU 版目录：

```text
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu
```

核心组件：

```text
src/ftccl_midware.cpp
python/ftccl_megatron_patch/
scripts/run_megatron_real_kill_demo.py
scripts/megatron_env.sh
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/run_dp_ftccl.sh
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/train_dp_ftccl.sh
```

迁移原则：

- 保持 NV 版 `LD_PRELOAD + Python sitecustomize patch + real SIGKILL supervisor` 的实现思路。
- C 层继续拦截 NCCL 兼容 ABI；DCU 下实际链接 RCCL/custom-rccl-dtk。
- CUDA runtime 类型替换为 HIP runtime 类型，例如 `cudaStream_t` 替换为 `hipStream_t`。
- 不手工改 Megatron 主流程代码，通过 Python patch 注入训练期行为。
- 通信库使用已经迁移并编译完成的：

```text
/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04
```

## 单节点如何跑通

单节点路径最终跑通的关键点如下。

### 1. 环境加载对齐

早期失败包括：

- `open comgr lib:libamd_comgr.so error`
- `load comgr library error`
- `no ROCm-capable device is detected`
- 登录节点直接看到 `get hyhal driver version error!`

根因是任务在登录节点或远程 ssh 环境里没有正确加载 DTK/ROCm/RCCL 环境。修正方向是让 `run_dp_ftccl.sh`/supervisor 的远程启动环境对齐已验证的 Megatron 脚本 `run_llama8B.sh`：

- 远程节点启动训练。
- source `/public/home/scnethpc26107/sdr/env.sh`。
- 设置 conda env、DTK gcvm lib、custom RCCL、midware `LD_PRELOAD`。
- 不在登录节点直接跑 GPU 训练进程。

### 2. 放弃 mpirun kill 整体作业模型

早期使用 `mpirun` 做 real kill 时，rank 被 SIGKILL 后 Hydra/launcher 会认为作业失败，从而终止整个 job。这与 FTCCL 的目标冲突。

因此切换到 NV 版一致的思路：

- supervisor 独立 ssh 启动每个 rank。
- 每个 rank 写自己的 `pid.<rank>` marker。
- supervisor 在计划 iteration 边界 kill 指定 rank。
- survivor ranks 不依赖 mpirun 作业存活语义。

### 3. 恢复路径

单节点验证中，主链路已经跑通：

1. Python patch 在 iteration 边界写 `pause_ready.<rank>.<iter>`。
2. supervisor 等所有 rank pause 后 kill rank 7。
3. supervisor/control plane 发布：

```text
bypass_request.json
failed_rank
survivors
generation
trigger
killed.7
propagate_error
continue_after_kill.<iter>
```

4. C 层在下一次 collective 中给 survivor 返回一次 recoverable `ncclSystemError`。
5. Python patch 捕获该 recoverable communication error。
6. Python rollback 当前 train step，重放 dataloader microbatches。
7. C 层执行：

```text
ncclCommPrepareSurvivorTopo()
ncclCommActivateSurvivorTopo()
```

8. C 层写 `activate.<rank>` 后，Python patch 才真正切换 Megatron 的有效训练视角。
9. survivor ranks 继续训练到 final iteration。

### 4. 单节点修过的问题

单节点阶段修过的主要问题：

- Python patch 过早把 DP 视角切到 survivor view，后来改为只有 C 层 `activate.<rank>` 后才激活训练视角。
- recoverable error 后需要 rollback 当前 train step，否则 Megatron 内部 rerun state 和 dataloader 状态不一致。
- 短训练最后可能进入 eval/test 导致 `StopIteration`，恢复后默认关闭 eval/test。
- rank 7 被 kill 后默认只有 last rank 打 iteration log，导致后续无 iteration 输出；改成 bypass 后 survivor last rank 输出训练日志。
- 统一日志输出到 `logs/combined.live.log`，减少 per-rank 日志拆散带来的排障困难。

## 2 节点暴露的问题

2 节点后，问题不再是 Megatron 主流程或单机恢复逻辑，而是分布式控制面设计本身不合理。

### 问题 1：N x N 共享文件轮询不可扩展

旧方案中，每个 rank 在 Python rollback barrier 或 C prepare/activate barrier 内部都去共享目录扫描/检查所有 survivor 的 marker。

例如：

- 每个 rank 等所有 `rollback_ready.<rank>.<generation>.<iteration>`。
- 每个 rank 等所有 `prepare.<rank>`。
- 每个 rank 等所有 `activate.<rank>`。

这在单节点可能工作，但在多节点会变成：

```text
N 个 rank * N 个 marker stat/list = O(N^2) 文件系统元数据轮询
```

规模越大，越容易被共享文件系统缓存、元数据延迟、目录一致性和负载放大影响。

### 问题 2：公共控制文件被多个 rank 竞争写

2 节点 run 中出现过：

```text
rank 5 failed to publish .../sync/survivors
```

当时 C 层临时文件名只包含 `pid`：

```text
survivors.tmp.<pid>
```

多节点上 PID 可以重复，所以两个节点上的不同 rank 可能写同一个 tmp 文件，导致：

- tmp 文件互相覆盖。
- `rename` 失败。
- 残留 `survivors.tmp.*`。
- 部分 rank 看到的 `survivors/generation/failed_rank` 状态不一致。

这会进一步导致只有部分 survivor 进入 recoverable rollback，剩余 rank 还卡在旧 collective 或旧状态里。

### 问题 3：公共状态职责不清

这些文件是全局控制面状态：

```text
failed_rank
survivors
generation
bypass_request.json
trigger
propagate_error
```

它们不应该由所有 survivor rank 竞争写。合理职责应是：

- detector 或 supervisor 发布全局状态。
- rank 只发布自己的 per-rank 状态。

## 分布式修正方向

新的方向是把恢复控制面改成 supervisor 聚合放行模型。

### 设计原则

1. rank 只写自己的 marker。
2. supervisor 负责观察所有 survivor marker。
3. supervisor 到齐后发布 release 文件。
4. rank 只等待 release，不再互相扫描所有 rank。
5. 公共全局控制文件尽量单点发布；C 层只在文件缺失时兜底写。
6. 所有临时文件名必须包含 hostname、rank、pid，避免跨节点 PID 冲突。

### 复杂度变化

旧方案：

```text
每个 rank 扫所有 rank marker => O(N^2)
```

新方案：

```text
每个 rank 写 O(1) marker，等待 O(1) release
supervisor 扫 O(N) marker
整体控制面约 O(N)
```

这才适合后续扩大到更多节点。

### 当前实现改动

#### C 层

文件：

```text
src/ftccl_midware.cpp
```

关键改动：

- `touchFile()` / `writeTextFile()` 的 tmp 文件名改为：

```text
<target>.tmp.<hostname>.<rank>.<pid>
```

- 写 marker 后执行：

```text
fflush
fsync(file)
rename
fsync(directory)
```

- 公共文件改成已存在则不重复写：

```text
failed_rank
survivors
generation
```

- C prepare/activate barrier 优先等待 supervisor release：

```text
prepare_release
activate_release
```

如果不是 supervisor launch，仍保留目录扫描/stat fallback，便于 smoke 或非 supervisor 场景。

#### Python patch

文件：

```text
python/ftccl_megatron_patch/state.py
```

关键改动：

- `touch_marker()` tmp 文件名改为：

```text
<marker>.tmp.<hostname>.<rank>.<pid>
```

- 写 marker 后 fsync 文件和目录。
- supervisor launch 模式下，rollback barrier 不再由 rank 扫所有 `rollback_ready`，而是等待：

```text
rollback_release.<generation>.<iteration>
```

#### Supervisor

文件：

```text
scripts/run_megatron_real_kill_demo.py
```

关键改动：

- 增加 release aggregator 线程。
- supervisor 观察所有 survivor 的：

```text
rollback_ready.<rank>.<generation>.<iteration>
prepare.<rank>
activate.<rank>
```

- 到齐后分别发布：

```text
rollback_release.<generation>.<iteration>
prepare_release
activate_release
```

- supervisor 日志记录：

```text
DP_FTCCL_RELEASED phase=<rollback|prepare|activate> ...
```

## 当前状态

已经确认：

- 单节点 real kill 主链路可以跑通。
- 2 节点已经能完成启动、训练第 1 个 iteration、pause、kill rank 7、发布 bypass request。
- 2 节点问题集中在恢复控制面，不是 Megatron 脚本本身，也不是 kill 没触发。
- 原共享文件 barrier 方案不适合分布式，需要改成 supervisor 聚合放行。

最近一次 2 节点问题路径：

```text
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/runs/dp_ftccl_20260623-165955
```

该 run 说明：

- rank 7 已被 kill。
- 部分 survivor 进入 recoverable communication error。
- 共享控制文件写入出现 `survivors` publish 失败。
- 这是公共文件竞争和 tmp 文件名冲突导致的控制面一致性问题。

## 下一步验证

因为 C 层有改动，需要重新编译：

```bash
cd /public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu
module load compiler/dtk/26.04
source /public/home/scnethpc26107/sdr/env.sh
cmake --build build -j
```

然后重新跑 2 节点：

```bash
cd /public/home/scnethpc26107/sdr/Megatron-LM/examples/llama
bash run_dp_ftccl.sh /public/home/scnethpc26107/sdr/hostfile 2 \
  --kill-rank=7 \
  --pause-iter=1 \
  --train-iters=3
```

预期关键证据：

```text
DP_FTCCL_RELEASED phase=rollback generation=1 iteration=1
rollback complete; retrying train_step generation=1 iteration=1
DP_FTCCL_RELEASED phase=prepare generation=1 iteration=1
DP_FTCCL_RELEASED phase=activate generation=1 iteration=1
bypass complete: survivorRank=<...> survivorCount=15
activated survivor training view generation=1
iteration        3/       3
[after training is done]
FTCCL_REAL_KILL_DEMO_PASS
```

如果失败，优先看：

```text
logs/supervisor.log
logs/combined.live.log
sync/
```

关注是否出现：

- `DP_FTCCL_RELEASED phase=rollback`
- `DP_FTCCL_RELEASED phase=prepare`
- `DP_FTCCL_RELEASED phase=activate`
- `failed to publish`
- 残留 `*.tmp.*`
- 缺少某些 `comm_error.<rank>`、`rollback_ready.<rank>`、`prepare.<rank>`、`activate.<rank>`

## 后续建议

当前 supervisor release 仍然以共享文件系统作为传输介质，但通信模式已经从 rank 间 N x N 轮询改成中心聚合。后续如果要支持更大规模，建议继续演进到更明确的控制面服务：

- supervisor 使用 TCP/Unix socket 接收 rank 状态并广播 release；
- 或 detector/CCL-D 提供独立控制通道；
- 共享文件系统只保留审计日志和 run artifact，不作为高频同步 barrier。

在当前阶段，supervisor 聚合放行是最小改动且符合 NV real-kill supervisor 思路的修正方案。

## 当前功能概览

当前 DCU 版 FTCCL midware 已经具备以下功能：

- 通过 `LD_PRELOAD` 拦截 RCCL/NCCL 兼容 C ABI，不需要修改 Megatron-LM 主体源码。
- 支持 DP-only 训练场景下的单 rank 故障绕过，当前目标配置是 `TP=1`、`PP=1`、`CP=1`、`DP=N`。
- 支持计划性 real SIGKILL：supervisor 在指定 iteration 边界 kill 指定 rank。
- 支持 survivor ranks 捕获一次 recoverable `ncclSystemError`，由 Python patch rollback 当前 train step 后继续训练。
- 支持 C 层调用 custom RCCL/FTCCL 的 survivor topology 接口：

```text
ncclCommPrepareSurvivorTopo()
ncclCommActivateSurvivorTopo()
```

- 支持 C 层 activate 后再切换 Megatron 有效训练视角，避免 Python 过早切 DP 状态。
- 支持恢复后调整 Megatron `data_parallel_size`、`global_batch_size`、microbatch calculator。
- 支持恢复后关闭 eval/test，避免短训练在 fault recovery 后进入 eval 导致 iterator 状态问题。
- 支持 bypass 后由 survivor last rank 继续输出 iteration 日志，避免 killed rank 是原 log rank 时训练看起来无输出。
- 支持统一日志 `logs/combined.live.log` 和 supervisor 审计日志 `logs/supervisor.log`。
- 单节点 real kill 主链路已经跑通。
- 2 节点已经验证到训练、pause、kill、bypass request 发布阶段；当前正在修正分布式恢复控制面，目标是通过 supervisor 聚合 release 完成多节点 rollback、prepare、activate。

当前限制：

- 只覆盖 DP-only，不覆盖 TP/PP/CP 组合并行下的 rank 绕过。
- 只覆盖单 rank failure，不覆盖多 rank 同时失败。
- 共享文件系统仍作为当前控制面介质；后续更大规模建议演进为 TCP/CCL-D 独立控制通道。
- C 层有改动后必须重新编译 `libftccl_midware.so`，Python patch 修改则重新启动训练即可生效。
