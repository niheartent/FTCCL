# FTCCl Megatron Bypass Middleware Demo

这个仓库是一个基于 `LD_PRELOAD` 的 FTCCl/FCCL bypass 中间件 demo，目标是在**不侵入 Megatron-LM 源码**的前提下，控制底层 NCCL/FCCL 通信域动态减少一个 DP rank，同时让上层 Megatron 进程仍然以原来的启动参数、原来的 `world_size` 继续训练。

完整容错实现路线，包括 FTCCl/FCCL C middleware 与 Python patch middleware 的职责划分、dataloader 重建、真实 rank 掉线和测试计划，见：

```text
docs/megatron_dp_rank_fault_tolerance_design.md
```

当前 demo 用到的技术栈、关键机制、控制流和测试证据，见：

```text
docs/current_technical_stack.md
```

当前仓库已经包含两层 demo 中间件：

- `src/ftccl_midware.cu`：`LD_PRELOAD` C/CUDA 层，负责 FCCL survivor communicator bypass。
- `python/ftccl_megatron_patch/`：Python runtime patch 层，负责 Megatron survivor DP 视角、`args.data_parallel_size`/microbatch 重配、训练 dataloader 重建。

当前 demo 面向一个受控场景：

- 单机 8 GPU，纯数据并行 DP。
- `TP=1`、`CP=1`、`PP=1`。
- 不开启 ZeRO，不开启 Megatron distributed optimizer。
- 默认 bypass 一个 DP rank，当前默认 victim rank 是 `7`。
- FCCL 使用 survivor collective 路径，当前固定推荐 `NCCL_ALGO=RING`、`NCCL_PROTO=SIMPLE`。
- 梯度修正暂时只保留设计位置，当前优先验证“bypass 后 Megatron 能继续训练”。

## 核心原理

Megatron 和 PyTorch 分布式训练本身并不知道底层通信域被收缩。训练仍然由 `torchrun --nproc_per_node 8` 启动，Megatron 仍然认为自己运行在原始 8 个 DP rank 上。

中间件通过 `LD_PRELOAD` 抢先加载 `libftccl_midware.so`，覆盖常用 NCCL 入口函数：

- `ncclCommInitRank`
- `ncclCommInitRankConfig`
- `ncclGroupStart`
- `ncclGroupEnd`
- `ncclAllReduce`
- `ncclReduceScatter`
- `ncclAllGather`
- `ncclBroadcast`
- `ncclReduce`

这样 PyTorch/Megatron 调 NCCL 时会先进入中间件。中间件记录每个 `ncclComm_t` 的原始 rank、原始 nranks、是否已经 bypass、survivor rank 映射等状态。

当检测到 bypass trigger 后：

1. victim rank 写入 killed 标记。
2. survivor ranks 等待 victim 标记出现。
3. survivor ranks 调用 FCCL 提供的 survivor API：
   - `ncclCommPrepareSurvivorTopo(comm, survivorCount, survivorRanks)`
   - `ncclCommActivateSurvivorTopo(comm, survivorCount, survivorRanks)`
4. 原来的 8-rank NCCL communicator 在底层被切换成 7-rank survivor 拓扑。
5. 后续 NCCL collective 仍然由 Megatron 原路径发出，但实际只在 survivor ranks 上完成。

对 Megatron 来说，Python 进程、dataloader、optimizer、训练 loop、`torchrun` 控制流都还在；对 FCCL/NCCL 通信层来说，真实 collective 域已经少了一个 rank。

## victim rank 的两种模式

逻辑隔离 demo 中默认使用：

```bash
FTCCL_BYPASS_VICTIM_MODE=noop
```

这个模式下，victim rank 不参与后续 NCCL 通信，但 Python 进程仍然存活。它后续再进入被中间件拦截的 NCCL collective 时，中间件直接返回成功，不再调用真实 NCCL。

这样做是为了保持 `torchrun`、Gloo 控制流、Megatron 训练 loop 和 dataloader 不被进程退出打断，先验证最核心的能力：**底层通信域收缩后，Megatron 训练可以继续向前推进**。

真实进程死亡 demo 使用外部 supervisor 启动 8 个独立 Megatron rank，不再依赖 `torchrun` 的 worker 生命周期管理。supervisor 在受控 iteration 边界向 victim rank 发送 `SIGKILL`，然后写入 survivor control files，剩余 7 个 rank 继续训练。

smoke test 中使用：

```bash
FTCCL_BYPASS_VICTIM_MODE=exit0
```

这样 victim rank 会退出，方便 `mpirun` 测试干净结束。

## group 拦截与延迟执行

PyTorch 经常用 NCCL group 包住多个 collective：

```c
ncclGroupStart();
ncclAllReduce(...);
ncclGroupEnd();
```

如果在 group 中间直接改 communicator，容易破坏 NCCL 的调用顺序。中间件因此对 group 做了延迟：

1. 拦截 `ncclGroupStart()`，不立即调用真实 NCCL group start。
2. group 内的 collective 先记录为 pending op。
3. 到 `ncclGroupEnd()` 时，先检查是否需要 bypass。
4. 如需 bypass，先完成 FCCL survivor topo prepare/activate。
5. 再重新调用真实 `ncclGroupStart()`，回放 pending collective，最后调用真实 `ncclGroupEnd()`。

这保证了 bypass 发生在一组 collective 真正提交给 NCCL 之前。

## 统一 Bypass 触发接口

主动下线和被动故障检测都通过同一个控制面接口发布 bypass request：

```bash
cd /home/ubuntu/sdr/demo/ftccl-midware
./scripts/ftccl_bypassctl.py \
  --sync-dir /tmp/ftccl_bypass \
  request \
  --rank 7 \
  --world-size 8 \
  --source manual \
  --reason operator-drain
```

这个命令会原子发布：

```text
bypass_request.json
failed_rank
survivors
generation
trigger
```

C middleware 和 Python patch 都读取这些 canonical files。也就是说：

- 主动下线：operator 或调度器调用 `ftccl_bypassctl.py request --rank N`，victim rank 仍存活时会在下一次被拦截的 NCCL collective 中进入 `noop/park/exit0` 逻辑。
- 被动故障：detector/supervisor 确认 rank 已死亡后调用同一个接口，但加 `--mark-killed`，让 survivor 不再等待 victim 自己写 `killed.<rank>`。
- 需要测试错误传播/rollback 时加 `--propagate-error`，survivor 下一次 collective 会先收到一次 recoverable NCCL error，然后由 Python patch 触发 rollback/retry。

示例：

```bash
# 主动逻辑隔离 rank 7
./scripts/ftccl_bypassctl.py --sync-dir /tmp/ftccl_bypass request --rank 7 --world-size 8

# 被动故障检测：rank 7 已经死亡
./scripts/ftccl_bypassctl.py --sync-dir /tmp/ftccl_bypass_real request \
  --rank 7 --world-size 8 --source detector --reason heartbeat-timeout --mark-killed

# 查看当前请求和各 rank marker
./scripts/ftccl_bypassctl.py --sync-dir /tmp/ftccl_bypass_real status
```

这个接口是本 demo 的统一入口。脚本中手动触发、supervisor 真实 kill、未来 heartbeat detector 都应该只负责调用这个接口，不再各自手写 `failed_rank/survivors/generation/trigger`。

## 检错集成：host preflight + FTCCL kernel metrics

当前实现分成两类信号，不能把第一类单独称为完整 CCL-D：

- `ccld-host-preflight`：`LD_PRELOAD` 层在 collective 提交前做 rank/op 对齐检查。
- `ftccl-kernel-ccld`：FTCCL `ProtoSimple` send/recv primitive 内部埋点，统计每个 channel 的 send/recv count、bytes、post count、wait spins 和最后更新时间戳；中间件读取这些真实内核计数后参与故障判定。

host preflight 在每个被拦截的 collective 真正提交给 NCCL/FCCL 前，为当前 communicator 生成 trace id：

```text
comm_ordinal + collective_seq
```

每个 rank 会在 `FTCCL_BYPASS_DIR` 下写入一条 probe frame：

```text
probe.<run_id>.<comm_ordinal>.<seq>.<rank>
```

probe frame 记录 operation type set，包括 op name、count、datatype、redop、root、`NCCL_ALGO` 和 `NCCL_PROTO`。同一轮 collective 中，detector 会等待所有 parent ranks 发布相同 trace id 的 probe：

- 如果某个 rank 在 `FTCCL_DETECT_TIMEOUT_MS` 内没有进入该 collective，判定为 `not-entered-or-slow-rank-timeout`。
- 如果所有 rank 都进入但 operation signature 不一致，判定为 `operation-signature-mismatch`。
- 一旦定位 failed rank，detector 直接发布 canonical bypass request，并默认写入 `killed.<rank>`、`propagate_error` 和 `detected_fault.<rank>`。

这条路径复用现有恢复链路：当前 collective 返回一次 `ncclSystemError`，Python patch 捕获 recoverable communication error 后 rollback/retry；retry 时 C 层看到 detector 发布的 trigger，调用 `ncclCommPrepareSurvivorTopo` / `ncclCommActivateSurvivorTopo` 完成 survivor bypass。

FTCCL kernel metrics 由 FTCCL 库中的扩展符号提供：

```c
ncclCommCcldGetMetrics(comm, maxChannels, outMetrics, outBytes);
ncclCommCcldResetMetrics(comm);
```

启用后，中间件会把每个 rank 的累计 `ccld_send_count`、`ccld_recv_count`、`ccld_send_bytes`、`ccld_recv_bytes`、`ccld_*_wait_spins` 写入同一个 probe frame。当前 analyzer 使用相邻 trace 的累计指标做 delta：当多数 rank 的 FTCCL primitive 指标有进展、只有一个 rank 没有进展时，发布 `kernel-ccld-progress-stall` bypass request，`detector` 字段为 `ftccl-kernel-ccld`。

默认 Megatron 环境脚本会打开 detector 和 FTCCL kernel metrics：

```bash
NCCL_PROTO=SIMPLE
NCCL_CCLD_ENABLE=1
FTCCL_DETECT_ENABLE=1
FTCCL_DETECT_CCLD_ENABLE=1
FTCCL_DETECT_TIMEOUT_MS=5000
FTCCL_DETECT_MARK_KILLED=1
FTCCL_DETECT_PROPAGATE_ERROR=1
```

可用 smoke 脚本验证无故障情况下的 probe 路径：

```bash
./scripts/run_smoke.sh --np 8 --detect
```

当前已经有真实 FTCCL 内核埋点，但仍不是论文完整 CCL-D：目前只覆盖 `NCCL_PROTO=SIMPLE` 的 primitive，LL/LL128 还未埋点；analyzer 运行在现有 interposer/probe 路径上，不是独立常驻的全量 CCL-D analyzer；如果应用在一次 in-flight collective 后完全不再回到 host interposer，当前中间件无法单靠下一轮 probe 做判定。

## 触发时序

Megatron 训练脚本中默认配置：

```bash
FTCCL_BYPASS_TRIGGER_AT_START=1
FTCCL_BYPASS_MIN_COMM_ORDINAL=2
FTCCL_BYPASS_VICTIM_MODE=noop
```

`FTCCL_BYPASS_TRIGGER_AT_START=1` 表示一开始就允许触发 bypass。

`FTCCL_BYPASS_MIN_COMM_ORDINAL=2` 表示跳过第一个 full-size communicator。实践中第一个 full-size NCCL communicator 往往属于 PyTorch/Megatron 初始化阶段，太早 bypass 容易影响初始化同步；从第二个 full-size communicator 开始触发，可以更早进入训练前后的 DP 通信路径，同时避开最脆弱的初始化窗口。

也可以使用手动触发：

```bash
export FTCCL_BYPASS_TRIGGER_AT_START=0
./scripts/ftccl_bypassctl.py --sync-dir ${FTCCL_BYPASS_DIR} request --rank 7 --world-size 8
```

## 构建

```bash
cd /home/ubuntu/sdr/demo/ftccl-midware
cmake -S . -B build -DFTCCl_NCCL_HOME=/home/ubuntu/sdr/ftccl/build
cmake --build build -j
```

生成的中间件库：

```bash
/home/ubuntu/sdr/demo/ftccl-midware/build/libftccl_midware.so
```

## Smoke 测试

普通 8-rank all-reduce，通过中间件但不触发 bypass：

```bash
cd /home/ubuntu/sdr/demo/ftccl-midware
./scripts/run_smoke.sh --np 8
```

期望结果是 8 个 rank 都参与求和。验证中三轮结果为：

```text
36, 44, 52
```

bypass rank 7：

```bash
cd /home/ubuntu/sdr/demo/ftccl-midware
./scripts/run_smoke.sh --np 8 --bypass --dead-rank 7
```

期望结果是 rank 7 不再参与通信，survivor ranks 为 0-6。验证中三轮结果为：

```text
28, 35, 42
```

这说明底层 all-reduce 确实从 8-rank 求和变成了 7-rank survivor 求和。

## Megatron 验证

训练脚本：

```bash
/home/ubuntu/sdr/Megatron-LM/examples/llama/train_llama3.2_1b.sh
```

该脚本已经 source：

```bash
/home/ubuntu/sdr/demo/ftccl-midware/scripts/megatron_env.sh
```

该环境脚本会同时设置：

```bash
LD_PRELOAD=/home/ubuntu/sdr/demo/ftccl-midware/build/libftccl_midware.so
PYTHONPATH=/home/ubuntu/sdr/demo/ftccl-midware/python:${PYTHONPATH}
FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS=126
```

短步数验证命令：

```bash
cd /home/ubuntu/sdr/Megatron-LM
TRAIN_ITERS=3 EVAL_ITERS=0 FTCCL_LOG_LEVEL=1 \
FTCCL_BYPASS_MIN_COMM_ORDINAL=2 FTCCL_BYPASS_VICTIM_MODE=noop \
timeout 240s bash examples/llama/train_llama3.2_1b.sh \
> /tmp/ftccl_megatron_final.log 2>&1
```

也可以直接运行带日志检查的 demo 脚本：

```bash
cd /home/ubuntu/sdr/demo/ftccl-midware
./scripts/run_megatron_fault_tolerance_demo.sh /tmp/ftccl_megatron_patch.log
```

验证日志：

```bash
/tmp/ftccl_megatron_patch.log
```

关键证据：

```text
rank 7 entering logical victim mode=noop
rank 0-6 bypass complete: survivorCount=7
activated survivor training view generation=1 ... dp_size 8->7 global_batch_size 128->126
dataloader rebuilt generation=1 dp_rank=0 dp_size=7
iteration 1/3 ... lm loss ...
iteration 2/3 ... global batch size: 126 ...
iteration 3/3 ... global batch size: 126 ...
[after training is done]
```

本地验证中，训练命令 exit code 为 `0`，说明 bypass 后 Megatron 不只是没有立即崩溃，而是实际完成了两个训练 iteration 并正常走到训练结束。

## 真实 SIGKILL 验证

真实进程死亡 demo 使用 supervisor 启动 8 个 rank，并在 Megatron 完成第 1 个 iteration 的 checkpoint/exit 判断后杀掉 rank 7。默认还会打开一次性 recoverable communication error 注入，用来验证“通信错误上抛 -> Python rollback -> survivor topo replay”闭环：

```bash
cd /home/ubuntu/sdr/demo/ftccl-midware
./scripts/run_megatron_real_kill_demo.py \
  --log-dir /tmp/ftccl_real_kill_logs \
  --sync-dir /tmp/ftccl_bypass_real
```

成功输出：

```text
FTCCL_REAL_KILL_DEMO_PASS killed_rank=7 victim_exit=-9 log_dir=/tmp/ftccl_real_kill_logs combined=/tmp/ftccl_real_kill_logs/combined.log
```

关键证据：

```text
SIGKILL rank 7
continue_after_kill received iteration=1
activated survivor training view generation=1 ... dp_size 8->7 global_batch_size 128->126
dataloader rebuilt generation=1 dp_rank=0 dp_size=7 consumed_samples=128
propagating recoverable ncclSystemError
caught recoverable communication error
reset Megatron rerun state before fault replay
rollback complete; retrying train_step
rank 0-6 bypass complete: survivorCount=7
[after training is done]
```

同步目录中还会检查每个 survivor 的最后一个 step marker，例如 `train_step_done.0.2` 到 `train_step_done.6.2`，证明 rank 7 被 `SIGKILL` 后剩余 7 个 rank 继续跑完 `TRAIN_ITERS=3`。

如果只想验证原来的“边界 kill 后直接 bypass”，可以关闭错误传播：

```bash
./scripts/run_megatron_real_kill_demo.py --no-error-propagation
```

## 通信中断、错误传播和回滚

当前 demo 的三块机制如下：

- 通信中断：`src/ftccl_midware.cu` 在故障 generation 已发布、但 communicator 还未完成 survivor bypass 时，可以通过 `FTCCL_BYPASS_PROPAGATE_ERROR_BEFORE_BYPASS=1` 对 survivor 的下一次 NCCL collective 返回一次 `ncclSystemError`。这模拟 in-flight collective 被通信层 abort 后向上层返回可恢复错误。
- 错误传播：PyTorch `ProcessGroupNCCL` 会把这个 NCCL 返回值转换成 `torch.distributed.DistBackendError`。Python patch 在 `train_step` wrapper 中识别 NCCL/ProcessGroupNCCL 通信错误，并让所有 survivor 进入 `rollback_ready.<rank>.<generation>.<iteration>` barrier。
- 训练状态回滚：Python patch 丢弃失败 iteration 中未提交的 dataloader 状态，按 survivor DP view 重建 dataloader，重置 Megatron rerun state，然后重试当前 `train_step`。成功后写入 `rollback_replay.<rank>.<generation>.<iteration>` 和 `train_step_done.<rank>.<iteration>`。

这是一版 iteration 级 rollback demo：故障 iteration 的 optimizer step 尚未提交，retry 会重新跑当前 iteration。它不是 bucket 级精确恢复，也没有覆盖任意 CUDA kernel 已经提交后不可撤销的情况。

## 梯度补偿策略

bypass 后默认采用 survivor global batch 语义：

```bash
FTCCL_PATCH_GRAD_COMPENSATION=survivor_global_batch
```

此模式不额外缩放梯度；Python patch 已经把 `args.data_parallel_size` 改为 survivor count，并把 `global_batch_size` 重配为 `126`，后续训练按 7-rank survivor batch 继续。

如果需要验证“保持旧 world size 梯度尺度”的占位策略，可以打开：

```bash
FTCCL_PATCH_GRAD_COMPENSATION=old_world_size
```

Python patch 会在 Megatron `train_step` 的 `optimizer.step()` 前遍历模型参数的 `main_grad`/`grad`，按：

```text
parent_world_size / survivor_count
```

做一次原地缩放，并写入：

```text
grad_compensation.<rank>.<generation>.<iteration>
```

也支持实验性的：

```bash
FTCCL_PATCH_GRAD_COMPENSATION=zero_missing_average
FTCCL_PATCH_GRAD_COMPENSATION=custom
FTCCL_PATCH_GRAD_COMPENSATION_SCALE=1.0
```

`zero_missing_average` 使用 `survivor_count / parent_world_size`，相当于把失败 rank 的梯度视为 0 后再按旧 world size 平均。`custom` 用显式 scale，主要用于验证和消融实验。

## 重要环境变量

```bash
FTCCL_BYPASS_ENABLE=1
FTCCL_BYPASS_DIR=/tmp/ftccl_bypass
FTCCL_BYPASS_WORLD_SIZE=8
FTCCL_BYPASS_DEAD_RANK=7
FTCCL_BYPASS_VICTIM_MODE=noop
FTCCL_BYPASS_TRIGGER_AT_START=1
FTCCL_BYPASS_MIN_COMM_ORDINAL=2
FTCCL_BYPASS_TIMEOUT_SEC=30
FTCCL_LOG_LEVEL=1
FTCCL_PATCH_ENABLE=1
FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS=126
FTCCL_PATCH_VICTIM_DATA_MODE=keep
FTCCL_PATCH_GRAD_COMPENSATION=survivor_global_batch
FTCCL_BYPASS_PROPAGATE_ERROR_BEFORE_BYPASS=1
FTCCL_PATCH_ROLLBACK_ENABLE=1
FTCCL_PATCH_ROLLBACK_MAX_RETRIES=1
```

NCCL/FCCL 推荐固定：

```bash
NCCL_ALGO=RING
NCCL_PROTO=SIMPLE
NCCL_COLLNET_ENABLE=0
TORCH_NCCL_ASYNC_ERROR_HANDLING=0
NCCL_ASYNC_ERROR_HANDLING=0
```

## 当前限制

这个仓库当前是 demo，不是完整生产级 fault tolerance：

- 当前只验证纯 DP，不覆盖 TP/PP/CP 混合并行。
- 当前不支持 ZeRO / distributed optimizer。
- 真实进程死亡当前验证了受控 iteration 边界 `SIGKILL`，以及下一次 collective 上抛 recoverable NCCL error 后的 iteration 级 rollback/retry；还不是任意 CUDA kernel 已提交后的精确中断恢复。
- 梯度补偿当前是 `optimizer.step()` 前的参数梯度缩放占位，尚未实现 bucket 级或 optimizer-state 感知的生产级修正。
- 当前会对后续新建的 full-size NCCL communicator 在首次使用时继续执行 survivor bypass。
- `Broadcast/Reduce` 的 root 映射只实现了基础 parent-rank 到 survivor-rank 转换，尚未做复杂场景验证。
- 当前以 `RING/SIMPLE` survivor collective 路径为主要验证对象。

## 后续完善方向

1. 将梯度补偿从当前 `optimizer.step()` 前缩放占位推进到 bucket 级或 optimizer-state 感知策略。
2. 将 bypass generation 做成显式状态，避免重复日志和重复切换逻辑。
3. 细分 Megatron 中不同 communicator 的角色，只对 DP communicator 启用 bypass。
4. 将真实 victim 进程退出扩展到更通用的上层 rendezvous/store/control-plane 处理。
5. 增加更系统的验证脚本，自动检查 smoke 和 Megatron 日志中的关键证据。
6. 扩展到更复杂的 Megatron 并行配置前，先明确 TP/PP/CP communicator 是否允许收缩。

## Change Log

### 2026-06-10

- 初始化 FTCCl Megatron bypass middleware demo。
- 实现 `LD_PRELOAD` C/CUDA 层 NCCL 拦截，支持 full-size communicator 记录、survivor topo prepare/activate、NCCL group 延迟回放、victim `noop/park/exit0` 模式。
- 实现 Python patch middleware，支持 Megatron DP size/rank survivor view、global batch/microbatch 重配、训练 dataloader 7-rank 重建。
- 增加统一 bypass 触发接口 `scripts/ftccl_bypassctl.py`，主动下线和被动故障检测都发布同一份 bypass request。
- 增加 Megatron 逻辑 bypass demo 和真实 `SIGKILL` supervisor demo。
- 增加 recoverable NCCL error 传播路径，验证通信错误上抛到 PyTorch `DistBackendError`。
- 增加 iteration 级 rollback/retry：通信错误后 survivor barrier、dataloader rebuild、Megatron rerun state reset、当前 `train_step` 重试。
- 增加梯度补偿策略开关，支持默认 survivor global batch 语义，以及 `old_world_size` / `zero_missing_average` / `custom` 梯度缩放占位。
- 验证通过 smoke、bypass smoke、Megatron 真实 rank 7 `SIGKILL` 后 7-rank survivor 继续训练。
