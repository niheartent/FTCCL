# ftccl_midware 工程实现与 Megatron 联调复盘

这份文档总结 `ftccl_midware` 从 FTCCl survivor collective 能力，到 Megatron-LM 纯 DP 容错 demo 的完整实现过程。它不是用户使用说明，而是工程复盘：当时如何拆问题、如何联调 Megatron、踩了哪些坑、每个坑对应做了哪些修改，以及最后如何验证。

当前代码布局：

```text
/home/ubuntu/sdr/ftccl
├── ftccl_lib/       # FTCCl/NCCL fork，提供 survivor topology 与 CCLD metrics 扩展
└── ftccl_midware/   # LD_PRELOAD interposer + Python patch + supervisor demos
```

## 1. 最初目标

目标是一句话：

```text
在不侵入 Megatron-LM 核心源码的前提下，让纯 DP 训练中的任意一个 DP rank 可以被隔离或真实杀死，剩余 DP ranks 缩小 NCCL/FTCCl 通信域后继续训练。
```

这个目标被拆成四层：

1. **FTCCl 通信层**：原始 8-rank communicator 需要动态缩成 7-rank survivor communicator。
2. **LD_PRELOAD C middleware**：PyTorch/Megatron 不改源码，所有 NCCL collective 从 C ABI 层被拦截。
3. **Python Megatron patch**：Megatron 上层仍按 8 进程启动，但 bypass 后要看到 7-rank DP view，并重建 dataloader。
4. **Supervisor/control plane**：主动 bypass、被动检测、真实 SIGKILL 都走统一控制接口，survivor 继续训练。

最终验证目标：

```text
smoke:       all-reduce 从 8-rank sum 变成 7-rank sum
Megatron:    rank 7 逻辑 bypass 后继续训练到 done
real-kill:   supervisor SIGKILL rank 7 后，0-6 rollback/retry 并训练到 done
detect:      host preflight + FTCCL CCLD metrics probe 可写出检测信号
```

## 2. 总体实现路线

### 2.1 为什么必须分成 C 层和 Python 层

一开始最容易误判的是：能不能只用 Python monkey patch 改 Megatron？

结论是不行。

原因：

- PyTorch `ProcessGroupNCCL` 内部已经缓存了 `ncclComm_t`。
- Megatron Python 层能改 `data_parallel_size`、dataloader、训练循环，但不能可靠替换底层已创建 NCCL communicator。
- 真正的通信域收缩必须发生在 NCCL/FTCCl communicator 上。

所以最终分工是：

```text
C middleware:
  拦截 NCCL C ABI
  跟踪 ncclComm_t
  等待/发布 bypass 控制状态
  调 ncclCommPrepareSurvivorTopo / ncclCommActivateSurvivorTopo
  处理 group replay、victim noop/park/exit0、root remap、detect probe

Python patch:
  patch Megatron parallel_state
  patch dataloader builder 和 train loop
  bypass 后重建 7-rank dataloader
  调整 args.data_parallel_size/global_batch_size/num_microbatches
  捕获 recoverable NCCL error
  做 rollback/retry
  提供梯度补偿占位

Supervisor:
  不依赖 torchrun worker 管理
  独立启动 8 个 rank
  在 iteration 边界 SIGKILL victim
  发布统一 bypass request
  验证 survivor 完成
```

### 2.2 关键文件

```text
ftccl_midware/src/ftccl_midware.cu
  LD_PRELOAD NCCL interposer。

ftccl_midware/python/sitecustomize.py
  Python 自动注入入口。

ftccl_midware/python/ftccl_megatron_patch/state.py
  Python 控制状态读取、survivor view、rollback 配置、梯度补偿配置。

ftccl_midware/python/ftccl_megatron_patch/runtime.py
  Megatron runtime 状态、dataloader rebuild、梯度补偿实现。

ftccl_midware/python/ftccl_megatron_patch/installer.py
  monkey patch 安装点：parallel_state、dataloader、train、train_step、checkpoint boundary。

ftccl_midware/scripts/ftccl_bypassctl.py
  统一 bypass 控制面 CLI。

ftccl_midware/scripts/megatron_env.sh
  Megatron 运行时注入 LD_PRELOAD/PYTHONPATH/FTCCL_* 环境变量。

ftccl_midware/scripts/run_smoke.sh
  C 层 smoke：no bypass / bypass / detect probe。

ftccl_midware/scripts/run_megatron_fault_tolerance_demo.sh
  Megatron 逻辑 bypass demo。

ftccl_midware/scripts/run_megatron_real_kill_demo.py
  真实 SIGKILL supervisor demo。

/home/ubuntu/sdr/Megatron-LM/examples/llama/train_llama3.2_1b.sh
  Megatron demo 训练入口，source 当前 ftccl_midware/scripts/megatron_env.sh。
```

## 3. C middleware 是怎么实现的

### 3.1 拦截 NCCL C ABI

`src/ftccl_midware.cu` 覆盖了以下入口：

```text
ncclCommInitRank
ncclCommInitRankConfig
ncclGroupStart
ncclGroupEnd
ncclAllReduce
ncclReduceScatter
ncclAllGather
ncclBroadcast
ncclReduce
```

真实 NCCL/FTCCl 符号通过：

```c++
dlsym(RTLD_NEXT, "ncclAllReduce")
```

获取。

每个 communicator 记录：

```c++
struct CommState {
  int parentRank;
  int parentNranks;
  int fullSizeOrdinal;
  int deadRank;
  int survivorRank;
  int survivorCount;
  bool bypassed;
  bool victimParked;
  bool propagatedRecoverableError;
  uint64_t collectiveSeq;
};
```

`recordComm()` 在 `ncclCommInitRank*` 成功后记录 parent rank、parent nranks、full-size ordinal。

### 3.2 为什么要做 NCCL group 延迟回放

PyTorch 很多 collective 是这样发的：

```c
ncclGroupStart();
ncclAllReduce(...);
ncclGroupEnd();
```

如果在 group 中间切 communicator，会破坏 NCCL 的调用顺序和内部一致性。

所以 middleware 做了延迟：

```text
ncclGroupStart:
  tlsGroup.deferred = true
  pending ops 清空

group 内 collective:
  只记录 PendingOp
  不立即调用真实 NCCL

ncclGroupEnd:
  先 performBypassIfNeeded()
  再真实 ncclGroupStart()
  replay pending ops
  最后真实 ncclGroupEnd()
```

这是能稳定跑 PyTorch/Megatron 的关键点之一。

### 3.3 bypass 主流程

核心函数是：

```text
performBypassIfNeeded(comm, where)
```

流程：

1. 读取 `failed_rank` / `survivors` / `generation`，没有文件时 fallback 到环境变量。
2. 检查当前 communicator 是否是 full-size parent world communicator。
3. 用 `FTCCL_BYPASS_MIN_COMM_ORDINAL` 跳过太早的初始化 communicator。
4. 如果要求先向上抛错，则返回一次 `ncclSystemError`。
5. victim rank 进入 `noop` / `park` / `exit0`。
6. survivor 等待 `killed.<rank>`。
7. 调：

```c++
ncclCommPrepareSurvivorTopo(comm, survivorCount, survivorRanks);
ncclCommActivateSurvivorTopo(comm, survivorCount, survivorRanks);
```

8. survivor ranks 互相等待 `prepare.*` 和 `activate.*` barrier。
9. 写入 `bypassed=true`、`survivorRank`、`survivorCount`。

### 3.4 victim 三种模式

```text
FTCCL_BYPASS_VICTIM_MODE=noop
  victim Python 进程继续活着，但后续 NCCL collective 直接返回 success。
  用于 Megatron 逻辑 bypass demo。

FTCCL_BYPASS_VICTIM_MODE=park
  victim 写 killed marker 后挂起。
  用于观察 survivor 是否独立前进。

FTCCL_BYPASS_VICTIM_MODE=exit0
  victim 直接退出。
  用于 mpirun smoke 测试，方便干净结束。
```

### 3.5 root remap

`Broadcast` / `Reduce` 多了 root 参数。bypass 后 survivor rank 编号不是 parent rank 编号。

当前基础策略：

```text
parent rank < deadRank: survivor rank = parent rank
parent rank > deadRank: survivor rank = parent rank - 1
parent rank == deadRank: -1
```

`mapRoot()` 会把 surviving root 从 parent rank 映射到 survivor rank。failed root 的复杂策略尚未生产化，只做了基础处理。

### 3.6 检测 probe 和 CCLD metrics

后来又加了 detection 路径：

```text
runCollectiveProbe()
```

每个 collective 提交前写：

```text
probe.<run_id>.<comm_ordinal>.<seq>.<rank>
```

probe 中记录：

```text
signature=op=allreduce;count=...;datatype=...;redop=...;root=...;algo=...;proto=...
schema=ftccl.detect.v1
rank=...
comm_ordinal=...
seq=...
ccld_available=...
ccld_send_count=...
ccld_recv_count=...
ccld_send_bytes=...
ccld_recv_bytes=...
```

两类判定：

```text
host preflight:
  某 rank 没有进入同一个 collective trace
  或 operation signature 不一致

kernel CCLD:
  从 ncclCommCcldGetMetrics 读取 FTCCL ProtoSimple primitive counters
  相邻 trace 之间多数 ranks 有进展，只有一个 rank 无 delta，则判 kernel-ccld-progress-stall
```

检测到失败后不直接特殊处理，而是复用统一 control plane：

```text
publishDetectorBypass()
  写 failed_rank/survivors/generation/trigger/bypass_request.json
  可选写 killed.<rank>
  可选写 propagate_error
```

## 4. Python patch 是怎么接入 Megatron 的

### 4.1 为什么用 sitecustomize

目标是不改 Megatron 核心源码，所以用了：

```text
ftccl_midware/python/sitecustomize.py
```

只要 `PYTHONPATH` 包含 `ftccl_midware/python`，Python 启动时会自动 import `sitecustomize`，然后：

```python
import ftccl_megatron_patch
ftccl_megatron_patch.install()
```

`megatron_env.sh` 负责设置：

```bash
export PYTHONPATH="${FTCCL_MIDWARE_DIR}/python:${PYTHONPATH:-}"
```

### 4.2 patch parallel_state

Megatron dataloader 和训练逻辑大量依赖：

```python
megatron.core.parallel_state.get_data_parallel_world_size()
megatron.core.parallel_state.get_data_parallel_rank()
```

patch 后：

```text
未 bypass:
  返回 Megatron 原始值。

bypass active:
  get_data_parallel_world_size -> survivor_count
  get_data_parallel_rank -> survivor_ranks.index(global_rank)
  victim -> 返回 0 避免上层断言炸掉
```

这一步解决“Megatron 上层仍以为 DP size=8”的问题。

### 4.3 patch dataloader builder

Megatron 构造 pretraining dataloader 时会按 DP size/rank 做数据切分。

patch 点：

```text
megatron.training.datasets.data_samplers.build_pretraining_data_loader
megatron.training.training.build_pretraining_data_loader
```

做的事：

1. 捕获 train dataset。
2. 用 `FaultTolerantDataIterator` 包装原 iterator。
3. bypass generation 变化时，在下一次取 batch 前重建 dataloader。
4. 重建时 parallel_state 已经返回 survivor DP view，所以 dataloader 变成 7-way shard。

关键日志：

```text
dataloader rebuilt generation=1 dp_rank=0 dp_size=7 consumed_samples=...
```

### 4.4 patch train / train_step

patch 点：

```text
megatron.training.training.train
megatron.training.training.train_step
megatron.training.training.checkpoint_and_decide_exit
```

`train` patch 负责包装 `train_data_iterator`。

`train_step` patch 负责：

```text
每个 step 前:
  apply_effective_training_args()
  data_iterator.begin_iteration()

成功后:
  train_step_done.<rank>.<iteration>

捕获 recoverable comm error:
  rollback_ready.<rank>.<generation>.<iteration>
  等所有 survivor rollback_ready
  reset Megatron rerun state
  dataloader rollback/rebuild
  rollback_replay.<rank>.<generation>.<iteration>
  retry 当前 train_step
```

`checkpoint_and_decide_exit` patch 用来做真实 kill demo 的受控 iteration 边界：

```text
pause_ready.<rank>.<iteration>
等待 continue_after_kill.<iteration>
```

### 4.5 调整 Megatron args 和 microbatch

bypass 后调用：

```python
args.data_parallel_size = view.survivor_count
args.global_batch_size = FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS  # 默认 126
reconfigure_num_microbatches_calculator(...)
update_num_microbatches(...)
```

为什么 global batch 从 128 改成 126？

```text
原来 DP size = 8, micro batch = 1, global batch = 128，可以整除 8。
bypass 后 DP size = 7，128 不能整除 7。
因此 demo 默认改成 126，满足 126 / (1 * 7) = 18 microbatches。
```

### 4.6 梯度补偿占位

后来补了：

```text
FTCCL_PATCH_GRAD_COMPENSATION=survivor_global_batch
FTCCL_PATCH_GRAD_COMPENSATION=old_world_size
FTCCL_PATCH_GRAD_COMPENSATION=zero_missing_average
FTCCL_PATCH_GRAD_COMPENSATION=custom
```

实现位置：

```text
RuntimeState.apply_gradient_compensation()
```

接入方式：

```text
train_step wrapper 临时包装 optimizer.step()
在真实 optimizer.step() 前遍历 model/optimizer 参数
对 param.main_grad / param.grad 做原地 mul_(scale)
写 grad_compensation.<rank>.<generation>.<iteration>
```

这是占位，不是最终生产策略。它的价值是把补偿位置和 marker 先打通，后续可以替换成 bucket 级策略。

## 5. 统一 control plane 是怎么形成的

早期容易写成多个触发方式：

```text
手动 touch trigger
supervisor 自己写 failed_rank/survivors
detector 自己写 failed_rank/survivors
C middleware 自己补 generation
```

这样很快会失控：字段不一致、generation 不一致、survivors 不一致。

最终统一成：

```text
ftccl_bypassctl.py request
```

发布：

```text
bypass_request.json
failed_rank
survivors
generation
trigger
killed.<rank>      # --mark-killed
propagate_error    # --propagate-error
```

调用方只负责说明“谁坏了”和“为什么”：

```bash
./ftccl_midware/scripts/ftccl_bypassctl.py \
  --sync-dir /tmp/ftccl_bypass_real \
  request \
  --rank 7 \
  --world-size 8 \
  --source supervisor \
  --reason sigkill-demo \
  --mark-killed \
  --propagate-error
```

C middleware 和 Python patch 都只读这套 canonical files。

## 6. 我是怎么联调 Megatron 的

### 6.1 第一阶段：先证明 C 层 survivor communicator 能工作

先不碰 Megatron，写最小 all-reduce smoke：

```text
ftccl_midware/tests/ftccl_allreduce_smoke.cu
```

验证三种情况：

```bash
./ftccl_midware/scripts/run_smoke.sh --np 8
./ftccl_midware/scripts/run_smoke.sh --np 8 --bypass --dead-rank 7
./ftccl_midware/scripts/run_smoke.sh --np 8 --bypass --dead-rank 3
```

判断标准：

```text
no bypass:
  1+2+3+4+5+6+7+8 = 36
  第二轮 44
  第三轮 52

bypass rank 7:
  1+2+3+4+5+6+7 = 28
  第二轮 35
  第三轮 42

bypass rank 3:
  1+2+3+5+6+7+8 = 32
  第二轮 39
  第三轮 46
```

这个阶段确认：

- LD_PRELOAD 生效。
- `ncclCommPrepareSurvivorTopo` / `ncclCommActivateSurvivorTopo` 符号可用。
- 任意 failed rank 的 survivor rank 映射正确。
- group replay 不破坏 NCCL 调用序。

### 6.2 第二阶段：让 Megatron 逻辑 bypass 跑完

Megatron 入口是：

```text
/home/ubuntu/sdr/Megatron-LM/examples/llama/train_llama3.2_1b.sh
```

脚本最前面 source：

```bash
source /home/ubuntu/sdr/ftccl/ftccl_midware/scripts/megatron_env.sh
```

`megatron_env.sh` 注入：

```bash
LD_LIBRARY_PATH=${FTCCL_HOME}/lib:...
LD_PRELOAD=${FTCCL_MIDWARE_DIR}/build/libftccl_midware.so
PYTHONPATH=${FTCCL_MIDWARE_DIR}/python:...
NCCL_ALGO=RING
NCCL_PROTO=SIMPLE
FTCCL_BYPASS_ENABLE=1
FTCCL_PATCH_ENABLE=1
FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS=126
```

逻辑 demo：

```bash
./ftccl_midware/scripts/run_megatron_fault_tolerance_demo.sh /tmp/ftccl_megatron_patch.log
```

关键日志：

```text
rank 7 entering logical victim mode=noop
rank 0 bypass complete: survivorRank=0 survivorCount=7
activated survivor training view generation=1 ... dp_size 8->7 global_batch_size 128->126
dataloader rebuilt generation=1 dp_rank=0 dp_size=7
[after training is done]
```

这个阶段确认：

- Megatron 不改核心源码也能加载 Python patch。
- C 层收缩和 Python 层 DP view 对齐。
- dataloader 按 7-way survivor shard 重建。
- global batch / microbatch 重新配置不会触发 Megatron assert。
- victim noop 可以让 torchrun/Gloo/Megatron 控制流不崩。

### 6.3 第三阶段：真实 SIGKILL

`torchrun` 默认看到一个 worker 死亡通常会杀全体 worker，所以真实 kill demo 不能用 torchrun 管 worker 生命周期。

因此写了：

```text
ftccl_midware/scripts/run_megatron_real_kill_demo.py
```

它做：

```text
for rank in 0..7:
  subprocess.Popen(["bash", train_script], preexec_fn=os.setsid)
  设置 WORLD_SIZE/RANK/LOCAL_RANK/MASTER_ADDR/MASTER_PORT

等待 pause_ready.<rank>.1
SIGKILL rank 7 的进程组
调用 ftccl_bypassctl.py request --mark-killed --propagate-error
写 continue_after_kill.1
等待 survivor 退出
检查日志和 marker
```

关键证据：

```text
FTCCL_REAL_KILL_DEMO_PASS killed_rank=7 victim_exit=-9
continue_after_kill received iteration=1
propagating recoverable ncclSystemError
caught recoverable communication error
rollback complete; retrying train_step
bypass complete: survivorRank=0 survivorCount=7
[after training is done]
```

marker：

```text
rollback_replay.0.1.1 ... rollback_replay.6.1.1
train_step_done.0.2 ... train_step_done.6.2
```

### 6.4 第四阶段：检测路径

检测路径先用 smoke 展示，避免和 real-kill supervisor 竞争。

```bash
./ftccl_midware/scripts/run_smoke.sh --np 8 --detect --sync-dir /tmp/ftccl_smoke_detect
```

检查：

```bash
ls /tmp/ftccl_smoke_detect/probe.*
head -80 /tmp/ftccl_smoke_detect/probe.*
```

确认有：

```text
schema=ftccl.detect.v1
ccld_available=1
ccld_send_count=...
ccld_recv_count=...
```

这个阶段确认：

- interposer 可以写 host preflight probe。
- C 层可以动态查到 `ncclCommCcldGetMetrics`。
- FTCCL kernel metrics path 和 normal collective 不冲突。

## 7. 踩过的坑和具体修改

### 坑 1：只改 Python 不够，ProcessGroupNCCL 内部 comm 已经固定

**现象**

Megatron 上层能看到 patch 后的 DP size，但真实 all-reduce 仍然卡在 8-rank communicator 或语义不一致。

**原因**

PyTorch C++ `ProcessGroupNCCL` 已经持有 `ncclComm_t`，Python patch 无法替换底层 communicator。

**修改**

引入 `LD_PRELOAD` C middleware：

```text
ftccl_midware/src/ftccl_midware.cu
```

拦截 NCCL C ABI，调用 FTCCL survivor topology API。

**验证**

`run_smoke.sh --bypass` 证明 all-reduce sum 从 36 变成 28。

### 坑 2：在 NCCL group 中间切 communicator 会破坏调用序

**现象**

PyTorch/Megatron 有时 collective 被包在 `ncclGroupStart/End` 中。直接在 `ncclAllReduce` 内切换 communicator 容易出现 hang 或 NCCL internal error。

**原因**

NCCL group 要求所有 group 内 op 按一致顺序提交。中途 prepare/activate 会打乱顺序。

**修改**

实现 pending op：

```text
ncclGroupStart -> 不调用真实 group start
collective -> 记录 PendingOp
ncclGroupEnd -> 先 bypass，再真实 group start + replay ops + real group end
```

**验证**

Megatron 训练中大量 group collective 能继续完成。

### 坑 3：太早 bypass 会干扰 PyTorch/Megatron 初始化

**现象**

一开始 `FTCCL_BYPASS_TRIGGER_AT_START=1` 时，如果第一个 full-size communicator 就 bypass，Megatron 初始化阶段容易卡住或后续 process group 状态不一致。

**原因**

第一个 full-size NCCL communicator 常常属于初始化/参数同步阶段，不一定是稳定的 DP training collective。

**修改**

加入：

```text
FTCCL_BYPASS_MIN_COMM_ORDINAL=2
```

C 层记录 `fullSizeOrdinal`，低于 min ordinal 的 communicator 不触发 bypass。

**验证**

Megatron demo 稳定看到：

```text
record comm ... fullSizeOrdinal=1
record comm ... fullSizeOrdinal=2
rank 7 entering logical victim mode=noop
```

### 坑 4：victim 真实退出会触发 torchrun 杀所有 worker

**现象**

如果在 torchrun 下让 rank 7 `exit0` 或 SIGKILL，torchrun/elastic 会终止整个 job。

**原因**

torchrun 默认管理 worker 生命周期，一个 worker 异常退出不是“局部故障”，而是 job failure。

**修改**

分两种 demo：

```text
逻辑 bypass:
  FTCCL_BYPASS_VICTIM_MODE=noop
  victim 进程活着但 NCCL collective no-op

真实 kill:
  不用 torchrun 管 worker
  run_megatron_real_kill_demo.py 用 subprocess/os.setsid 启动每个 rank
  supervisor 自己 SIGKILL victim
```

**验证**

逻辑 demo 和 real-kill demo 都能完成训练。

### 坑 5：Megatron dataloader 仍按 8-way shard

**现象**

C 层 all-reduce 已经变成 7 ranks，但 Megatron dataloader 仍按 8-way 切数据，训练语义不一致。

**原因**

dataloader 构建时读的是 Megatron `mpu.get_data_parallel_rank/world_size()` 和 `args.data_parallel_size`。

**修改**

Python patch：

```text
patch parallel_state.get_data_parallel_world_size/rank
patch build_pretraining_data_loader 捕获 train dataset
用 FaultTolerantDataIterator 在 generation 变化时重建 dataloader
```

**验证**

日志：

```text
dataloader rebuilt generation=1 dp_rank=0 dp_size=7 consumed_samples=...
```

### 坑 6：global batch 128 不能被 7 整除

**现象**

bypass 后如果 DP size 改成 7，但 global batch 仍是 128，Megatron microbatch calculator 会出现整除/一致性问题。

**原因**

Megatron 要求：

```text
global_batch_size % (micro_batch_size * data_parallel_size) == 0
```

`128 % (1 * 7) != 0`。

**修改**

默认：

```bash
FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS=126
```

并在 bypass active 时：

```python
args.data_parallel_size = 7
args.global_batch_size = 126
reconfigure_num_microbatches_calculator(...)
```

**验证**

日志：

```text
dp_size 8->7 global_batch_size 128->126
iteration 3/3 ... global batch size: 126
```

### 坑 7：真实 rank 死亡时 victim 不会写 killed marker

**现象**

survivor 等待 `killed.<rank>` 超时。

**原因**

逻辑 bypass 中 victim 还能写 marker；真实 SIGKILL 后 victim 进程不存在，不能写任何文件。

**修改**

统一控制接口支持：

```bash
ftccl_bypassctl.py request --mark-killed
```

由 supervisor 或 detector 写：

```text
killed.<rank>
```

**验证**

real-kill demo 中 supervisor 杀 rank 7 后发布：

```text
FTCCL_BYPASS_REQUEST ... failed_rank=7 ... mark_killed=True
```

### 坑 8：通信错误需要上抛到 Python，才能触发 rollback/retry

**现象**

如果 C 层直接完成 bypass，无法验证“in-flight collective 出错 -> 上层 rollback -> retry 当前 step”的链路。

**原因**

真实故障通常会在某次 collective 中以 ProcessGroupNCCL error 表现出来。只做静默 bypass 不足以验证训练状态恢复。

**修改**

加入：

```text
FTCCL_BYPASS_PROPAGATE_ERROR_BEFORE_BYPASS=1
propagate_error marker
```

C 层 survivor 下一次 collective 返回一次：

```c++
return ncclSystemError;
```

Python patch 捕获：

```text
torch.distributed.DistBackendError
```

然后做 rollback/retry。

**验证**

日志：

```text
propagating recoverable ncclSystemError
caught recoverable communication error
rollback complete; retrying train_step
```

### 坑 9：Megatron rerun state 会影响 retry

**现象**

通信错误后直接重新调用 `train_step`，Megatron rerun state 可能仍处于异常/重放状态，导致 retry 不干净。

**原因**

Megatron 自己有 `RerunStateMachine`，它保存 iteration、saved results、validation state 等。

**修改**

在 rollback 时显式 reset：

```python
machine.state = RerunState.NOT_RUNNING_YET
machine.current_iteration = iter_value
machine.rerun_requested = False
machine.checkpoint_requested = False
machine.restart_again_requested = False
machine.continue_requested = False
machine.data_iterator_checkpoints = None
machine.saved_results.clear()
machine.validation_counts.clear()
```

**验证**

日志：

```text
reset Megatron rerun state before fault replay iteration=1
rollback complete; retrying train_step
```

### 坑 10：real-kill demo 和 detector 会抢触发

**现象**

真实 kill demo 还没到 supervisor 计划 SIGKILL，host preflight detector 已经判了某个 rank timeout，提前发布 bypass request。结果 supervisor 等不到 `pause_ready` 或 failed rank 不等于计划 kill rank。

**原因**

`megatron_env.sh` 默认打开：

```bash
FTCCL_DETECT_ENABLE=1
```

真实 kill demo 本身又是一个显式 supervisor fault decision。两条故障决议路径并发，会互相竞争。

**修改**

在 `run_megatron_real_kill_demo.py` 的 `base_env` 中显式关闭 detector：

```python
"FTCCL_DETECT_ENABLE": "0",
"FTCCL_DETECT_CCLD_ENABLE": "0",
```

检测路径改由：

```bash
run_smoke.sh --detect
```

单独展示。

**验证**

修复后 real-kill demo 输出：

```text
rank 0 reached pause_iter=1
...
rank 7 reached pause_iter=1
SIGKILL rank 7
FTCCL_REAL_KILL_DEMO_PASS killed_rank=7 victim_exit=-9
```

### 坑 11：新目录布局后旧路径硬编码全部失效

**现象**

把旧 `/home/ubuntu/sdr/ftccl` 改成：

```text
ftccl_lib/
ftccl_midware/
```

后，脚本仍指向旧路径：

```text
/home/ubuntu/sdr/ftccl/build
/home/ubuntu/sdr/demo/ftccl-midware
```

导致链接到错误 libnccl、找不到 middleware、Megatron source 旧 env。

**修改**

1. `ftccl_midware/CMakeLists.txt`：

```cmake
get_filename_component(FTCCL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(FTCCl_NCCL_HOME "${FTCCL_REPO_ROOT}/ftccl_lib/build" CACHE PATH ...)
```

2. `run_smoke.sh`：

```bash
midware_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${midware_dir}/.." && pwd)"
ftccl_home="${FTCCL_HOME:-${repo_root}/ftccl_lib/build}"
```

3. `megatron_env.sh`：

```bash
FTCCL_MIDWARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FTCCL_REPO_ROOT="$(cd "${FTCCL_MIDWARE_DIR}/.." && pwd)"
FTCCL_HOME="${FTCCL_HOME:-${FTCCL_REPO_ROOT}/ftccl_lib/build}"
LD_PRELOAD="${FTCCL_MIDWARE_DIR}/build/libftccl_midware.so..."
PYTHONPATH="${FTCCL_MIDWARE_DIR}/python:..."
```

4. `run_megatron_real_kill_demo.py`：

```python
ctl_script = Path(__file__).resolve().parent / "ftccl_bypassctl.py"
```

5. Megatron 训练脚本：

```bash
source /home/ubuntu/sdr/ftccl/ftccl_midware/scripts/megatron_env.sh
```

**验证**

```bash
ldd ftccl_midware/build/libftccl_midware.so | rg 'nccl|cuda'
```

输出指向：

```text
/home/ubuntu/sdr/ftccl/ftccl_lib/build/lib/libnccl.so.2
```

### 坑 12：CMake cache 是旧目录生成的

**现象**

新目录下跑：

```bash
cmake -S ftccl_midware -B ftccl_midware/build
```

报：

```text
The current CMakeCache.txt directory ... is different than ...
source ... does not match the source ... used to generate cache
```

**原因**

`build/` 是从旧目录搬过来的，CMakeCache 记录了旧 source path。

**修改**

使用：

```bash
cmake --fresh -S ftccl_midware -B ftccl_midware/build \
  -DFTCCl_NCCL_HOME=/home/ubuntu/sdr/ftccl/ftccl_lib/build
```

**验证**

`cmake --build ftccl_midware/build -j` 通过。

### 坑 13：运行时加载了错误 NCCL，survivor symbol undefined

**现象**

smoke bypass 报：

```text
undefined symbol: ncclCommPrepareSurvivorTopo
```

**原因**

`LD_LIBRARY_PATH` 指向旧或错误 NCCL 构建，那个 libnccl 没有 FTCCL survivor API。

**修改**

统一让 `LD_LIBRARY_PATH` 指向：

```text
/home/ubuntu/sdr/ftccl/ftccl_lib/build/lib
```

并用 `ldd` 验证。

**验证**

```bash
nm -D ftccl_lib/build/lib/libnccl.so.2 | rg 'ncclComm(Prepare|Activate)SurvivorTopo'
ldd ftccl_midware/build/libftccl_midware.so | rg nccl
```

### 坑 14：并行跑两个 smoke 会污染同一个 sync dir

**现象**

同时跑 no-bypass 和 bypass smoke，两个 MPI job 默认都用 `/tmp/ftccl_bypass`，出现 bootstrap/check-in 冲突或 hang。

**原因**

NCCL unique id 和 bypass control files 都写同一个目录。

**修改**

验证时每次使用不同 sync dir：

```bash
--sync-dir /tmp/ftccl_smoke_nobypass
--sync-dir /tmp/ftccl_smoke_bypass7
--sync-dir /tmp/ftccl_smoke_bypass3
```

README 也把 demo 命令写成显式 sync dir。

### 坑 15：smoke 日志一开始误导 survivor_count

**现象**

no-bypass 情况 all-reduce 结果是 36/44/52，但日志打印 `survivor_count=7`。

**原因**

测试程序里 `survivorCount = nranks - 1` 是硬编码，没判断 bypass 是否真的 active。

**修改**

在 `ftccl_allreduce_smoke.cu` 中检查：

```c++
bool bypassActive = exists((bypassDir + "/activate." + std::to_string(rank)).c_str());
printf(..., bypassActive ? survivorCount : nranks);
```

**验证**

no-bypass 输出：

```text
observed=36 survivor_count=8
```

### 坑 16：tracked __pycache__ 被 py_compile 刷新

**现象**

跑 Python 编译检查后，`git status` 出现一堆 `__pycache__/*.pyc` 修改。

**原因**

仓库里已有 tracked pyc，`python3 -m py_compile` 更新了它们。

**处理**

最终用 `git show HEAD:<path>` 恢复这些 pyc，只保留真实代码和文档修改。

长期建议：把 pyc 从仓库移除并加入 `.gitignore`。

## 8. 每轮验证命令

### 8.1 符号和链接

```bash
cd /home/ubuntu/sdr/ftccl

nm -D ftccl_lib/build/lib/libnccl.so.2 \
  | rg 'ncclComm(Prepare|Activate)SurvivorTopo|ncclCommCcldGetMetrics'

cmake --fresh -S ftccl_midware -B ftccl_midware/build \
  -DFTCCl_NCCL_HOME=/home/ubuntu/sdr/ftccl/ftccl_lib/build

cmake --build ftccl_midware/build -j

ldd ftccl_midware/build/libftccl_midware.so | rg 'nccl|cuda'
```

### 8.2 Smoke

```bash
./ftccl_midware/scripts/run_smoke.sh \
  --np 8 \
  --sync-dir /tmp/ftccl_smoke_nobypass

./ftccl_midware/scripts/run_smoke.sh \
  --np 8 \
  --bypass \
  --dead-rank 7 \
  --sync-dir /tmp/ftccl_smoke_bypass7

./ftccl_midware/scripts/run_smoke.sh \
  --np 8 \
  --bypass \
  --dead-rank 3 \
  --sync-dir /tmp/ftccl_smoke_bypass3

./ftccl_midware/scripts/run_smoke.sh \
  --np 8 \
  --detect \
  --sync-dir /tmp/ftccl_smoke_detect
```

### 8.3 Megatron 逻辑 bypass

```bash
./ftccl_midware/scripts/run_megatron_fault_tolerance_demo.sh \
  /tmp/ftccl_megatron_patch.log

rg 'logical victim|bypass complete|activated survivor|dataloader rebuilt|after training is done' \
  /tmp/ftccl_megatron_patch.log
```

### 8.4 Megatron real-kill

```bash
python3 ./ftccl_midware/scripts/run_megatron_real_kill_demo.py \
  --log-dir /tmp/ftccl_real_kill_logs \
  --sync-dir /tmp/ftccl_bypass_real

rg 'continue_after_kill|propagating recoverable|caught recoverable|rollback complete|bypass complete|after training is done' \
  /tmp/ftccl_real_kill_logs/combined.log
```

## 9. 当前改动清单

### 9.1 ftccl_midware/CMakeLists.txt

做了：

- 从旧硬编码 `/home/ubuntu/sdr/ftccl/build` 改为基于 source dir 推导 repo root。
- 默认 `FTCCl_NCCL_HOME=${repo_root}/ftccl_lib/build`。
- 保持 `FTCCl_NCCL_HOME` 可由命令行覆盖。

价值：

- 新布局下直接 configure。
- 避免目录搬动后继续链接旧 NCCL。

### 9.2 ftccl_midware/scripts/run_smoke.sh

做了：

- 从脚本位置推导 `midware_dir`、`repo_root`、`ftccl_home`。
- 默认使用 `ftccl_lib/build`。
- 保留 `--build-dir` 和 `FTCCL_HOME` 覆盖能力。
- 支持 `--detect`。

价值：

- smoke 不依赖旧绝对路径。
- no-bypass/bypass/detect 三类测试都从同一入口跑。

### 9.3 ftccl_midware/scripts/megatron_env.sh

做了：

- 从脚本位置推导 `FTCCL_MIDWARE_DIR` 和 `FTCCL_REPO_ROOT`。
- `LD_LIBRARY_PATH` 指向 `ftccl_lib/build/lib`。
- `LD_PRELOAD` 指向当前 `ftccl_midware/build/libftccl_midware.so`。
- `PYTHONPATH` 指向当前 `ftccl_midware/python`。
- 统一设置 NCCL/FTCCL/Python patch 环境变量。

价值：

- Megatron 入口只需要 source 这个文件。
- 迁移目录时不再大面积改路径。

### 9.4 ftccl_midware/scripts/run_megatron_real_kill_demo.py

做了：

- `ctl_script` 改为同目录解析。
- real-kill demo 环境中关闭 detector：

```python
"FTCCL_DETECT_ENABLE": "0"
"FTCCL_DETECT_CCLD_ENABLE": "0"
```

- 保留 `--propagate-error` 默认路径，用来验证 rollback/retry。

价值：

- supervisor fault decision 不再和 host preflight detector 竞争。
- 真实 SIGKILL demo 稳定通过。

### 9.5 /home/ubuntu/sdr/Megatron-LM/examples/llama/train_llama3.2_1b.sh

做了：

```bash
source /home/ubuntu/sdr/ftccl/ftccl_midware/scripts/megatron_env.sh
```

价值：

- Megatron 训练脚本加载当前新布局 middleware。

### 9.6 根目录 README.md

做了：

- 写明新布局。
- 写明当前功能状态。
- 写明技术路线。
- 写明 demo 入口。
- 写明已验证结果和限制。

## 10. 当前能力边界

已经做到：

```text
纯 DP 8 -> 7 survivor communicator
任意单 DP rank bypass
Megatron 上层 survivor DP view
7-way dataloader rebuild
global batch/microbatch 重配
真实 SIGKILL rank 7 后 survivor 继续训练
communication error -> rollback -> retry
统一 bypass request
host preflight probe
FTCCL ProtoSimple CCLD metrics probe
梯度补偿 step 前占位
```

还没做到：

```text
TP/PP/CP 混合并行 communicator 精准识别和收缩策略
ZeRO / distributed optimizer
任意 CUDA kernel 中断后的精确恢复
LL/LL128 CCLD metrics 覆盖
独立常驻 detector/control-plane 服务
bucket 级梯度补偿
failed root 的 Broadcast/Reduce 完整语义
```

## 11. 后续建议

1. **把 detector 从 interposer 内联逻辑拆成独立服务**  
   当前 detector 和 supervisor 已经证明需要避免竞争。生产化应该有一个 single writer 或 quorum control plane。

2. **只对 DP communicator bypass**  
   当前主要靠 pure-DP 场景和 full-size ordinal。扩展 TP/PP/CP 前必须明确 communicator 类型。

3. **把 CCLD metrics 覆盖到更多协议**  
   目前重点是 `NCCL_PROTO=SIMPLE`。LL/LL128 需要继续埋点。

4. **把 rollback 从 iteration 级推进到更细粒度**  
   当前 demo 假设 optimizer step 尚未提交。更复杂故障需要明确 CUDA work、optimizer state、dataloader state 的一致边界。

5. **把梯度补偿从 step 前占位变成 bucket 级策略**  
   目前 `old_world_size`/`zero_missing_average` 是验证补偿位置和日志 marker。生产策略要更靠近 DDP bucket 或 optimizer。

6. **清理仓库里的 pyc 和旧日志**  
   当前仓库里仍有 tracked `__pycache__` 和历史日志。建议后续单独清理并加入 `.gitignore`。

## 12. 结论

这次实现的关键不是某一个 patch，而是把三件事对齐：

```text
底层事实:
  FTCCL communicator 真的从 8 ranks 变成 7 ranks。

上层语义:
  Megatron 在 bypass 后按 7-rank DP view 继续取数据、算 microbatch、更新训练进度。

控制闭环:
  主动 bypass、detector、真实 SIGKILL 都发布同一种 failed_rank/survivors/generation/trigger 状态。
```

最终 demo 已经证明：

```text
rank 可以被动态隔离；
survivor communicator 可以收缩；
Megatron 可以不改核心源码继续训练；
真实 rank 被 SIGKILL 后 survivor 可以 rollback/retry 并跑到 training done。
```

