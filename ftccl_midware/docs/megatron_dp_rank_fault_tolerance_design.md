# Megatron DP Rank 下线后继续训练容错功能指引

本文整理如何在 Megatron-LM 纯 DP 训练中实现“下线一个 DP rank 后继续训练”的容错能力。目标不是只做一个短 demo，而是给出从当前 FTCCl/FCCL bypass demo 演进到可维护中间件的工程路线。

本文默认场景：

- 单机或多机纯数据并行优先，`TP=1`、`PP=1`、`CP=1`。
- 不开启 ZeRO，不开启 Megatron distributed optimizer。
- 一次只隔离一个 DP rank，后续再扩展到多 rank 或整节点隔离。
- 下线 rank 后，survivor DP ranks 继续训练。
- Megatron 源码文件尽量不改，通过 `LD_PRELOAD` 与 Python runtime patch 实现。

当前 demo 已实现到“逻辑下线一个 DP rank + Python 上层 survivor 视角修正”的阶段：

- C 层通过 FTCCl/FCCL 将 NCCL communicator 从 8-rank 收缩到 7-rank survivor 拓扑。
- Python patch 层在 bypass 后将 Megatron 的有效 DP size 从 8 改为 7。
- Python patch 层将 `global_batch_size` 从 128 重配为 126，以满足 `micro_batch_size * survivor_dp_size` 整除。
- Python patch 层在训练取 batch 时重建 survivor dataloader，survivor ranks 使用 7-way sampler。
- victim rank 当前仍采用 `FTCCL_BYPASS_VICTIM_MODE=noop`，这是逻辑下线 demo，不是真实进程死亡。
- 验证脚本：`scripts/run_megatron_fault_tolerance_demo.sh`。

## 结论

完整方案需要两层中间件：

```text
FTCCl/FCCL C middleware, LD_PRELOAD
  负责真实 NCCL/FCCL communicator 动态收缩
  负责 ncclCommPrepareSurvivorTopo / ncclCommActivateSurvivorTopo
  负责 NCCL group 延迟、collective replay、victim NCCL noop/隔离

Python patch middleware
  负责 Megatron 上层语义修正
  负责 DP rank/world size 的 survivor 视角
  负责 dataloader/sampler 重建
  负责 args.data_parallel_size、global batch、consumed_samples、日志与 checkpoint 元数据修正
```

不能把所有功能都放到 Python patch 中。Python patch 可以拦截 Megatron 和 `torch.distributed` Python API，但无法可靠改写 PyTorch C++ `ProcessGroupNCCL` 内部已缓存的 NCCL communicator，也无法替代 FCCL 的 survivor topology 切换。因此，FTCCl/FCCL bypass 仍然是底层通信容错的核心。

## 当前 demo 能力基线

当前 `/home/ubuntu/sdr/demo/ftccl-midware` 已经实现：

- `LD_PRELOAD` 加载 `libftccl_midware.so`。
- 拦截常用 NCCL C ABI：
  - `ncclCommInitRank`
  - `ncclCommInitRankConfig`
  - `ncclGroupStart`
  - `ncclGroupEnd`
  - `ncclAllReduce`
  - `ncclReduceScatter`
  - `ncclAllGather`
  - `ncclBroadcast`
  - `ncclReduce`
- 使用 FCCL survivor API：
  - `ncclCommPrepareSurvivorTopo(comm, survivorCount, survivorRanks)`
  - `ncclCommActivateSurvivorTopo(comm, survivorCount, survivorRanks)`
- 通过 `FTCCL_BYPASS_VICTIM_MODE=noop` 让 victim rank 不参与 NCCL，但 Python 进程继续活着。
- 已验证 Megatron 2 step 训练可完成，日志在 `/tmp/ftccl_megatron_final.log`。

当前 demo 还不是完整真实掉线恢复：

- dataloader 没有重建，仍按 8 DP ranks 切数据。
- rank 7 没有真实退出，而是逻辑 victim/noop。
- 梯度补偿没有实现。
- `torchrun`/Gloo/default process group 没有处理真实 worker 退出。

## 目标语义

希望最终达到的训练语义是：

```text
故障前：
  DP world size = 8
  DP ranks = 0,1,2,3,4,5,6,7
  all-reduce = 8 ranks
  dataloader = 8-way shard

rank 7 下线后：
  survivor DP world size = 7
  survivor ranks = 0,1,2,3,4,5,6
  all-reduce = 7 ranks
  dataloader = 7-way shard
  optimizer 更新基于 7 个 survivor batch
  consumed samples / global batch / LR schedule 有一致定义
```

对于 Megatron 上层，有两种可选语义。

### 语义 A：逻辑 world size 保持 8

这是当前 demo 的语义。

```text
Megatron 仍认为 world size = 8
dataloader 仍按 8-way shard
victim rank 进程还活着但 NCCL noop
survivor ranks 在底层 7-rank communicator 上通信
```

优点：

- 最容易验证通信 bypass。
- 对 Megatron 上层侵入最小。
- 可以完全由 C middleware 先完成。

缺点：

- rank 7 的 batch 逻辑上被消费但不贡献有效梯度。
- 数据、loss、global batch、学习率进度都不是严格 7-rank 语义。
- 无法代表真实 rank 退出后的控制面恢复。

### 语义 B：Megatron 上层切到 7-rank survivor 视角

这是后续完整容错应该追求的语义。

```text
通信层：FCCL communicator 变成 7 ranks
Megatron 层：DP world size / DP rank / dataloader 也变成 7-rank survivor 视角
victim rank：停止取数据，进入 park/noop，或真实退出后由 supervisor 隔离
```

优点：

- 数据切分与通信域一致。
- 更接近真实容错语义。
- 后续可以做正确的 consumed samples、checkpoint、恢复训练。

缺点：

- 必须增加 Python patch middleware。
- 要处理 global batch 可整除性、统计和 scheduler。
- 真实 worker 退出时还要处理 `torchrun`、Gloo、store、rendezvous。

## 总体架构

推荐架构如下：

```text
train_llama3.2_1b.sh
  |
  |-- source scripts/megatron_env.sh
  |     export LD_PRELOAD=.../libftccl_midware.so
  |     export PYTHONPATH=.../python:${PYTHONPATH}
  |     export FTCCL_BYPASS_*
  |
  |-- torchrun pretrain_gpt.py
        |
        |-- Python 启动时自动 import sitecustomize.py
        |     安装 ftccl_megatron_adaptor 的 monkey patches
        |
        |-- Megatron 正常初始化
        |
        |-- C middleware 拦截 NCCL communicator 和 collective
        |
        |-- 触发 bypass
              |
              |-- C 层收缩 NCCL/FCCL communicator
              |-- Python 层在 iteration 边界重建 dataloader/sampler
```

两层之间通过文件状态或共享控制面同步。当前 demo 使用目录：

```bash
FTCCL_BYPASS_DIR=/tmp/ftccl_bypass
```

建议继续使用该目录作为第一版控制面，后续再换成更严格的 supervisor RPC 或 store。

## FTCCl/FCCL C Middleware 设计

### 职责

C middleware 只负责通信层事实，不负责 Megatron 训练语义：

- 跟踪 `ncclComm_t` 与 parent rank/nranks。
- 判断 communicator 是否是 full-size DP communicator 候选。
- 在 trigger 出现后构建 survivor rank list。
- 对 survivor ranks 调用 FCCL survivor API。
- 将 parent root 映射到 survivor root。
- 对 NCCL group 做延迟和 replay。
- 对 victim rank 做隔离策略。
- 将 bypass 状态发布给 Python patch middleware。

### 必要状态

每个 communicator 至少需要记录：

```c
struct CommState {
  int parentRank;
  int parentNranks;
  int fullSizeOrdinal;
  int deadRank;
  int survivorRank;
  int survivorCount;
  bool bypassed;
  bool victimParked;
};
```

建议后续增加：

```c
int bypassGeneration;
bool isDataParallelCandidate;
bool prepared;
bool activated;
uint64_t firstCollectiveSeq;
```

这样可以避免重复 bypass 同一个 generation，也方便 Python 层读取状态。

### 触发源

第一阶段可以保留当前方式：

```bash
FTCCL_BYPASS_TRIGGER_AT_START=1
FTCCL_BYPASS_MIN_COMM_ORDINAL=2
```

更完整的容错应支持：

```text
手动触发：
  touch ${FTCCL_BYPASS_DIR}/trigger

外部 detector 触发：
  echo 7 > ${FTCCL_BYPASS_DIR}/failed_rank
  touch ${FTCCL_BYPASS_DIR}/trigger

真实掉线 detector：
  heartbeat timeout / supervisor 发现 rank 失联
  survivor ranks 读到统一 failed rank 决议

中间件内置 detector：
  collective 提交前写入 rank probe frame
  用 communicator ordinal + operation counter 作为 Trace ID
  比较 op/count/datatype/redop/root/algo/proto Operation Type Set
  未进入或签名不一致时发布同一套 bypass request
  启用 NCCL_CCLD_ENABLE=1 时读取 FTCCL ProtoSimple primitive 指标
  多数 rank primitive 指标有进展而单个 rank 无进展时发布 kernel-ccld-progress-stall
```

真实 rank 掉线时不能依赖 victim 写 `killed.<rank>`。需要由外部 supervisor 或 survivor quorum 写入故障决议。

当前内置 detector 默认会在判定 failed rank 后写入 `killed.<rank>`，这是为了覆盖 victim 已死亡或已无法进入 NCCL interposer 的被动故障场景。若只想做主动 drain，可关闭 `FTCCL_DETECT_MARK_KILLED`，让 victim 在下一次 collective 中自己进入 `noop/park/exit0`。

### Victim 策略

建议保留三种模式：

```bash
FTCCL_BYPASS_VICTIM_MODE=noop
FTCCL_BYPASS_VICTIM_MODE=park
FTCCL_BYPASS_VICTIM_MODE=exit0
```

含义：

- `noop`：victim Python 进程继续跑，但所有被拦截 NCCL 调用直接返回成功。适合 Megatron demo。
- `park`：victim 写状态后挂起，不再进入训练。适合观察 survivor 是否能独立前进。
- `exit0`：victim 退出。适合 smoke test，不适合默认 `torchrun`。

真实故障场景中 victim 已经不存在，C middleware 只在 survivor ranks 上运行。此时需要 supervisor 保证 survivor 不被 launcher 杀掉。

### Group 延迟

必须保留 `ncclGroupStart/End` 延迟机制：

```text
拦截 ncclGroupStart
  tls.depth++
  pending_ops.clear()

拦截 group 内 collective
  记录 pending op
  不立即调用真实 NCCL

拦截 ncclGroupEnd
  若需要 bypass，先 prepare/activate
  调真实 ncclGroupStart
  replay pending ops
  调真实 ncclGroupEnd
```

否则可能在 group 中间切 communicator，破坏 NCCL 调用序。

### 通信结果补偿

当前 demo 还没有补偿。后续至少要明确以下策略之一：

```text
策略 1：survivor mean
  梯度按 survivor_count 平均
  等价于 global batch 缩小到 survivor_count * micro_batch * num_microbatches

策略 2：old world size 语义
  all-reduce 后乘以 old_world_size / survivor_count 或反向缩放
  试图保持原 global batch 梯度尺度

策略 3：Python 层调整 global batch
  DP size 变成 7 后，将 global batch 改成可被 7 整除的值
  LR schedule 与 consumed_samples 按新 batch 语义前进
```

推荐先做策略 3，语义最清楚。策略 2 可以作为兼容旧超参的实验模式。

## Python Patch Middleware 设计

### 为什么需要 Python patch

dataloader、sampler、Megatron 参数、训练进度都在 Python 层。`LD_PRELOAD` 无法直接重建这些对象。

Python patch 的目标是：

- 不修改 Megatron 源码文件。
- 在 Python 启动/import 阶段注入 patch。
- 让 Megatron 在 bypass 后看到 survivor DP 视角。

可参考 `/home/ubuntu/sdr/download/dcu_megatron` 的做法。它通过 `importlib`、`setattr`、`sys.modules` 在运行时替换 Megatron 函数和类，本质是 Python runtime patch。

### 推荐目录

建议在 `ftccl-midware` 中增加：

```text
python/
  sitecustomize.py
  ftccl_megatron_patch/
    __init__.py
    state.py
    patch_utils.py
    parallel_state_patch.py
    dataloader_patch.py
    training_patch.py
    args_patch.py
```

启动脚本加入：

```bash
export PYTHONPATH="/home/ubuntu/sdr/demo/ftccl-midware/python:${PYTHONPATH:-}"
```

`sitecustomize.py` 内容只做一件事：

```python
import ftccl_megatron_patch
ftccl_megatron_patch.install()
```

### Patch 状态源

Python 层读取 C middleware 发布的状态：

```text
${FTCCL_BYPASS_DIR}/trigger
${FTCCL_BYPASS_DIR}/failed_rank
${FTCCL_BYPASS_DIR}/activate.<rank>
${FTCCL_BYPASS_DIR}/generation
${FTCCL_BYPASS_DIR}/survivors
```

建议格式：

```text
failed_rank:
  7

survivors:
  0,1,2,3,4,5,6

generation:
  1
```

Python helper：

```python
def bypass_active() -> bool:
    return Path(bypass_dir, "survivors").exists()

def failed_rank() -> int:
    return int(Path(bypass_dir, "failed_rank").read_text())

def survivor_ranks() -> list[int]:
    return [int(x) for x in Path(bypass_dir, "survivors").read_text().split(",")]

def survivor_rank(global_rank: int) -> int | None:
    ranks = survivor_ranks()
    return ranks.index(global_rank) if global_rank in ranks else None
```

### parallel_state patch

Megatron dataloader 构造依赖：

```python
mpu.get_data_parallel_rank()
mpu.get_data_parallel_world_size()
```

因此需要 patch：

```python
megatron.core.parallel_state.get_data_parallel_world_size
megatron.core.parallel_state.get_data_parallel_rank
```

基本逻辑：

```python
def patched_get_data_parallel_world_size(*args, **kwargs):
    if bypass_active() and is_pure_dp_mode():
        return len(survivor_ranks())
    return original_get_data_parallel_world_size(*args, **kwargs)

def patched_get_data_parallel_rank(*args, **kwargs):
    if bypass_active() and is_pure_dp_mode():
        mapped = survivor_rank(torch.distributed.get_rank())
        if mapped is None:
            return 0  # victim rank 只用于避免上层断言，后续不取真实 batch
        return mapped
    return original_get_data_parallel_rank(*args, **kwargs)
```

注意：这只能修正 Python 函数返回值，不等价于重建 PyTorch process group。NCCL 真实通信域仍由 C middleware 负责。

### args patch

Megatron 很多地方使用 `args.data_parallel_size`，不是每次都调用 `parallel_state`。

因此需要在合适位置 patch：

```python
megatron.training.arguments.validate_args
megatron.training.global_vars.get_args 或 training loop iteration 边界
```

bypass 后设置：

```python
args.data_parallel_size = survivor_count
```

同时处理：

```python
args.global_batch_size
args.train_samples
args.consumed_train_samples
args.consumed_valid_samples
```

关键约束：

```text
global_batch_size % (micro_batch_size * survivor_dp_size) == 0
```

当前脚本中 `GLOBAL_BATCH_SIZE=128`，如果 survivor DP size 是 7 且 micro batch 是 1，则 128 不能被 7 整除。需要选择：

```text
global_batch_size = 126
global_batch_size = 112
global_batch_size = 140
```

或者实现动态 microbatch 策略。第一版建议直接在脚本中为容错 demo 设置 `GLOBAL_BATCH_SIZE=126`。

### dataloader patch

Megatron 的 pretraining dataloader 构造入口是：

```python
megatron.training.datasets.data_samplers.build_pretraining_data_loader
```

它会读取：

```python
mpu.get_data_parallel_rank()
mpu.get_data_parallel_world_size()
```

patch 策略：

1. 初始化阶段不改，按原 8-rank 创建。
2. bypass active 后，在下一个 iteration 边界设置 `need_rebuild_dataloader=True`。
3. survivor ranks 调用原 `build_pretraining_data_loader(dataset, consumed_samples)`，此时 `mpu.get_data_parallel_world_size()` 已返回 7。
4. victim rank 返回一个 dummy iterator，不再取真实 batch。

dummy iterator：

```python
class VictimDataIterator:
    def __iter__(self):
        return self

    def __next__(self):
        return None
```

训练 loop patch 需要让 victim 不进入真实 forward/backward，或者进入空转路径。

### training loop patch

最小可行 patch 点：

```python
megatron.training.training.train
megatron.training.training.train_step
```

推荐在 iteration 边界执行：

```text
before train_step:
  检查 bypass generation
  如果 generation 变化：
    survivor ranks rebuild dataloader
    update args.data_parallel_size
    update num_microbatches
    标记 victim rank inactive

train_step:
  survivor ranks 正常 forward/backward/optimizer
  victim rank 跳过真实 batch 和 NCCL
```

不要在以下位置切换：

- forward 中间
- backward 中间
- NCCL collective 已提交后
- optimizer step 中间

### checkpoint 和日志 patch

需要统一以下字段：

```text
effective_data_parallel_size
survivor_ranks
failed_ranks
fault_generation
global_batch_size_after_fault
consumed_samples_after_fault
```

第一版可以只记录到日志，不写 checkpoint。可训练更久时再写入 checkpoint metadata。

## 真实 DP Rank 掉线支持

当前 demo 使用 victim `noop`，不是真实掉线。要支持真实 rank 退出，需要额外控制面。

### 必须新增 supervisor

`torchrun` 默认看到一个 worker 退出后，可能终止整个 job。真实掉线容错需要：

- 自定义 launcher/supervisor，或者修改启动策略。
- survivor 进程不能因为 victim 退出被一起杀掉。
- supervisor 负责检测 rank 心跳并发布故障决议。

最小 supervisor 语义：

```text
每个 rank 定期写 heartbeat.<rank>
supervisor 检测 heartbeat 超时
supervisor 写 failed_rank 和 survivors
supervisor 写 trigger
survivor ranks 看到 trigger 后执行 FCCL bypass
```

### 真实掉线与逻辑隔离的差异

逻辑隔离：

```text
victim rank 还活着
victim 可写 killed marker
torchrun 不感知 worker failure
Python/Gloo 控制流仍是 8 进程
适合 demo
```

真实掉线：

```text
victim rank 不存在
victim 不会写 marker
torchrun 可能杀掉其他 worker
Gloo/default process group 可能失败
需要 supervisor 和 Python 控制面
```

第一版真实掉线建议使用“受控 SIGKILL 测试”：

```text
所有 rank 初始化完成
supervisor 在 iteration 边界 SIGKILL rank 7
survivor ranks 继续
supervisor 不杀 survivor
```

这比随机 GPU reset 或节点断电更容易定位问题。

## 实施阶段

### 阶段 0：当前 demo 固化

目标：

- 保留当前 `LD_PRELOAD` bypass。
- 保证 smoke test 和 Megatron 2 step 继续通过。
- 文档化当前限制。

验证：

```bash
cd /home/ubuntu/sdr/demo/ftccl-midware
cmake -S . -B build -DFTCCl_NCCL_HOME=/home/ubuntu/sdr/ftccl/build
cmake --build build -j
./scripts/run_smoke.sh --np 8
./scripts/run_smoke.sh --np 8 --bypass --dead-rank 7
```

Megatron：

```bash
cd /home/ubuntu/sdr/Megatron-LM
TRAIN_ITERS=2 EVAL_ITERS=0 FTCCL_LOG_LEVEL=1 \
FTCCL_BYPASS_MIN_COMM_ORDINAL=2 FTCCL_BYPASS_VICTIM_MODE=noop \
timeout 240s bash examples/llama/train_llama3.2_1b.sh \
> /tmp/ftccl_megatron_final.log 2>&1
```

### 阶段 1：Python patch 观测层

目标：

- 增加 `python/sitecustomize.py`。
- 打印 Megatron rank、DP rank、DP world size、bypass 状态。
- 不改变训练行为。

验证：

```text
日志中看到：
  [ftccl-python] installed
  [ftccl-python] global_rank=...
  [ftccl-python] dp_rank=... dp_world_size=...
```

### 阶段 2：Python patch survivor DP 视角

目标：

- bypass 后 `get_data_parallel_world_size()` 返回 7。
- survivor ranks 的 `get_data_parallel_rank()` 返回 0-6。
- victim rank 标记 inactive。
- 不立即重建 dataloader，只验证 API 返回。

验证：

```text
rank 0-6:
  effective_dp_world_size=7
  effective_dp_rank=0..6

rank 7:
  inactive=True
```

### 阶段 3：重建 dataloader

目标：

- 在 iteration 边界重建 train data iterator。
- survivor ranks 使用 7-way sampler。
- victim rank 不再取真实 batch。
- 调整 `args.data_parallel_size=7`。
- 使用可被 7 整除的 global batch，例如 126。

验证：

```text
日志中看到：
  dataloader rebuilt generation=1 dp_size=7 dp_rank=...
  consumed_samples_before=...
  consumed_samples_after=...
```

训练至少完成 bypass 后 2 个 iteration。

### 阶段 4：梯度和样本语义修正

目标：

- 明确选择 survivor global batch 语义或 old global batch 补偿语义。
- 实现梯度缩放占位。
- 修正日志中的 throughput、consumed samples、learning rate schedule。

建议第一版：

```text
使用 survivor global batch
global_batch_size 从 128 改为 126
LR schedule 按实际 consumed samples 前进
```

### 阶段 5：真实 rank 掉线

目标：

- 增加 supervisor/heartbeat。
- 受控 SIGKILL 一个 rank。
- 不让 launcher 杀掉 survivor。
- survivor 通过 FCCL bypass 继续。
- Python patch 重建 dataloader。

验证：

```text
rank 7 真实退出
rank 0-6 完成 bypass
rank 0-6 完成后续至少 2 个 train iteration
无 EOFError / ProcessGroupNCCL fatal abort / torchrun kill survivor
```

## 测试矩阵

### C middleware

```text
8 ranks, no bypass, allreduce
8 ranks, bypass rank 7, allreduce
8 ranks, bypass rank 0, allreduce
8 ranks, bypass rank k, reducescatter
8 ranks, bypass rank k, allgather
Broadcast/Reduce with surviving root
Broadcast/Reduce with failed root: should reject or remap policy must be explicit
```

### Python patch

```text
no bypass:
  DP size remains 8
  dataloader shard remains 8-way

bypass active:
  survivor DP size becomes 7
  survivor rank mapping correct
  victim inactive
  dataloader rebuilt once per generation
```

### Megatron end-to-end

```text
logical noop victim:
  train 2 iters
  train 10 iters
  eval disabled

logical park victim:
  survivor train continues if control flow allows

real SIGKILL:
  controlled kill at iteration boundary
  survivor train 2 more iters
```

## 关键风险

1. `torchrun` 默认故障策略可能杀掉 survivor。
2. `all_gather_object`、barrier、checkpoint 等可能走 Gloo/default process group，真实 victim 退出后会失败。
3. Megatron 内部很多对象在初始化时缓存了 DP group size，patch 返回值不一定覆盖所有路径。
4. `global_batch_size` 必须与 survivor DP size 可整除。
5. 如果 rank 在 NCCL collective 中间死亡，survivor 可能已经阻塞，恢复点比 iteration 边界复杂。
6. 多 rank 或整节点掉线还没有当前验证基础，需要单独扩展 survivor list 测试。
7. TP/PP/CP 不是简单 DP replica 丢失，不能直接套用本方案。

## 推荐下一步

推荐按以下顺序推进：

1. 保留现有 `LD_PRELOAD` C middleware，不要用 Python patch 替代。
2. 增加 `python/sitecustomize.py` 和最小 patch 框架，只做观测。
3. patch `parallel_state`，实现 survivor DP rank/world size。
4. patch dataloader 构造，在 bypass 后重建 7-rank sampler。
5. 将 Megatron demo 的 `GLOBAL_BATCH_SIZE` 改为 126 或做环境变量覆盖。
6. 增加一条 Megatron 端到端测试：bypass 后继续 2 个 iteration，并检查 dataloader rebuilt 日志。
7. 再进入真实 rank SIGKILL + supervisor。

## 参考当前文件

- FTCCl/FCCL middleware:
  - `/home/ubuntu/sdr/demo/ftccl-midware/src/ftccl_midware.cu`
- Megatron middleware env:
  - `/home/ubuntu/sdr/demo/ftccl-midware/scripts/megatron_env.sh`
- Megatron training script:
  - `/home/ubuntu/sdr/Megatron-LM/examples/llama/train_llama3.2_1b.sh`
- FCCL survivor experiment:
  - `/home/ubuntu/sdr/ftccl/exp/survivor_collective_ring/src/survivor_collective_after_kill.cu`
  - `/home/ubuntu/sdr/ftccl/exp/survivor_collective_ring/scripts/run_collective_matrix.sh`
- Python runtime patch 参考：
  - `/home/ubuntu/sdr/download/dcu_megatron/adaptor/patch_utils.py`
  - `/home/ubuntu/sdr/download/dcu_megatron/adaptor/megatron_adaptor.py`
