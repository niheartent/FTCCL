# FTCCl Megatron Bypass Middleware 当前技术文档

本文总结 `/home/ubuntu/sdr/demo/ftccl-midware` 当前 demo 已经用到的关键技术、模块边界、控制流和测试证据。本文描述的是当前实现，不是最终生产级方案。

## 1. 项目目标

本 demo 的目标是在 Megatron-LM 纯 DP 训练中，通过中间件把一个 DP rank 从底层 NCCL/FCCL 通信域中 bypass 掉，同时尽量不修改 Megatron 源码，让 survivor ranks 继续训练。

当前假设如下：

- 纯数据并行优先，`TP=1`、`PP=1`、`CP=1`。
- 不开启 ZeRO，不开启 Megatron distributed optimizer。
- 一次 bypass 一个 DP rank。
- Megatron 仍按原启动参数拉起进程，例如 8 个 rank；中间件在运行时把有效 DP 通信域切成 7 个 survivor rank。
- 梯度补偿已经有策略入口和 demo 实现，但当前验证重点仍是“触发 bypass 后训练继续运行”。
- 真实生产级“任意 collective 中途死亡恢复”还需要更强的超时、abort、状态快照和重新 rendezvous 机制，本文最后单独列出限制。

## 2. 总体架构

当前实现分成四层：

```text
Megatron-LM training script
  |
  |-- source scripts/megatron_env.sh
  |     设置 LD_PRELOAD、PYTHONPATH、FTCCL_* 环境变量
  |
  |-- Python 进程启动
  |     sitecustomize.py 自动安装 Megatron runtime patch
  |
  |-- Megatron 正常训练
  |     Python patch 修正 DP world/rank、dataloader、rollback、梯度补偿
  |
  |-- PyTorch ProcessGroupNCCL 调用 NCCL C ABI
        LD_PRELOAD C middleware 拦截 NCCL API
        FTCCl/FCCL survivor API 收缩底层 communicator
```

核心文件对应关系：

- `src/ftccl_midware.cu`：C/CUDA 层 `LD_PRELOAD` NCCL interposer。
- `scripts/ftccl_bypassctl.py`：统一 bypass 触发接口。
- `python/sitecustomize.py`：Python 自动注入入口。
- `python/ftccl_megatron_patch/installer.py`：Megatron monkey patch 安装逻辑。
- `python/ftccl_megatron_patch/runtime.py`：训练期 runtime 状态、dataloader 重建、rollback、梯度补偿。
- `python/ftccl_megatron_patch/state.py`：共享控制目录读取、survivor view 计算、marker 同步。
- `scripts/run_smoke.sh`：C 层 all-reduce smoke test。
- `scripts/run_megatron_real_kill_demo.py`：真实 SIGKILL victim rank 的 supervisor demo。

## 3. 统一 Bypass 控制面

当前 demo 使用共享目录作为控制面，默认目录是：

```bash
FTCCL_BYPASS_DIR=/tmp/ftccl_bypass
```

主动下线、被动故障检测、真实 kill demo 都应通过同一个接口发布 bypass 请求：

```bash
./scripts/ftccl_bypassctl.py \
  --sync-dir /tmp/ftccl_bypass \
  request \
  --rank 7 \
  --world-size 8 \
  --source manual \
  --reason operator-drain
```

该命令会原子写入以下文件：

```text
bypass_request.json
failed_rank
survivors
generation
trigger
```

可选行为：

- `--mark-killed`：同时写入 `killed.<rank>`，表示外部 supervisor 或 detector 已经确认 victim 进程死亡。
- `--propagate-error`：写入 `propagate_error`，要求 C 层先返回一次 recoverable NCCL error，用来测试 Python rollback/retry 链路。

控制面文件含义：

- `failed_rank`：原始 DP rank 空间中的被 bypass rank。
- `survivors`：原始 DP rank 空间中的 survivor rank 列表，例如 `0,1,2,3,4,5,6`。
- `generation`：bypass 请求代际，用来避免上层重复应用同一次拓扑变更。
- `trigger`：C 层允许开始 bypass 的触发标记。
- `killed.<rank>`：victim 已进入逻辑隔离或已经被外部杀死。
- `prepare.<rank>` / `activate.<rank>`：survivor ranks 在 C 层 FCCL prepare/activate barrier 上的进度标记。
- `rollback_ready.*` / `rollback_replay.*`：Python 层训练状态回滚和重试标记。

这个设计的重点是：无论主动 drain、脚本手动触发、还是 detector 发现 rank 死亡，都只发布同一种 bypass request。C 层和 Python 层只消费 canonical control files，不关心触发来源。

## 3.1 检错控制流

当前 C 层把检错结果发布到第 3 节的统一 bypass 控制面。实现分为两类信号：

- `ccld-host-preflight`：只在 `LD_PRELOAD` NCCL C ABI 层检查 rank 是否进入同一个 collective、Operation Type Set 是否一致。它不是完整 CCL-D。
- `ftccl-kernel-ccld`：FTCCL 内部 `ProtoSimple` send/recv primitive 埋点，提供真实内核级 send/recv count、bytes、post count、wait spins 和 timestamp，中间件读取这些指标后做进展异常判断。

启用开关：

```bash
NCCL_PROTO=SIMPLE
NCCL_CCLD_ENABLE=1
FTCCL_DETECT_ENABLE=1
FTCCL_DETECT_CCLD_ENABLE=1
FTCCL_DETECT_CCLD_MAX_CHANNELS=64
FTCCL_DETECT_TIMEOUT_MS=5000
FTCCL_DETECT_MARK_KILLED=1
FTCCL_DETECT_PROPAGATE_ERROR=1
FTCCL_DETECT_RUN_ID=<one training run id>
```

每个 collective 在提交给真实 NCCL/FCCL 前，C interposer 会生成：

```text
Trace ID = full-size communicator ordinal + local collective sequence
Operation Type Set = op/count/datatype/redop/root/NCCL_ALGO/NCCL_PROTO
```

随后每个 rank 原子写入：

```text
probe.<run_id>.<comm_ordinal>.<seq>.<rank>
```

detector 在同一 communicator/sequence 上聚合所有 rank 的 probe frame：

- 缺少某个 rank 的 probe 超过 `FTCCL_DETECT_TIMEOUT_MS`：判定为 not-entered 或 slow-at-start rank。
- 所有 rank 都进入但 Operation Type Set 不一致：判定为 inconsistent collective。
- 如果启用了 FTCCL kernel metrics，probe frame 还包含 `ccld_send_count`、`ccld_recv_count`、`ccld_send_bytes`、`ccld_recv_bytes`、`ccld_send_wait_spins`、`ccld_recv_wait_spins` 等累计指标。analyzer 对相邻 trace 做 delta；当多数 rank 的 primitive 指标有进展、只有一个 rank 没有进展时，判定为 `kernel-ccld-progress-stall`，并把 `detector` 写为 `ftccl-kernel-ccld`。
- 判定后写入 `bypass_request.json`、`failed_rank`、`survivors`、`generation`、`trigger`。
- 默认还写入 `killed.<rank>`，支持被动故障场景中 victim 已经无法自己发布 marker。
- 默认写入 `propagate_error`，让当前 collective 先向 Python 层返回一次 recoverable NCCL error，复用 rollback/retry 链路。

当前 FTCCL 侧新增了扩展 API：

```c
ncclCommCcldGetMetrics(comm, maxChannels, outMetrics, outBytes);
ncclCommCcldResetMetrics(comm);
```

这已经不是单纯 host preflight，但也仍不是论文完整 CCL-D。限制包括：只覆盖 `NCCL_PROTO=SIMPLE`，LL/LL128 primitive 未埋点；当前 analyzer 仍复用 interposer/probe 路径，不是独立常驻 analyzer；如果一次 in-flight collective 卡住后应用不再回到 host interposer，当前中间件不能单靠下一轮 probe 判定。

## 4. LD_PRELOAD NCCL 拦截层

`src/ftccl_midware.cu` 使用 `LD_PRELOAD` 抢先加载 `libftccl_midware.so`，覆盖 PyTorch ProcessGroupNCCL 常用的 NCCL C ABI。

当前拦截的入口包括：

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

中间件通过 `dlsym(RTLD_NEXT, ...)` 找到真实 NCCL/FCCL 符号。每次 `ncclCommInitRank` 或 `ncclCommInitRankConfig` 成功后，记录 communicator 状态：

```text
parentRank       原始 rank
parentNranks     原始 communicator 大小
fullSizeOrdinal  full-size communicator 出现序号
deadRank         要 bypass 的原始 rank
survivorRank     当前 rank 在 survivor communicator 中的新 rank
survivorCount    survivor communicator 大小
bypassed         当前 comm 是否已经完成 bypass
victimParked     victim 是否已经隔离
```

中间件只对 `parentNranks == FTCCL_BYPASS_WORLD_SIZE` 的 full-size communicator 做候选 bypass。`FTCCL_BYPASS_MIN_COMM_ORDINAL` 可以跳过初始化阶段较早创建的 communicator，降低破坏 Megatron/PyTorch 初始化同步的风险。

## 5. FCCL Survivor Topology

底层真正改变通信域的是 FCCL 提供的 survivor API：

```c
ncclCommPrepareSurvivorTopo(comm, survivorCount, survivorRanks);
ncclCommActivateSurvivorTopo(comm, survivorCount, survivorRanks);
```

当前 bypass 时序如下：

1. C 层发现 `trigger` 或 `FTCCL_BYPASS_TRIGGER_AT_START=1`。
2. 读取 `failed_rank` 和 `survivors`，没有文件时回退到环境变量 `FTCCL_BYPASS_DEAD_RANK`。
3. victim rank 写入 `killed.<rank>`，并按 victim mode 进入 `noop`、`park` 或 `exit0`。
4. survivor ranks 等待 `killed.<rank>` 出现。
5. survivor ranks 调用 `ncclCommPrepareSurvivorTopo`。
6. survivor ranks 写入 `prepare.<rank>` 并等待所有 survivor prepare marker。
7. survivor ranks 调用 `ncclCommActivateSurvivorTopo`。
8. survivor ranks 写入 `activate.<rank>` 并等待所有 survivor activate marker。
9. C 层更新 `survivorRank/survivorCount/bypassed`，后续 collective 走新拓扑。

victim rank 的模式由 `FTCCL_BYPASS_VICTIM_MODE` 控制：

- `noop`：victim 后续进入 NCCL collective 时直接返回 `ncclSuccess`，不调用真实 NCCL。适合 Megatron 逻辑下线 demo。
- `park`：victim 写 marker 后常驻等待，适合观察和调试。
- `exit0`：victim 直接退出，适合 smoke test 或外部 supervisor 管理进程生命周期的场景。

## 6. NCCL Group 延迟提交与 Replay

PyTorch 常把多个 collective 放在一个 NCCL group 中：

```c
ncclGroupStart();
ncclAllReduce(...);
ncclReduceScatter(...);
ncclGroupEnd();
```

如果在 group 内部某个 collective 直接切换 communicator，容易破坏 NCCL 对 group 调用顺序的假设。因此 C middleware 对 group 做延迟提交：

1. 拦截 `ncclGroupStart()`，只增加 thread-local group depth，不立即调用真实 NCCL。
2. group 内的 collective 被记录成 `PendingOp`，不立即提交。
3. 到最外层 `ncclGroupEnd()` 时，先检查是否需要 bypass。
4. 如需 bypass，先完成 FCCL prepare/activate。
5. 调用真实 `ncclGroupStart()`。
6. 按原顺序 replay pending collective。
7. 调用真实 `ncclGroupEnd()`。

这保证拓扑切换发生在一组 NCCL op 真正提交前，而不是 group 中途。

## 7. Python 注入机制

Python 注入依赖两点：

- 训练脚本把 middleware 的 `python/` 目录放入 `PYTHONPATH`。
- Python 启动时会自动尝试 import `sitecustomize.py`。

当前 `python/sitecustomize.py` 会导入 `ftccl_megatron_patch` 并调用安装函数。安装函数在 `installer.py` 中完成 monkey patch。

Python patch 的优势：

- 不需要直接修改 Megatron-LM 仓库源码。
- 可以替换 Python 层函数绑定，例如 Megatron parallel_state、training loop、dataloader builder。
- 可以感知 C 层写出的 bypass marker，并在训练边界修正 Megatron 上层语义。

Python patch 的边界：

- 不能可靠替换 PyTorch C++ ProcessGroupNCCL 内部已经缓存的 NCCL communicator。
- 不能独立完成 FCCL survivor topology 切换。
- 无法让已经死亡的 Python 进程继续执行，只能由外部 supervisor 管理 survivor 进程。

因此当前方案保留两层：C 层处理通信事实，Python 层处理 Megatron 训练语义。

## 8. Megatron DP 视角修正

`installer.py` patch 了 Megatron 的 `parallel_state`：

```text
get_data_parallel_world_size
get_data_parallel_rank
```

bypass 前返回 Megatron 原始结果。bypass 后返回 survivor view：

```text
原始 DP ranks:      0,1,2,3,4,5,6,7
failed_rank:        7
survivor_ranks:     0,1,2,3,4,5,6
survivor dp size:   7
survivor dp rank:   parent rank 在 survivor_ranks 中的下标
```

`runtime.py` 的 `apply_effective_training_args()` 会在 bypass 后修改 Megatron args：

- `args.data_parallel_size = survivor_count`
- 如设置 `FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS`，则更新 `args.global_batch_size`
- 调用 Megatron `reconfigure_num_microbatches_calculator`
- 调用 `update_num_microbatches(args.consumed_train_samples, consistency_check=False)`

这样 Megatron 的 batch 计算、日志和部分训练逻辑能看到 7-rank survivor DP 视角。

## 9. Dataloader 重建

Megatron 的 dataloader/sampler 通常在训练开始时按 DP rank/world size 构建。如果只切底层 communicator，而不重建 dataloader，数据仍然按 8-way shard 切分，rank 7 的数据贡献会丢失或语义不一致。

当前 Python patch 的处理方式：

1. patch `build_pretraining_data_loader`。
2. 捕获 train dataset 和 consumed samples。
3. patch `train`，把原始 train iterator 包装成 `FaultTolerantDataIterator`。
4. 每次取 batch 前检查 `state.current_view()`。
5. 如果发现新的 bypass generation，先应用有效训练参数，再调用原始 Megatron dataloader builder：

```text
original_build_pretraining_data_loader(train_dataset, args.consumed_train_samples)
```

由于此时 `get_data_parallel_world_size()` 和 `get_data_parallel_rank()` 已被 patch，Megatron sampler 会按 survivor DP size/rank 重新构建。

重建完成后，`FaultTolerantDataIterator` 替换内部 iterable，并清空 rerun/replay 缓存。日志中会出现类似：

```text
dataloader rebuilt generation=... dp_rank=... dp_size=... consumed_samples=...
```

## 10. 通信错误传播

真实故障中，collective 可能先在 PyTorch/NCCL 层表现为错误，而不是干净地到达训练 iteration 边界。当前 demo 提供 recoverable error 注入路径来验证上层恢复逻辑。

触发方式：

```bash
./scripts/ftccl_bypassctl.py --sync-dir /tmp/ftccl_bypass request \
  --rank 7 \
  --world-size 8 \
  --mark-killed \
  --propagate-error
```

C 层看到 `propagate_error` 后，对 survivor rank 的下一次 collective 返回一次 `ncclSystemError`，并写入：

```text
comm_error.<rank>
error_propagated.<rank>
```

Python 层 `train_step` wrapper 捕获异常后，通过字符串特征判断是否是 recoverable communication error，例如：

```text
nccl
ProcessGroupNCCL
DistBackendError
system error
connection closed
collective
```

如果错误可恢复，则进入 rollback/retry。

## 11. 训练状态 Rollback 与 Retry

当前 rollback 不是完整 checkpoint restore，而是 demo 级别的 iteration 内重试：

1. `FaultTolerantDataIterator.begin_iteration()` 开始记录当前 iteration 的 microbatch 状态。
2. `train_step` 捕获 recoverable comm error。
3. survivor ranks 写入 `rollback_ready.<rank>.<generation>.<iteration>`。
4. 等待所有 survivor ranks 到达 rollback barrier。
5. 重新应用 survivor training args。
6. 重置 Megatron rerun state machine。
7. dataloader 根据 survivor view 重建或回放。
8. 写入 `rollback_replay.<rank>.<generation>.<iteration>`。
9. 重试当前 `train_step`。

成功完成训练 step 后会写入：

```text
train_step_done.<rank>.<iteration>
```

这条链路用于验证“通信错误向上传播后，上层可以统一进入 survivor 拓扑、回滚到一致边界并继续训练”。

## 12. 梯度补偿

rank 被 bypass 后，全局 batch 和梯度平均语义会变化。当前 demo 在 optimizer step 前预留并实现了梯度缩放入口：

```bash
FTCCL_PATCH_GRAD_COMPENSATION=survivor_global_batch
```

当前支持策略：

- `survivor_global_batch` / `survivor_mean` / `off`：不额外缩放，认为后续训练以 survivor global batch 继续。
- `old_world_size`：按 `parent_world_size / survivor_count` 放大 survivor 梯度，模拟仍按原 world size 平均。
- `zero_missing_average`：按 `survivor_count / parent_world_size` 缩小，语义上把 missing rank 梯度看成 0。
- `custom`：读取 `FTCCL_PATCH_GRAD_COMPENSATION_SCALE`。

实现位置在 `runtime.py` 的 `apply_gradient_compensation()`。它会扫描 model 参数的 `main_grad` / `grad` 以及 optimizer 参数，在 `optimizer.step()` 前执行 `tensor.mul_(scale)`。

当前默认更适合 demo 的策略是 survivor batch 语义，即不强行补偿 missing rank。生产中应该根据 loss 归一化方式、global batch 定义、LR schedule 和样本消费语义重新确定。

## 13. 真实进程死亡 Demo

`torchrun` 默认会把任意 worker 死亡视为整个 job 失败。为了验证真实 victim rank `SIGKILL` 后 survivor 继续运行，当前 demo 使用外部 supervisor 启动 8 个独立 rank：

```text
scripts/run_megatron_real_kill_demo.py
```

基本流程：

1. supervisor 启动 8 个 Megatron rank。
2. Python patch 在指定 iteration 后写 `pause_ready.<rank>.<iteration>`。
3. supervisor 等待所有 rank 到达 pause 点。
4. supervisor 对 victim rank 发送 `SIGKILL`。
5. supervisor 调用 `ftccl_bypassctl.py request --mark-killed --propagate-error`。
6. survivor ranks 收到 continue marker 后继续训练。
7. C 层完成 survivor topology 切换。
8. Python 层处理 recoverable error、rollback、dataloader rebuild 和继续 train_step。

这证明当前 demo 能覆盖“进程真实死亡 + 外部控制面隔离 + survivor 继续”的受控路径。

## 14. 测试与证据

当前已有的验证类型：

### C 层 all-reduce smoke test

脚本：

```bash
./scripts/run_smoke.sh --np 8
./scripts/run_smoke.sh --np 8 --bypass --dead-rank 7
./scripts/run_smoke.sh --np 8 --bypass --dead-rank 3
```

意义：

- 不触发 bypass 时，8 个 rank 都参与 all-reduce。
- bypass rank 7 时，sum 从 8-rank 语义变成 7-rank survivor 语义。
- bypass rank 3 时，验证 survivor rank 映射不是只适用于最后一个 rank。

### Megatron 逻辑下线 demo

脚本：

```bash
./scripts/run_megatron_fault_tolerance_demo.sh
```

意义：

- Megatron 仍按原参数启动。
- victim rank 可使用 `noop` 逻辑隔离。
- survivor ranks 底层通信域收缩后继续训练。
- Python 层可应用 survivor DP view 和 dataloader rebuild。

### Megatron 真实 kill demo

脚本：

```bash
./scripts/run_megatron_real_kill_demo.py \
  --log-dir /tmp/ftccl_real_kill_logs \
  --sync-dir /tmp/ftccl_bypass_real \
  --master-port 6047
```

关键证据包括：

```text
FTCCL_REAL_KILL_DEMO_PASS
victim_exit=-9
propagating recoverable ncclSystemError
caught recoverable communication error
rollback complete; retrying train_step
bypass complete
[after training is done]
```

这些日志分别对应真实 kill、C 层错误传播、Python 捕获、rollback/retry、FCCL topology activate 和训练正常结束。

## 15. 当前限制

当前 demo 还不是完整生产级容错，主要限制如下：

- 只覆盖纯 DP；TP/PP/CP、ZeRO、distributed optimizer 还没有完整处理。
- 一次只处理一个 failed rank，多 rank 或整节点失效还需要扩展 survivor set、rank mapping 和 checkpoint 语义。
- 依赖共享文件目录作为控制面，适合 demo，不适合高并发生产环境。
- C 层目前主要拦截常见 NCCL collective，其他 NCCL API 或 PyTorch 内部特殊路径需要继续补齐。
- `noop` victim 是逻辑下线，不等同于真实进程死亡。
- 真实 kill demo 依赖外部 supervisor，尚未替代 torchrun 的完整弹性 rendezvous 能力。
- rollback 当前是 iteration 内 demo 级重试，不是完整模型、优化器、随机数、dataloader、scheduler 的事务级快照恢复。
- recoverable error 判断基于异常文本特征，生产中应改成更明确的错误分类和通信状态机。
- 任意 collective 中途死亡的恢复，需要 NCCL async error polling、comm abort、全员故障传播、统一 generation barrier 和训练状态 checkpoint 联动。

## 16. 后续生产化方向

建议后续按以下顺序推进：

1. 把共享文件控制面替换为 supervisor RPC 或高可靠 store，保留 `ftccl_bypassctl.py` 的接口语义。
2. 在 C 层补齐 NCCL async error 检测、communicator abort、generation 防重入和多 communicator 一致切换。
3. 在 Python 层把 rollback 从 iteration 内重试升级为轻量训练事务：model/optimizer/RNG/dataloader/scheduler 一致恢复。
4. 明确 global batch 语义，决定 survivor batch、old-world batch 或 zero-missing batch，并把 LR schedule、consumed samples 和 checkpoint metadata 统一。
5. 扩展到多 rank/整节点 bypass，统一处理 node-local rank、global rank、DP rank 和 survivor rank 映射。
6. 设计和 Megatron checkpoint 的恢复协议，让 bypass 后的 checkpoint 能被后续 7-rank job 或恢复后的 8-rank job 正确加载。

## 17. 一句话总结

当前 demo 的核心技术路线是：用 `LD_PRELOAD` 在 C 层接管 NCCL 调用并通过 FCCL survivor topology 改变真实通信域，用 Python `sitecustomize` 注入在 Megatron 层修正 DP 视角、dataloader、错误回滚和梯度补偿，再用统一 bypass control plane 让主动下线和被动故障走同一条恢复路径。
