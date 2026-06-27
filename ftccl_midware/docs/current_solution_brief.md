# 非侵入式大模型训练容错中间件：当前实现汇报简版

## 1. 一句话目标

在不修改 Megatron-LM 训练主流程的前提下，当一个 DP rank 失效或被判定异常时，底层 FTCCL/NCCL 通信域动态收缩为 survivor ranks，上层 Megatron 继续按原进程组运行，并在 Python runtime 中修正 DP 视角、batch 语义和恢复流程。

## 2. 当前最有价值的实现

### FTCCL survivor communicator bypass

FTCCL 在 NCCL communicator 内部支持 survivor topology 切换：

```text
原始 DP ranks: 0,1,2,3,4,5,6,7
异常 rank:     7
survivors:     0,1,2,3,4,5,6
真实通信域:    8-rank -> 7-rank
```

Megatron/PyTorch 仍调用原来的 NCCL collective，`LD_PRELOAD` 中间件在 C ABI 层拦截后调用 FTCCL survivor API 完成底层通信域收缩。

### Megatron runtime patch

Python 层通过 `sitecustomize.py` 自动注入，不侵入 Megatron 源码。核心职责：

- 将 Megatron 看到的 DP world/rank 映射成 survivor view。
- 在 rank 下线后重建 dataloader/sampler，使 survivor ranks 继续消费数据。
- 在通信错误返回后做一次训练步 rollback/retry。
- 对 survivor 训练语义做 batch 与梯度补偿。

### FTCCL 内核级 CCL-D 指标接入

当前已经把检错信号从单纯 host preflight 推进到 FTCCL 内核 primitive 埋点：

- FTCCL `ProtoSimple` send/recv primitive 内部累计 `sendCount`、`recvCount`、`sendBytes`、`recvBytes`、post count、wait spins 和 timestamp。
- FTCCL 暴露扩展 API：`ncclCommCcldGetMetrics()` / `ncclCommCcldResetMetrics()`。
- 中间件在 collective probe 中读取真实 FTCCL kernel metrics，并把异常判断结果发布到同一套 bypass control plane。

需要明确：这不是论文完整 CCL-D 复刻。当前覆盖的是最适合现阶段实验闭环的部分：FTCCL `NCCL_PROTO=SIMPLE` 内核 primitive 指标 + 中间件级 analyzer + bypass 触发。

## 3. 整体闭环

```text
Megatron training step
  -> PyTorch ProcessGroupNCCL
  -> LD_PRELOAD C middleware intercepts NCCL C ABI
  -> detector/probe checks collective consistency and FTCCL kernel metrics
  -> publish canonical bypass request
  -> FTCCL survivor topology prepare/activate
  -> Python patch catches recoverable communication error
  -> rollback/retry current train step
  -> Megatron continues with survivor DP view
```

统一控制面文件位于 `FTCCL_BYPASS_DIR`，核心状态包括：

```text
bypass_request.json
failed_rank
survivors
generation
trigger
propagate_error
```

这样主动下线、检测触发、真实 rank kill 后的 supervisor 决议都汇聚到同一个恢复入口。

## 4. 实验验证重点

### 逻辑下线后继续训练

日志：

```text
/home/ubuntu/sdr/demo/ftccl-midware/ftccl_megatron_final.log
/tmp/ftccl_megatron_patch.log
```

关键证据：

```text
rank 7 entering logical victim mode=noop
rank 0-6 bypass complete: survivorCount=7
iteration 2/3 ... global batch size: 126
iteration 3/3 ... global batch size: 126
[after training is done]
```

说明：底层通信域已经从 8 个 DP rank 收缩到 7 个 survivor rank，Megatron 没有退出，并继续完成后续训练 iteration。

### 真实 rank kill 后恢复

日志：

```text
/tmp/ftccl_real_kill_logs/combined.log
```

关键证据：

```text
pause_ready iteration=1
continue_after_kill received iteration=1
propagating recoverable ncclSystemError
caught recoverable communication error
rollback complete; retrying train_step
bypass complete: survivorRank=<0..6> survivorCount=7
[after training is done]
```

说明：rank 7 真实退出后，survivor ranks 收到一次可恢复通信错误，Python patch 完成 rollback/retry，FTCCL 完成 survivor topology 激活，训练继续结束。

### 梯度补偿与 batch 语义

日志：

```text
/tmp/ftccl_megatron_gradcomp.log
```

关键证据：

```text
global batch size: 126
applied gradient compensation generation=1 strategy=old_world_size scale=1.1428571 tensors=98
iteration 3/3
[after training is done]
```

说明：rank 下线后 survivor 训练不只是通信继续，Megatron 层的 batch 和梯度缩放语义也已经接入实验闭环。

## 5. 当前边界

- 当前 FTCCL CCL-D 指标只覆盖 `NCCL_PROTO=SIMPLE`，LL/LL128 还没有埋点。
- 当前 analyzer 运行在中间件 probe 路径中，不是独立常驻的完整 CCL-D analyzer。
- 当前实验重点是单机 8 GPU、纯 DP、单 rank 下线；多节点、多 rank、TP/PP/ZeRO 需要进一步扩展。
- 当前 FTCCL CCL-D 改动正在按 RTX 5090 对应 `sm_120` 重新编译验证；源码已接入，最终运行验证以新 `libnccl.so` 链接完成后的 smoke/Megatron 日志为准。

## 6. 汇报结论

当前实现已经形成最关键的实验闭环：检测信号进入统一控制面，FTCCL 收缩真实通信域，Python runtime 修正 Megatron 训练语义，survivor ranks 在 rank 下线后继续训练。相比单纯脚本重启或外部调度恢复，这条路径的价值在于不重启训练 job、不改 Megatron 主代码，并把通信层容错和训练层语义恢复连成一条可验证链路。
