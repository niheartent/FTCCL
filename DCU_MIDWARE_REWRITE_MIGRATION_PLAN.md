# FTCCL Midware DCU 重新移植计划

本文档用于新的会话从头重写 DCU 版 `ftccl_midware`，不要继续在当前 `ftccl_midware_dcu/` 的失败实现上修补。用户计划删除 `ftccl_midware_dcu/`，因此本文档放在 `ftccl` 根目录：

```text
/public/home/scnethpc26107/sdr/ftccl/DCU_MIDWARE_REWRITE_MIGRATION_PLAN.md
```

## 1. 背景和目标

原始 NV/CUDA 版本位于：

```text
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware
```

这个版本已经形成过清晰的设计：通过 `LD_PRELOAD` 拦截 NCCL C ABI，在通信层调用 FTCCL/FCCL survivor topology API；同时通过 `PYTHONPATH` 注入 `sitecustomize.py`，在 Megatron Python runtime 中修正 DP 视角、dataloader、rollback/retry 和 batch 语义。相关原始文档：

```text
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware/docs/megatron_dp_rank_fault_tolerance_design.md
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware/docs/current_solution_brief.md
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware/docs/current_technical_stack.md
```

DCU 迁移目标是基于这个原始版本重新移植，而不是延续当前 `ftccl_midware_dcu/` 中累积的临时改动。最终目标：

- 在 DCU/RCCL 环境下保持 Megatron 源码不变。
- Megatron 启动脚本仍是主入口，FTCCL 只通过环境变量接入。
- 底层通信库替换成改过的 RCCL/FTCCL。
- DP-only 训练中手动 kill 一个 DP rank 后，CCL-D/控制面触发 bypass，survivor DP ranks 继续训练到指定 iteration。

## 2. 必须遵守的范围

当前只做 DP 容错：

```text
TP=1
PP=1
CP=1
DP=N
```

不要处理 TP/PP/CP 容错，不要处理 tensor parallel shard 丢失后的继续训练。之前出现过 TP=8、DP=1 的误用，这种场景 kill rank 以后会丢 tensor shard，不属于本阶段目标。

DCU 精度范围：

- 主要支持 BF16。
- 当前不支持 FP8。
- Megatron 训练脚本必须使用 BF16 路径，不要启用 H100/FP8 配置。

接入方式：

- 不修改 Megatron 源码文件，除非用户明确批准。
- 使用 Megatron `examples/llama/*.sh` 作为训练入口。
- 在 `.sh` 里引入 `LD_PRELOAD`、`PYTHONPATH`、`FTCCL_*` 环境变量。
- 日志和 run 目录放在 Megatron example 工作目录下，例如：

```text
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/runs/
```
```

共享控制目录应随 run 创建，例如：

```text
${RUN_ROOT}/sync
${RUN_ROOT}/logs
```

## 3. 当前 DCU 已确认事实

DCU/RCCL 基础环境：

```text
RCCL version : 2.22.3-master:07a3100
HIP version  : 6.3.26113-6699a1f0
ROCm version : 6.3.3.0-0-c1d64df
Custom RCCL  : /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/build
```

当前 DCU C/RCCL bypass 层出现过正向结果：

- `run_ccld_16gpu.sh` normal allreduce 可完成并 dump CCL-D metrics。
- `run_bypass_16gpu_ccld.sh` kill rank 3 后，15 个 survivor rank 完成 `SURVIVOR_NATIVE_DONE`。
- `ftccl_midware_dcu` smoke 测试在正确 hostfile 和 rank 分配下跑通过。
- RCCL middleware 在 Megatron run 中能够写出 bypass 控制面并打印：

```text
bypass complete: survivorRank=<0..6> survivorCount=7
```

这些只能说明通信层 bypass 有基础，不等于 Megatron 训练容错成功。

## 4. 当前失败事实

最近 DP Megatron run：

```text
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/runs/dp_ftccl_20260622-100226/logs/mpirun.log
```

日志现象：

```text
iteration        3/       5
pause_ready iteration=3
rank 7 self-kill / external kill
survivor ranks catch recoverable communication error
bypass complete: survivorRank=<0..6> survivorCount=7
[after training is done]
```

但没有：

```text
iteration        4/       5
iteration        5/       5
```

因此这个 run 不是成功。内部 marker 例如 `train_step_done.*.4` 不能作为成功证据；必须以 Megatron 官方训练日志为准。

当前失败的核心判断：

- C/RCCL 层 bypass 大概率已经完成。
- Python runtime patch 没有正确维持 `dcu_megatron.training.training.train()` 的外层训练语义。
- 只 wrap `train_step()`、写内部 marker、或者临时修改 rerun validation 都不足以证明外层 train loop 会继续到 `train_iters=5`。

## 5. 成功验收标准

最小验收必须同时满足：

```text
iteration        5/       5
DP_FTCCL_DONE
```

并且日志中能看到：

```text
pause_ready iteration=3
kill rank 7 或 failed_rank=7
FTCCL_BYPASS_REQUEST generation=1 failed_rank=7 survivors=[0..6]
bypass complete: survivorRank=<0..6> survivorCount=7
survivor DP view 8->7
```

不能把以下内容单独当作成功：

- 只看到 `bypass complete`。
- 只看到 `[after training is done]`。
- 只看到 `train_step_done.*.4`。
- 只看到脚本进程返回 0，但没有 `iteration 5/5`。

## 6. 推荐重写原则

第一原则：回到原始 NV 版本。

新的 DCU 目录应该从这里复制：

```text
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware
```

而不是从旧的：

```text
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu
```

第二原则：先让原始语义在 DCU 上等价，再做 Megatron kill。

顺序必须是：

1. DCU 编译和 smoke。
2. RCCL survivor API 确认。
3. Megatron BF16 正常训练，不接 kill。
4. Megatron DP-only + FTCCL 注入，不接 kill。
5. Megatron DP-only + pause at iteration 3 + 手动 kill + bypass + 继续到 iteration 5。

不要一开始就同时改 C 层、Python patch、Megatron launcher、dataset、precision、kill supervisor。

第三原则：launcher 归 Megatron。

不要再把 Megatron 训练反向包装进 `ftccl_midware_dcu/scripts/*.py` 作为主流程。应该在：

```text
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/
```

维护 `.sh` 训练入口，在这个 `.sh` 中设置 FTCCL 环境变量。FTCCL 中间件只提供库、Python patch、bypassctl 和说明。

## 7. 迁移任务分解

### 7.1 重新创建 DCU 工作目录

建议流程：

```bash
cd /public/home/scnethpc26107/sdr/ftccl
mv ftccl_midware_dcu ftccl_midware_dcu.bak_$(date +%Y%m%d-%H%M%S)
cp -a ftccl_midware ftccl_midware_dcu
```

如果用户已经删除 `ftccl_midware_dcu/`，直接 `cp -a` 即可。

### 7.2 C/CMake 层迁移

从 CUDA/NCCL 迁到 HIP/RCCL/DTK。

需要重点检查：

```text
CMakeLists.txt
src/ftccl_midware.cu
tests/ftccl_allreduce_smoke.cu
scripts/run_smoke.sh
```

预期变化：

- `.cu` 可改为 `.cpp`/`.hip.cpp`，或保留文件名但用 `hipcc` 编译。
- `<cuda_runtime.h>` 改为 `<hip/hip_runtime.h>`。
- `cudaStream_t` 改为 `hipStream_t`，或者确认 RCCL headers 对 `cudaStream_t` 的 hipify typedef。
- `cudaGetDeviceCount` / `cudaSetDevice` / `cudaMemcpy` 等改为 HIP API。
- `cuda_bf16` 不应作为 DCU 主依赖；BF16 使用 PyTorch/Megatron BF16 训练路径，C smoke 只需验证通信。
- `libnccl.so`/`librccl.so` 链接到 custom RCCL build。

必须保留的 interpose ABI：

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

必须保留的 FTCCL/RCCL 扩展符号：

```c
ncclCommPrepareSurvivorTopo(...)
ncclCommActivateSurvivorTopo(...)
ncclCommCcldGetMetrics(...)
ncclCommCcldResetMetrics(...)
```

如果符号名在 DCU custom RCCL 中不同，先改中间件符号解析，不要改 Megatron。

### 7.3 DCU 环境要求

所有脚本要显式要求用户加载 DTK/RCCL 环境。已知 module 示例：

```bash
module load compiler/dtk/26.04
source /public/home/scnethpc26107/sdr/env.sh
```

运行前必须保证：

```text
DTKROOT=/public/software/compiler/dtk-26.04
HIP_PATH=/public/software/compiler/dtk-26.04/hip
ROCM_PATH=/public/software/compiler/dtk-26.04
LD_LIBRARY_PATH contains:
  /public/software/compiler/dtk-26.04/hip/lib
  /public/software/compiler/dtk-26.04/llvm/lib
  /public/software/compiler/dtk-26.04/lib
  /public/software/compiler/dtk-26.04/lib64
  /opt/hyhal/lib
  /opt/hyhal/lib64
```

之前出现过：

```text
open comgr lib:libamd_comgr.so error
load comgr library error
ImportError: libgcvm.so.17git
```

这些优先按 DTK module/LD_LIBRARY_PATH 处理，不要靠 `/tmp` 或临时 copy so 解决。

### 7.4 smoke 脚本要求

`scripts/run_smoke.sh` 必须用 hostfile 控制规模，不能硬编码进程都在一个节点。

要求：

- 参数只接收 hostfile 和可选 bypass/dead-rank。
- `np` 根据 hostfile slots 自动计算。
- `map_by` 根据每节点 slots 自动生成，例如 `ppr:8:node`。
- hostfile 中每行类似：

```text
h15r1n01 slots=8
```

错误案例：

- 在 login 节点直接跑导致 `no ROCm-capable device is detected`。
- `--np 32` 但 hostfile 只有一个节点，导致 RCCL duplicate GPU detected。
- hostfile 路径写错导致 mpiexec 参数解析失败。

成功 smoke 后再进入 Megatron。

### 7.5 Python runtime patch 迁移

从原始 NV 版本迁移：

```text
ftccl_midware/python/sitecustomize.py
ftccl_midware/python/ftccl_megatron_patch/installer.py
ftccl_midware/python/ftccl_megatron_patch/runtime.py
ftccl_midware/python/ftccl_megatron_patch/state.py
ftccl_midware/python/ftccl_megatron_patch/patch_utils.py
```

只做必要 DCU/Megatron 兼容：

- 支持 `OMPI_COMM_WORLD_RANK`、`PMI_RANK`、`RANK` 的 rank 解析。
- 支持 `dcu_megatron` 可能存在的 module path。
- 保持 `sitecustomize.py` 自动注入。
- 控制目录默认来自 `FTCCL_BYPASS_DIR`，不使用 `/tmp`。
- 日志默认写到 `${RUN_ROOT}/logs` 或 stderr。

不要先引入临时机制：

- 不要用内部 `train_step_done.*` 作为训练成功判据。
- 不要只靠 `training_log_wrapper` 判断外层训练继续。
- 不要随意关闭 Megatron rerun validation，除非已经定位到必需且有日志证明。

关键排查点：

`dcu_megatron.training.training.train()` 外层 train loop 在 recoverable error 以后为什么停止。必须阅读该函数和它调用的 `train_step()` 返回值语义，确认：

- `train_step()` 异常被捕获后，外层 iteration 是否仍会递增。
- retry 后返回的 loss/skip/nan/reduced_loss 是否满足外层逻辑。
- `should_exit`、rerun state machine、checkpoint flag 是否被错误置位。
- dataloader rollback/rebuild 是否导致 StopIteration 或 all ranks 不一致。

### 7.6 Megatron 训练脚本

训练脚本放在：

```text
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/
```

推荐保留两个入口：

```text
run_llama8B.sh        # 正常 BF16 baseline
run_dp_ftccl.sh       # DP-only + FTCCL 注入 + 可选 kill
```

`run_dp_ftccl.sh` 应该只是在已跑通的 Megatron BF16 baseline 上增加：

```bash
export LD_PRELOAD=/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu/build/libftccl_midware.so:${LD_PRELOAD:-}
export PYTHONPATH=/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu/python:${PYTHONPATH:-}
export FTCCL_BYPASS_DIR=${RUN_ROOT}/sync
export FTCCL_LOG_DIR=${RUN_ROOT}/logs
```

以及必要的：

```bash
export NCCL_ALGO=RING
export NCCL_PROTO=SIMPLE
export NCCL_CCLD_ENABLE=1
export FTCCL_BYPASS_ENABLE=1
export FTCCL_DETECT_ENABLE=1
export FTCCL_DETECT_CCLD_ENABLE=1
```

Megatron 参数必须是 DP-only：

```text
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--context-parallel-size 1
--use-distributed-optimizer false 或不启用
--bf16
```

训练规模先轻量：

```text
--train-iters 5
pause/kill at iteration 3
```

数据集和 tokenizer 使用已处理好的路径：

```text
data prefix:
/public/home/scnethpc26107/sdr/requirements/wikitext_llama3_text_document

tokenizer:
/public/home/scnethpc26107/sdr/requirements/llama3.1-8B
```

## 8. 推荐验证顺序和命令模板

### 8.1 编译中间件

```bash
cd /public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu
module load compiler/dtk/26.04
source /public/home/scnethpc26107/sdr/env.sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DRCCL_ROOT=/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/build
cmake --build build -j
```

### 8.2 smoke

```bash
cd /public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu
bash scripts/run_smoke.sh --hostfile /public/home/scnethpc26107/sdr/hostfile
bash scripts/run_smoke.sh --hostfile /public/home/scnethpc26107/sdr/hostfile --bypass --dead-rank 3
```

hostfile 由用户控制规模。当前可用 16 节点版本为：

```text
h15r1n01 slots=8
h15r1n02 slots=8
h15r1n03 slots=8
h15r1n04 slots=8
h15r3n15 slots=8
h15r3n16 slots=8
k14r4n15 slots=8
k14r4n16 slots=8
k14r4n17 slots=8
k14r4n18 slots=8
k14r4n19 slots=8
k14r4n20 slots=8
m13r1n01 slots=8
m13r1n02 slots=8
m13r1n03 slots=8
m13r1n04 slots=8
```

### 8.3 Megatron baseline

先不接 FTCCL：

```bash
cd /public/home/scnethpc26107/sdr/Megatron-LM/examples/llama
bash run_llama8B.sh /public/home/scnethpc26107/sdr/hostfile 1
```

确认官方日志至少能到 `[after training is done]`，并没有 Triton/TE/DTK 动态库错误。

### 8.4 Megatron + FTCCL 无 kill

```bash
cd /public/home/scnethpc26107/sdr/Megatron-LM/examples/llama
bash run_dp_ftccl.sh /public/home/scnethpc26107/sdr/hostfile 1 --train-iters=5
```

必须先确认不 kill 时能完整到：

```text
iteration        5/       5
DP_FTCCL_DONE
```

### 8.5 Megatron + FTCCL + kill

```bash
cd /public/home/scnethpc26107/sdr/Megatron-LM/examples/llama
bash run_dp_ftccl.sh /public/home/scnethpc26107/sdr/hostfile 1 --kill-rank=7 --pause-iter=3 --train-iters=5
```

验收只看：

```text
iteration        5/       5
DP_FTCCL_DONE
```

如果只到 `3/5`，必须判定失败。

## 9. CCL-D 和手动 kill 的关系

当前可以先用手动 kill 验证恢复闭环。完整目标是：

```text
Megatron iteration boundary
  -> manual kill rank
  -> CCL-D / detector / supervisor writes canonical bypass request
  -> C middleware performs RCCL survivor topology bypass
  -> Python runtime patch repairs DP training view
  -> Megatron official train loop continues
```

手动 kill、CCL-D 检测、外部 supervisor 都必须收敛到同一个控制面：

```text
bypass_request.json
failed_rank
survivors
generation
trigger
killed.<rank>
propagate_error
```

不要在不同脚本里各自发明控制文件格式。

## 10. 新会话首先要做的检查

新的会话开始后，先执行只读检查：

```bash
cd /public/home/scnethpc26107/sdr/ftccl
diff -ru ftccl_midware/python ftccl_midware_dcu/python
diff -ru ftccl_midware/src ftccl_midware_dcu/src
```

如果 `ftccl_midware_dcu/` 已删除，则跳过 diff，直接从 `ftccl_midware/` 复制。

然后检查 Megatron 侧 baseline：

```bash
cd /public/home/scnethpc26107/sdr/Megatron-LM/examples/llama
sed -n '1,240p' run_llama8B.sh
sed -n '1,260p' train_llama8B.sh
```

确保 `run_dp_ftccl.sh` 不重新发明训练流程，只复用 baseline 并增加 FTCCL 注入。

## 11. 常见错误和禁止方向

不要把 `mpirun` 所有进程落在一个节点上，否则会触发：

```text
Duplicate GPU detected
invalid usage
```

不要在 login 节点跑 GPU smoke，否则会触发：

```text
no ROCm-capable device is detected
```

不要把 TE/Triton/DTK 动态库问题误判为 FTCCL 问题。比如：

```text
AttributeError: module 'transformer_engine' has no attribute 'pytorch'
ImportError: libgcvm.so.17git
```

这类问题先修 Python/DTK 环境。

不要为了绕过 TE 或 Megatron import 错误直接长期修改 Megatron 源码。可以临时 patch 验证，但最终方案应回到脚本环境和 runtime injection。

不要继续依赖 `ftccl_midware_dcu/scripts/run_megatron_*.py` 作为主训练入口。用户明确要求 Megatron `.sh` 是入口，FTCCL 是被注入的中间件。

不要宣称 `3/5 + after training is done` 成功。当前这正是失败现象。

## 12. 最终交付物

重写完成后应至少包含：

```text
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu/README.md
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu/CMakeLists.txt
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu/src/...
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu/python/...
/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu/scripts/run_smoke.sh
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/run_dp_ftccl.sh
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/train_dp_ftccl.sh
```

README 必须写清：

- DCU/RCCL 版本和 custom RCCL 路径。
- BF16-only，FP8 不支持。
- hostfile/slots 使用方式。
- smoke 命令。
- Megatron baseline 命令。
- Megatron DP FTCCL kill 命令。
- 成功和失败判据。

## 13. 当前结论

现在的问题不是“RCCL bypass 完全不可用”，而是 DCU 版 Megatron runtime patch 没有正确恢复 `dcu_megatron.training.training.train()` 外层训练 loop。下一轮必须从原始 NV `ftccl_midware` 重新移植，先恢复等价结构，再最小化加入 DCU/HIP/RCCL 和 Megatron BF16 兼容。验收只认 Megatron 官方日志到 `iteration 5/5`。
