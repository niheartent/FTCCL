# FTCCL Midware DCU/RCCL Port

这个目录是从 NV 版 `ftccl_midware/` 重新迁移出的 DCU 版中间件，不继承旧 `ftccl_midware_dcu/` 的临时实现。目标仍然保持不修改 Megatron-LM 源码，通过 `LD_PRELOAD` 拦截 RCCL/NCCL 兼容 C ABI，并用 Python `sitecustomize.py` 注入 Megatron runtime patch，实现 DP-only 训练中的单 rank bypass。

当前支持范围：

- `TP=1`、`PP=1`、`CP=1`、`DP=N`。
- BF16 训练路径。
- 不支持 FP8。
- 使用 custom RCCL/FTCCL：`/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/build`。
- 统一控制面仍是 `scripts/ftccl_bypassctl.py`，发布 `bypass_request.json`、`failed_rank`、`survivors`、`generation`、`trigger`、`killed.<rank>`、`propagate_error`。

## 环境

运行前加载 DTK，并确保 `env.sh` 已设置网络/RCCL/PyTorch 相关环境：

```bash
module load compiler/dtk/26.04
source /public/home/scnethpc26107/sdr/env.sh
```

关键路径：

```text
HIP_PATH=/public/software/compiler/dtk-26.04/hip
RCCL_ROOT=/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/build
```

## 编译

在登录节点只准备命令，不要直接运行 GPU 程序：

```bash
cd /public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu
module load compiler/dtk/26.04
source /public/home/scnethpc26107/sdr/env.sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DRCCL_ROOT=/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/build \
  -DHIP_PATH=/public/software/compiler/dtk-26.04/hip
cmake --build build -j
```

确认链接：

```bash
ldd build/libftccl_midware.so | rg 'rccl|amdhip64'
```

## Smoke

`run_smoke.sh` 必须使用 hostfile，不允许默认把所有 rank 放到登录节点或单节点。

```bash
cd /public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu
bash scripts/run_smoke.sh --hostfile /public/home/scnethpc26107/sdr/hostfile
bash scripts/run_smoke.sh --hostfile /public/home/scnethpc26107/sdr/hostfile --bypass --dead-rank 3
```

脚本会从 hostfile 的 `slots=` 自动计算 `np` 和 `--map-by ppr:<slots>:node`。

## Megatron

Megatron 入口放在：

```text
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/run_dp_ftccl.sh
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/train_dp_ftccl.sh
```

先跑无 kill 注入，确认 FTCCL preload 和 Python patch 不破坏正常训练：

```bash
cd /public/home/scnethpc26107/sdr/Megatron-LM/examples/llama
bash run_dp_ftccl.sh /public/home/scnethpc26107/sdr/hostfile 1 --no-kill --train-iters=5
```

再跑计划性 kill：

```bash
cd /public/home/scnethpc26107/sdr/Megatron-LM/examples/llama
bash run_dp_ftccl.sh /public/home/scnethpc26107/sdr/hostfile 1 \
  --kill-rank=7 \
  --pause-iter=3 \
  --train-iters=5
```

默认 `--bypass-source=detector`：脚本只在 iteration 边界 kill victim rank 并放行 survivor，bypass request 由 CCL-D/detector 路径写入。如果需要先验证 supervisor 兜底控制面，可以显式使用：

```bash
bash run_dp_ftccl.sh /public/home/scnethpc26107/sdr/hostfile 1 \
  --kill-rank=7 \
  --pause-iter=3 \
  --train-iters=5 \
  --bypass-source=supervisor
```

run 目录默认在：

```text
/public/home/scnethpc26107/sdr/Megatron-LM/examples/llama/runs/dp_ftccl_<timestamp>/
```

包含：

```text
logs/mpirun.log
logs/supervisor.log
sync/
hostfile_slots
```

## 验收标准

成功必须同时看到：

```text
iteration        5/       5
DP_FTCCL_DONE
```

kill 路径还应看到：

```text
pause_ready iteration=3
FTCCL_BYPASS_REQUEST generation=1 failed_rank=7 survivors=[0, 1, 2, 3, 4, 5, 6]
bypass complete: survivorRank=<0..6> survivorCount=7
activated survivor training view ... dp_size 8->7
```

以下不能单独判定成功：

- 只看到 `bypass complete`。
- 只看到 `[after training is done]`。
- 只看到 `train_step_done.*` marker。
- 只看到脚本返回 0，但没有 `iteration 5/5`。
