# RLinf 迁移运行说明（官方 Pi05 LIBERO）

这是一份给解包后 Agent 使用的操作手册。目标是在**不使用 Docker**的前提下，使用官方 Hugging Face 模型 `RLinf/RLinf-Pi05-LIBERO-SFT` 启动 `run_tmux_binary.sh`。

## 0. 前置条件

目标机器需要：

- Linux、NVIDIA GPU 和可用的 NVIDIA 驱动；推荐至少 2 张 GPU。
- `python` 3.11、`git`、`curl`/`wget`、`tmux`、`ray` 安装能力。
- 足够磁盘：Pi05 模型和依赖较大；模型不要放进代码 tar。
- EGL/GL 系统库。若有 sudo，安装脚本会调用 `requirements/embodied/sys_deps.sh`；没有 sudo 则先让管理员安装这些库。
- Hugging Face 网络访问，或配置 `HF_ENDPOINT`/代理。

Docker、`models_ckpts`、本地训练结果和本地权重都不在代码包中。

## 1. 解包

```bash
mkdir -p /path/to/work
cd /path/to/work
tar -xzf /path/to/RLinf-RLSF-code-runpack.tar.gz
cd RLinf-RLSF
```

不要把代码包解到只读目录。训练日志和输出会写入仓库下的 `logs/`、`data/models/` 和配置指定的结果目录。

## 2. 建立 Python 环境

推荐使用仓库自带的 uv 安装器（它创建标准 venv，不是 Docker）：

```bash
bash requirements/install.sh embodied \
  --model openpi \
  --env maniskill_libero \
  --venv .venv
source .venv/bin/activate
```

如果目标机已有 Conda，也可以让 `VENV_DIR` 指向一个包含 `bin/activate` 的兼容环境，但最稳妥的是使用上面的 `.venv`，因为安装脚本会把 LIBERO、OpenPI tokenizer 和环境路径写入该环境。

没有 root 权限时：

```bash
bash requirements/install.sh embodied \
  --model openpi \
  --env maniskill_libero \
  --venv .venv \
  --no-root
```

`--no-root` 仅适用于系统 EGL/GL、FFmpeg、Vulkan/显卡运行库已经由管理员准备好的机器。

## 3. 下载官方权重

默认位置是 `/mnt/storage/public_data/models/RLinf-Pi05-LIBERO-SFT`。使用仓库脚本下载：

```bash
source .venv/bin/activate
python -m pip install -U "huggingface_hub[cli]"
bash tools/prepare_pi05_libero.sh
```

官方来源：<https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>

如果不能使用 `/mnt/storage/public_data`，下载到任意可写目录：

```bash
MODEL_DIR=/path/to/models/RLinf-Pi05-LIBERO-SFT
bash tools/prepare_pi05_libero.sh "$MODEL_DIR"
```

启动时用同一个路径：

```bash
PUBLIC_DATA_DIR=/path/to/models-parent \
RLINF_PI05_MODEL_PATH=/path/to/models/RLinf-Pi05-LIBERO-SFT \
GPUS=0,1 \
bash run_tmux_binary.sh
```

`RLINF_PI05_MODEL_PATH` 优先级最高；它会覆盖配置中的默认路径。

## 4. 启动

两卡默认启动：

```bash
GPUS=0,1 RAY_NUM_CPUS=32 bash run_tmux_binary.sh
tmux attach -t rlinf_train
```

常用参数：

```bash
GPUS=2,3 RAY_PORT=29501 TMUX_SESSION=rlinf_train_2 bash run_tmux_binary.sh
```

启动脚本会：

1. 检查 `.venv` 和官方权重目录；
2. 设置 `CUDA_VISIBLE_DEVICES`、EGL 和仓库 `PYTHONPATH`；
3. 启动本机 Ray head；
4. 执行 `denserwd/run_embodiment_hcrm.sh libero_10_grpo_openpi_hcrm`。

该入口使用仓库已有的官方 Pi05 配置，训练配置中的 actor/rollout GPU 布局是 2 卡。不要使用旧的 `run_tmux_binary.sh` 备份或 `libero_10_grpo_openpi_pi05` 配置来加载 `models_ckpts/global_step_1500`，那是另一种 `openpi_pytorch` checkpoint 格式。

## 5. 启动前快速验证

```bash
source .venv/bin/activate
python - <<'PY'
import torch
import ray
import mani_skill
import libero
import openpi
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())
print('imports: OK')
PY

nvidia-smi
```

如果 import 失败，先不要启动 Ray；检查 Python 是否来自 `.venv/bin/python`，以及 `requirements/install.sh` 是否完整执行。

## 6. 排错

- `Missing environment`：执行第 2 步，或设置 `VENV_DIR` 指向正确环境。
- `Missing official checkpoint`：执行第 3 步，确认目录中包含模型文件和 `physical-intelligence/libero` norm stats。
- GPU 数量/布局错误：用 `GPUS=0,1` 这样的逗号列表覆盖默认值，并确保至少 2 张 GPU。
- EGL/MuJoCo 报错：确认 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`，并让管理员补齐系统 GL/EGL 库。
- Ray 端口占用：改用 `RAY_PORT=29501`，并使用不同的 `TMUX_SESSION`。
- 内存不足：降低配置中的环境数量或 `RAY_NUM_CPUS`；不要同时启动多个训练实例。

## 7. 代码包内容

`pack_code_runpack.sh` 生成的是纯代码包，明确排除 Docker 文件、`.venv`、权重、`data`、`logs`、`results`、`assets`、Git 和缓存。权重与 Python 环境必须在目标机单独建立。
