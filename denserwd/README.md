# denserwd

一个用于训练机器人操作任务的视觉-语言-动作（VLA）模型和奖励模型的综合性研究框架，支持模拟和真实机器人环境。

## 项目概述

denserwd 是一个专为机器人操作基准任务上的具身智能体训练和评估而设计的研究框架。该项目主要关注：

- **奖励模型训练**：使用 Octo 架构训练奖励模型，为机器人任务提供密集奖励信号
- **VLA 训练**：训练视觉-语言-动作模型，支持多种后端（OpenVLA、π0、GR00t 等）
- **数据收集**：为 LIBERO 和 ManiSkill 基准提供自动化数据收集管道
- **强化学习**：支持 PPO 和 GRPO 算法，并结合奖励模型指导
- **监督微调**：支持单样本和少样本学习场景的 SFT 能力

### 核心特性

- 结合视觉、语言和机器人状态的多模态奖励模型
- 支持多种预训练 VLA 模型（π0、π0.5）
- 使用 FSDP 后端的灵活训练管道，支持分布式训练
- 在 LIBERO 和 ManiSkill 基准上的全面评估
- 自动化数据收集和预处理工具
- 与 RLinf 框架集成，实现可扩展的强化学习训练

## 安装
1. 首先需要安装 RLinf（建议用docker）
2. 安装好 RLinf 后，cd 到工作目录如 /workspace/RLinf，再执行 `git clone git@github.com:duowuyms/denserwd.git`，将 denserwd 安装到 RLinf 目录下作为其子目录。
3. 默认环境使用 openpi，可在 docker 内通过 `source switch_env openpi` 实现
3. 如果需要启动 Octo 奖励模型训练的代码，则需安装相关依赖：
```bash
# 安装依赖
pip install -r requirements.txt
```

## 目录结构

```
denserwd/
├── config/                    # 训练和评估的配置文件
│   ├── reward_model/         # 奖励模型训练与测试配置
│   │   ├── default.yaml
│   │   ├── libero_object.yaml
│   │   ├── libero_spatial.yaml
│   │   ├── libero_goal.yaml
│   │   └── libero_10.yaml
│   │   └── ...
│   └── vla/                 # VLA 训练与测试配置
│       ├── model/           # 模型架构配置
│       │   ├── openvla.yaml
│       │   ├── pi0.yaml
│       │   ├── gr00t.yaml
│       │   └── flow_policy.yaml
│       │   └── ...
│       ├── env/              # 环境配置
│       ├── reward_model/     # 奖励模型配置（通常为训练好后的奖励模型）
│       └── training_backend/ # 训练后端配置
│
├── datasets/                # 数据集实现
│   ├── datasets.py          # 针对Octo奖励模型训练实现的主要数据集类（SequenceDataset 等）
│   └── lerobot_dataset.py   # LeRobot 数据集自定义，实现 VLA few-shot SFT
│
├── envs/                     # 重写 RLinf 环境和工作器，以支持功能自定义，如集成 Octo 奖励模型
│   ├── env_worker.py        # RL 训练的环境工作器
│   └── libero_rm.py         # 集成 Octo 奖励模型的 LIBERO 环境
│
├── models/                   # 奖励模型实现
│   ├── octo.py              # OctoReward 模型实现
│   └── encoders/            # 编码器模块
│       ├── image.py         # ResNet50 图像编码器
│       ├── language.py      # T5 语言编码器
│       ├── robot.py         # 机器人状态编码器
│       └── timestep.py      # 时间步编码器
│
├── workers/                  # 重写 RLinf 工作器以支持功能自定义
│   ├── fsdp_sft_worker.py   # 基于 FSDP 的 SFT 工作器
│   └── huggingface_rollout_collector.py  # 轨迹收集工作器
│
├── pipelines/                # Octo 奖励模型训练管道
│   ├── training.py          # 主训练管道
│   └── testing.py           # 测试和评估管道
│
├── utils/                    # 工具函数
│   ├── data_utils.py        # 数据处理工具
│   ├── dataset_utils.py     # 数据集特定工具
│   ├── tensor_utils.py      # 张量操作工具
│   └── utils.py             # 通用工具
│
├── octo/                     # Octo 模型组件（改编自 Octo）
│   ├── model/               # 核心 Octo 模型组件
│   │   ├── octo_model_pt.py
│   │   ├── components/      # Transformer 模块、编码器等
│   │   └── octo_module.py
│   ├── data/                # 数据处理组件
│   └── utils/               # Octo 特定工具
│
├── scripts/                  # 实用脚本
│   ├── download.py          # 模型和数据下载脚本
│   ├── analyze_success_rate.py  # 成功率分析
│   ├── plot_training_curves.py  # 可视化工具
│   ├── visualize_demo.py    # 演示可视化
│   ├── read_hdf5.py         # HDF5 数据读取工具
│   └── copy_parquet_by_task_index.py  # 数据处理
│
├── train_embodied_agent.py      # 具身智能体训练主脚本 (RLinf原版)
├── train_embodied_agent_rm.py   # 具身智能体训练主脚本 (支持Reward Model版本)
├── train_embodied_sft.py        # SFT 训练脚本 (RLinf原版)
├── train_embodied_sft_fs.py     # SFT 训练脚本 (支持Few Shot版本)
├── train_reward_model.py        # 奖励模型训练脚本
├── eval_embodied_agent.py       # 评估脚本 (RLinf原版)
├── collect_libero_data.py       # LIBERO 数据收集
├── collect_maniskill_data.py    # ManiSkill 数据收集
├── run_embodiment.sh            # 训练启动脚本 (RLinf原版)
├── run_embodiment_sft.sh        # SFT 训练启动脚本 (RLinf原版)
└── eval_embodiment.sh           # 评估启动脚本 (RLinf原版)
```

## 核心组件

### 1. 奖励模型 ([models/octo.py](models/octo.py))

使用 Octo 架构实现多模态奖励模型：
- **语言编码器**：T5-base 用于指令编码
- **视觉编码器**：ResNet50 用于图像特征提取
- **机器人状态编码器**：基于 MLP 的机器人状态编码器
- **时间步编码器**：用于时间信息的正弦编码
- **Transformer**：用于特征融合和奖励预测的 Block Transformer

### 2. 数据集 ([datasets/datasets.py](datasets/datasets.py))

支持以下功能的数据集实现：
- 包含轨迹数据的 HDF5 格式
- 多种观察模态（图像、机器人状态）
- 灵活的序列长度和采样策略
- 奖励标注和边界数据提取
- 目前支持 LIBERO 数据格式

### 3. 奖励模型训练管道 ([pipelines/training.py](pipelines/training.py))

灵活的训练管道支持：
- 使用课程学习的奖励模型训练
- 在保留轨迹上进行评估
- 使用 Weights & Biases 进行检查点保存和日志记录

### 4. RLinf 环境重定义 ([envs/](envs/))

用于 RL 训练的环境集成：
- [env_worker.py](envs/env_worker.py)：基础环境工作器
- [libero_rm.py](envs/libero_rm.py)：集成奖励模型的 LIBERO 环境

## 使用方法

### 训练奖励模型

```bash
python train_reward_model.py \
    --config_name libero_spatial \
    --device cuda:0 \
    --data_dir data/datasets/libero_spatial \
    --seed 42
```

### 训练具身智能体（使用 VLA）
1. RLinf 原版：
```bash
bash run_embodiment.sh libero_spatial_ppo_openpi 
```
2. Reward Model 版本：
```bash
python train_embodied_agent_rm --config-name libero_spatial_ppo_openpi --cuda-list 0,1,2,3
```

### 收集数据

```bash
# LIBERO 数据收集
python collect_libero_data.py

```

### 评估

```bash
bash eval_embodiment.sh
```

## 配置说明

项目使用 Hydra 进行配置管理。主要配置目录：

- **[config/reward_model/](config/reward_model/)**：奖励模型训练与测试配置
  - `default.yaml`：默认配置
  - `libero_*.yaml`：LIBERO 基准的任务特定配置

- **[config/vla/](config/vla/)**：VLA 训练与测试配置
  - `model/`：模型架构配置（OpenVLA、π0、GR00t 等）
  - `env/`：环境配置
  - `training_backend/`：分布式训练配置

## 支持的基准

- **LIBERO**：LIBERO Spatial、LIBERO Object、LIBERO Goal、LIBERO-10

## 致谢

本项目基于以下工作构建：
- [Octo](https://github.com/octo-models/octo)：通用机器人策略模型
- [RLinf](https://github.com/anthropics/rlinf)：可扩展强化学习训练框架
