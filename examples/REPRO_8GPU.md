# DIAL 8 卡训练完整复现指南（RoboCasa GR1 Tabletop，24GB 显存适配版）

> 傻瓜式操作指南：**按序号逐条执行**，每步含验证方法；所有命令可直接复制（`<...>` 尖括号处替换为你的实际路径）。

---

## 第 0 步：前置条件核对

| 项 | 要求 | 验证命令 |
|---|---|---|
| GPU | **8×24GB**（如 RTX 4090/3090） | `nvidia-smi` |
| 驱动 / CUDA | 驱动 ≥550，CUDA 12.x | `nvidia-smi` 顶部信息 |
| 系统 | Ubuntu 20.04/22.04，Python 3.10 | `python3 --version` |
| 磁盘 | ≥200GB 空闲（数据 52GB + 权重 ~6GB + 输出） | `df -h` |
| 网络 | 能访问 HuggingFace（首次需下载 Qwen 权重 ~6GB） | `curl -sI https://huggingface.co -m 5` |
| 代码 | 完整的 DIAL 仓库（含 `gr00t/`、`scripts/`、`examples/`、`preprocessing/`） | `ls` 确认 |

---

## 第 1 步：安装训练环境（约 30 分钟）

```bash
conda create -n dial python=3.10 -y
conda activate dial
cd /path/to/DIAL        # ← 仓库根目录
pip install -e .[base]
pip install flash-attn==2.7.1.post4 --no-build-isolation
pip install qwen-vl-utils[decord]==0.0.8
pip install scikit-learn
```

**验证**：
```bash
python -c "import torch, transformers, gr00t; print(torch.__version__, transformers.__version__)"
# 期望: 2.5.x  4.52.0
```

**国内网络提示**：pip 命令追加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；flash-attn 编译失败时改用 `pip install flash-attn --no-build-isolation` 或安装预编译 wheel。

---

## 第 2 步：仿真环境（数据预处理与在线评估需要，约 1-2 小时）

> **先判断这一步是否需要做**（重要）：
> - 若你拿到的是**原始数据集**（解压后有 `HDF5/` 和 `LeRobot/` 两个目录，`modality.json` 里**没有** `wrist_r_pos`/`wrist_r_rot6d` 键）→ **必须执行本步**。因为原始数据只有关节角，第 3.2 步解算 EEF 位姿需要在仿真环境中回放轨迹，不装仿真环境就生成不了可训练数据。
> - 若你拿到的是**已预处理数据**（直接就是 `LeRobot-AugPosRot-Correct/` 目录，`modality.json` 里有 `wrist_r_pos`/`wrist_r_rot6d`）→ **跳过本步**，同时跳过 3.1-3.3，直接进第 5 步训练。
> - **如果你直接拿到了一个 `Datasets/` 目录（内含 `LeRobot-AugPosRot-Correct/` 和 `raw/`）**：`LeRobot-AugPosRot-Correct/` 就是**已预处理好的最终训练数据**（24 个任务、含 EEF 位姿）→ **同样跳过本步和 3.1-3.3**。`raw/` 是原始数据备份，训练用不上。
> - 训练本身全程不需要仿真环境；它只用于"数据预处理（3.2）"和"在线评估（eval.sh）"两个环节。
> - 快速判断：`grep -c wrist_r_pos <数据目录>/meta/modality.json`，输出 ≥1 就是已预处理数据。

```bash
# 系统依赖
sudo apt-get install -y libegl1-mesa libegl1-mesa-dev libosmesa6-dev patchelf

# robosuite v1.5.1
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite && git checkout v1.5.1 && pip install -e . && cd ..

# RoboCasa GR1 任务包
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
cd robocasa-gr1-tabletop-tasks && pip install -e . && cd ..

# IK 求解器（GR1 需要）
pip install pin==3.9.0 mink==0.0.5

# 下载桌台仿真资产
python robocasa-gr1-tabletop-tasks/robocasa/scripts/download_tabletop_assets.py -y
```

**必须打一个补丁**（否则部分任务会报错）：编辑
`robocasa-gr1-tabletop-tasks/robocasa/models/objects/kitchen_object_utils.py`，
在 `sample_kitchen_object_helper()` 中 `elif split == "B":` 分支的
`reg_choices = reg_choices[split_th:]` 之后加入：

```python
if "assets/objects/sketchfab/basket/" in reg_choices[0]:
    reg_choices = [c for c in reg_choices if not c.endswith('basket_4/model.xml')]
```

**验证**：`python -c "import robosuite, robocasa, pinocchio, mink; print('OK')"`

---

## 第 3 步：下载并预处理数据（3-6 小时，24 个任务可并行）

### 3.1 下载原始数据
 从 HuggingFace 下载 `nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim`，解压后得到：
```
<DATA_ROOT>/PhysicalAI-Robotics-GR00T-Teleop-Sim/
├── HDF5/
│   ├── PnPBottleToCabinetClose.hdf5          # 每个任务 1 个 HDF5
│   └── ...
└── LeRobot/
    ├── gr1_unified.PnPBottleToCabinetClose/ # LeRobot 格式
    └── ...
```

### 3.1.1 数据已下载时，目录怎么改？（重要）

**本指南中所有 `<DATA_ROOT>` 只是一个占位符**，你只需记住下面 3 条映射规则，把后续命令里的 `<DATA_ROOT>` 全部替换成你的实际路径即可。

假设你把下载的压缩包解压到了 `/data/GR1/`（举例），解压后实际目录为
`/data/GR1/PhysicalAI-Robotics-GR00T-Teleop-Sim/`，则：

| 指南中的写法 | 实际替换为（示例） |
|---|---|
| `<DATA_ROOT>/HDF5/<TaskName>.hdf5` | `/data/GR1/PhysicalAI-Robotics-GR00T-Teleop-Sim/HDF5/<TaskName>.hdf5` |
| `<DATA_ROOT>/LeRobot/gr1_unified.<TaskName>` | `/data/GR1/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/gr1_unified.<TaskName>` |
| `<DATA_ROOT>/Replay-Correct/<TaskName>` | `/data/GR1/Replay-Correct/<TaskName>`（解算产物，目录不存在会自动创建） |
| `<DATA_ROOT>/LeRobot-AugPosRot-Correct/gr1_unified.<TaskName>` | `/data/GR1/LeRobot-AugPosRot-Correct/gr1_unified.<TaskName>`（**最终训练数据**，预处理后生成） |

**验证你的路径是否正确**（把下面命令中的路径换成你的）：
```bash
ls /data/GR1/PhysicalAI-Robotics-GR00T-Teleop-Sim/HDF5/PnPCanToDrawerClose.hdf5   # 应显示该文件
ls /data/GR1/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/gr1_unified.PnPCanToDrawerClose   # 应显示该目录
```
两条命令都有输出 → 路径正确，后面把 `<DATA_ROOT>` 换成
`/data/GR1/PhysicalAI-Robotics-GR00T-Teleop-Sim` 即可（注意：**只换到 `...Teleop-Sim` 这一层**，
不要带上 `HDF5/` 或 `LeRobot/`）。

> 提示：为方便书写，建议在终端先执行：
> `export DATA_ROOT=/data/GR1/PhysicalAI-Robotics-GR00T-Teleop-Sim`
> 之后所有命令里的 `<DATA_ROOT>` 直接照抄即可（`$DATA_ROOT` 会自动展开成你的实际路径）。
> 训练时第 4 步的 `GR1_DIR` 则指向预处理输出目录：
> `GR1_DIR=/data/GR1/LeRobot-AugPosRot-Correct`

**如果你直接拿到了已预处理的 `Datasets/` 目录（含 `LeRobot-AugPosRot-Correct/`）**：
无需做 3.1-3.3 任何预处理，直接跳到第 4 步，把 `GR1_DIR` 指向：
```
GR1_DIR=/你的路径/Datasets/LeRobot-AugPosRot-Correct
```
训练命令中的 `--dataset-path <GR1_DIR>/gr1_unified.<TaskName>` 会自动对应到
`/你的路径/Datasets/LeRobot-AugPosRot-Correct/gr1_unified.<TaskName>`。
**验证**：`ls <你的路径>/Datasets/LeRobot-AugPosRot-Correct/ | wc -l` 应输出 `24`。

### 3.2 解算 EEF 位姿（每任务一条命令）
```bash
python preprocessing/extract_and_visualize_3d-pos_6d-rot_from_gr1.py \
    --dataset <DATA_ROOT>/HDF5/<TaskName>.hdf5 \
    --output_dir <DATA_ROOT>/Replay-Correct/<TaskName> \
    --render_image_names egoview \
    --verbose --render_height 800 --render_width 1280 \
    --num_parallel_jobs 10
```

### 3.3 增广出训练数据（每任务一条命令）
```bash
python preprocessing/aug_lerobot_data.py \
    --lerobot_base_path <DATA_ROOT>/LeRobot/gr1_unified.<TaskName> \
    --replay_base_path <DATA_ROOT>/Replay-Correct/<TaskName>/parquet/ \
    --output_base_path <DATA_ROOT>/LeRobot-AugPosRot-Correct/gr1_unified.<TaskName>
```

**24 个任务名**（可直接放入 for 循环批量处理）：
```bash
TASKS="PnPBottleToCabinetClose PnPCanToDrawerClose PnPCupToDrawerClose \
PnPMilkToMicrowaveClose PnPPotatoToMicrowaveClose PnPWineToCabinetClose \
PosttrainPnPNovelFromCuttingboardToBasketSplitA PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA \
PosttrainPnPNovelFromCuttingboardToPanSplitA PosttrainPnPNovelFromCuttingboardToPotSplitA \
PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA PosttrainPnPNovelFromPlacematToBasketSplitA \
PosttrainPnPNovelFromPlacematToBowlSplitA PosttrainPnPNovelFromPlacematToPlateSplitA \
PosttrainPnPNovelFromPlacematToTieredshelfSplitA PosttrainPnPNovelFromPlateToBowlSplitA \
PosttrainPnPNovelFromPlateToCardboardboxSplitA PosttrainPnPNovelFromPlateToPanSplitA \
PosttrainPnPNovelFromPlateToPlateSplitA PosttrainPnPNovelFromTrayToCardboardboxSplitA \
PosttrainPnPNovelFromTrayToPlateSplitA PosttrainPnPNovelFromTrayToPotSplitA \
PosttrainPnPNovelFromTrayToTieredbasketSplitA PosttrainPnPNovelFromTrayToTieredshelfSplitA"
```

**验证**：每个任务目录含 `meta/stats.json`、`data/chunk-000/episode_000000.parquet`、`videos/`。

### 3.4（可选）EgoDex 人类数据（论文场景 A 共训才需要）
按 `README.md` 第 132-159 行：EgoDex HDF5 → LeRobot v2.1 → v2.0，产出
`egodex_lerobot_gr00t/part2/basic_pick_place`。

---

## 第 4 步：修改脚本路径（10 分钟）

编辑 `examples/example_commands.sh` 顶部：
```bash
GR1_DIR=/path/to/LeRobot-AugPosRot-Correct      # ← 3.3 节输出目录
EGODEX_DIR=/path/to/egodex_lerobot_gr00t        # ← 仅场景 A 需要
```

---

## 第 5 步：训练（核心步骤）

### 5.1 先配置省显存环境变量（24GB 卡必设）
```bash
export GRADIENT_CHECKPOINTING=1
export MODEL_BF16=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 5.2 单卡冒烟测试（约 1 小时，先验证管线，强烈建议）
```bash
bash examples/train.sh decoupled \
  --dataset-path <GR1_DIR>/gr1_unified.PnPCanToDrawerClose \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 --data_split "[:20]" \
  --num-gpus 1 --batch-size 1 --gradient-accumulation-steps 32 \
  --dataloader-num-workers 4 --max-steps 100 --save-steps 100 \
  --base_model_path gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json \
  --compute-bridge-loss --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --report_to tensorboard --output_dir outputs/smoke-test
```
**验收**：日志出现 `100/100`，打印 `bridge_loss`/`action_loss`，生成 `outputs/smoke-test/checkpoint-100/`。若 OOM → 见第 8 节排查表。

### 5.3 Stage 1：8 卡预训练（decoupled / golden，80,000 步）
```bash
bash examples/train.sh decoupled \
  --dataset-path <GR1_DIR>/gr1_unified.PnPBottleToCabinetClose <GR1_DIR>/gr1_unified.PnPCanToDrawerClose ...（24 个任务全列） \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 --data_split "[:-10]" \
  --num-gpus 8 --batch-size 2 --gradient-accumulation-steps 16 \
  --dataloader-num-workers 4 --max-steps 80000 --save-steps 5000 \
  --base_model_path gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json \
  --compute-bridge-loss --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --report_to tensorboard --output_dir outputs/pretrain-decoupled-8gpu
```
> `--num-gpus 8` 时脚本自动执行 `torchrun --standalone --nproc_per_node=8 --nnodes=1`（仅支持单机 8 卡）。
> **有效 batch = 2×8×16 = 256，与论文一致。**

**监控**：
```bash
tensorboard --logdir outputs/pretrain-decoupled-8gpu/runs
# 或 tail -f 训练日志（建议用 nohup 后台运行：nohup bash examples/train.sh ... > train.log 2>&1 &）
```
**验收**：8 卡 GPU 利用率 90%+；每 5000 步生成 `checkpoint-5000`...`checkpoint-80000`。

### 5.4 Stage 2：8 卡微调（end2end，80,000 步）
```bash
bash examples/train.sh end2end \
  --dataset-path <GR1_DIR>/gr1_unified.PnPBottleToCabinetClose ...（24 个任务） \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 --data_split "[:-10]" \
  --num-gpus 8 --batch-size 2 --gradient-accumulation-steps 16 \
  --max-steps 80000 --save-steps 5000 \
  --base_model_path outputs/pretrain-decoupled-8gpu/checkpoint-80000 \
  --compute-bridge-loss --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --report_to tensorboard --output_dir outputs/finetune-end2end-8gpu
```

---

## 第 6 步：显存适配速查表

| GPU 显存 | bs/卡 | accum | 有效 batch | 备注 |
|---|---|---|---|---|
| ≥80GB（官方 A100/H100） | 32 | 1 | 256 | 论文原始配置 |
| 48GB | 4 | 8 | 128 | — |
| **24GB（本指南）** | **2** | **16** | **256** | 必须设 5.1 节 3 个环境变量 |
| 24GB OOM 兜底 | 1 | 32 | 256 | bs=1 最稳 |

---

## 第 7 步：评估（训练完成后）

```bash
# 离线评估（无需仿真器）：action MSE + bridge loss
bash examples/eval_loss.sh outputs/finetune-end2end-8gpu/checkpoint-80000

# 在线仿真评估（需第 2 步环境）：24 个 ID 任务成功率
bash examples/eval.sh outputs/finetune-end2end-8gpu/checkpoint-80000 id
```

---

## 第 8 步：常见问题排查

| 现象 | 解决 |
|---|---|
| HuggingFace 连不上 / 下载卡住 | `export HF_HUB_OFFLINE=1`（权重已缓存时）；或提前下载 Qwen2.5-VL-3B-Instruct 到 `~/.cache/huggingface/` |
| CUDA OOM | 确认 5.1 节 3 个环境变量已设置；仍 OOM 按第 6 节降 bs=1 |
| flash-attn 编译失败 | `pip install flash-attn --no-build-isolation` 或装预编译 wheel |
| 数据集报 `meta/stats.json` 缺失 | 3.2/3.3 未完成或路径指错 |
| loss 为 NaN | `--data_split "[:-10]"` 必须带引号；确认 `--compute-bridge-loss` 已加 |
| 多机 torchrun 报错 | 脚本仅支持单机 8 卡（`--nnodes=1` 硬编码） |
| 中途中断想续训 | 原命令加 `--resume`，`--base_model_path` 指向已有 checkpoint |
| 视频解码报错（非 GR1 数据） | 加 `--video-backend torchvision_av` |
| tensorboard 报 `pkg_resources` 缺失 | `pip install "setuptools<81"` |

---

## 关键路径速查

| 内容 | 位置 |
|---|---|
| 5 个完整场景命令模板 | `examples/example_commands.sh` |
| 训练入口（decoupled/end2end） | `examples/train.sh` |
| 训练主脚本与全部参数 | `scripts/dual_system_train.py` |
| 模型默认配置 | `gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json` |
| 数据预处理脚本 | `preprocessing/*.py` |
| 官方文档 | `README.md` |

---

**备注**：本指南以论文协议（预训练 80k + 微调 80k，24 任务全量数据）为准；24GB 卡按 bs=2/accum=16 达成论文级有效 batch 256。
