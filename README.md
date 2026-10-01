# DIAL02 — State-Future / MoT 增量版

本文档汇总本仓库相对 GitHub 原版 **`DIAL02-main`** 的**完整差异**（代码、配置、脚本、清理）。

- **基线**：`weibinlee2024/DIAL02` @ `986a7dd`（2026-09-08，即下载包 `DIAL02-main`）
- **本分支**：新增 `query_ca` 状态未来解码器、可选回灌 System-1、可选 MoT（`vis|sf|act`），并清理历史实验脚本
- 原 `docs/` 设计文档已删除，其内容已合并进本文档
- 复核命令：

```bash
git diff --stat 986a7dd HEAD          # 总览
git diff 986a7dd HEAD -- gr00t/       # 模型/数据代码差异
```

## 1. 差异总览

| 类型 | 数量 | 说明 |
|---|---:|---|
| 修改 Modified | 12 | 模型 / 数据 / 训练 / 脚本，合计 **+1020 / -114** 行 |
| 新增 Added | 12 | 本 README、`MANIFEST.txt`、`.gitignore` + 9 个脚本 |
| 删除 Deleted | 74 | 39 个历史脚本 + 35 个被纳入版本管理的 `.pyc` |

> 删除的 `docs/` 6 个文件相对基线属于"新增后又删除"，净差异为零，不列入上表。

## 2. 功能差异（相对基线）

### 2.1 System-2 状态未来：MLP mean-pool → `query_ca` 跨注意力解码器

- 基线：`state_future_projector`（MLP + mean-pool，常配 `D_sf=128`），仅作辅助损失。
- 现在：默认 `StateFutureDecoder`（`state_future_decoder=query_ca`，跨注意力查询未来帧），输出形状 `[B, 17, 30]`；`mean_pool` 路径仍可选。
- 预测视野 = `action_horizon`（17 帧）；`state_future_wrist_only=True` 时维度降为 18。
- SF 相关配置随 checkpoint 写入 `bridge_cfg`，`from_pretrained` 可自描述重建：
  `use_state_future`、`compute_state_future_loss`、`state_future_loss_weight`、`state_future_horizon`、
  `tune_state_future_projector`、`state_future_decoder`、`state_future_dim`、`state_future_num_heads`。

### 2.2 可选：把 Ŝ 回灌 System-1（`feed_state_future_to_dit`）

- 新增 `state_future_to_cond`（含 type embedding），把 17 个未来状态 token 追加到 bridge pack：
  `[goal | obs | sf]`。
- `sf_dit_repr` 可选 `shat`（预测未来状态）或 `mid_hidden`（解码器中间隐层）。

### 2.3 可选：MoT System-1（`vis | sf | act`）

- 新增 `MoTDiT`、`MoTTransformerBlock`、`MoTGlobalSelfAttention`：模态私有 QKV / WO / FFN / LN，共享全局自注意力。
- 由 `flow_matching_action_head_dial.py` 的 `_split_vis_sf`、`_build_mot_sequence` 组装序列。
- 默认 `use_mot_system1=False`，Dense DiT 路径与基线一致。

### 2.4 损失合成

```text
L = action_loss + bridge_w * bridge_loss + w * state_future_loss   # loss_combine_mode=additive（默认）
```

- 备选 `legacy_avg`；CLI 默认 `state_future_loss_weight=1.0`，对齐 / MoT 实验常用 `w=0.05`。
- SF 梯度默认不 detach，可回灌 decoder → bridge / VLM（`--tune-llm` 时）。

### 2.5 数据管线与 transforms

- `data_config_dial.py`：新增 `state_future` modality、`DEFAULT_STATE_FUTURE_KEYS`、`get_state_future_dim()` / `get_state_future_concat_order()`，按需构建 SF pipeline。
- `dataset.py`：生成 `state_future_time_mask`（轨迹尾部 pad 帧）。
- `transforms.py`：`_prepare_state_future` 输出 `state_future` / `state_future_mask`，帧 mask 与维度 mask 合并；缺少 SF key 时 fail-fast。
- `concat.py`：`torch.as_tensor` 兼容非 tensor 输入，避免 `.clone()` 崩溃。
- `vlm_backbone_dial.py`：打印 backbone 可训练参数量（`n_train / n_total`）。

### 2.6 训练 / 工程修复

- `trainer.py`：AdamW 默认 `foreach=False`，修复 48GB 显卡 + `--tune-llm` 时首步 `_foreach_sqrt` 大临时张量 OOM。
- `run_1gpu_finetune_sf_80k.sh`：SF 80k 微调改为 `--tune-llm`，`bs=2 × accum=32`（有效 batch 64），独立输出目录。

## 3. 修改文件明细（12）

| 文件 | 行数 | 变更 |
|---|---:|---|
| `gr00t/model/gr00t_n1_dial.py` | +433 / -60 | `StateFutureDecoder`（`query_ca`）、`state_future_to_cond` 回灌、MoT 接线、`loss_combine_mode`、SF 配置写入 ckpt |
| `gr00t/model/action_head/cross_attention_dit.py` | +253 / -0 | 新增 `MoTDiT` / `MoTTransformerBlock` / `MoTGlobalSelfAttention` |
| `gr00t/model/action_head/flow_matching_action_head_dial.py` | +113 / -15 | `_split_vis_sf`、`_build_mot_sequence`，SF token 加入 DiT 序列 |
| `gr00t/experiment/data_config_dial.py` | +77 / -12 | SF modality 定义、维度 / 拼接顺序、pipeline 构建 |
| `scripts/dual_system_train.py` | +42 / -1 | SF / MoT / feed 全量 CLI 接线与 data_config 开关 |
| `gr00t/model/transforms.py` | +39 / -16 | `_prepare_state_future`、`state_future(_mask)`、缺 key fail-fast |
| `gr00t/data/dataset.py` | +24 / -0 | `state_future_time_mask`（尾部 pad 帧） |
| `gr00t/data/transform/concat.py` | +13 / -3 | `torch.as_tensor` 健壮性修复 |
| `gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json` | +10 / -1 | 新增 SF / MoT 默认字段（见 §6） |
| `scripts/run_1gpu_finetune_sf_80k.sh` | +8 / -6 | SF 80k 微调切到 tune-llm |
| `gr00t/experiment/trainer.py` | +5 / -0 | AdamW `foreach=False`，修复 tune-llm 首步 OOM |
| `gr00t/model/backbone/vlm_backbone_dial.py` | +3 / -0 | 打印可训练参数规模 |

## 4. 新增文件明细（12）

| 文件 | 行数 | 用途 |
|---|---:|---|
| `README.md` | — | 本文档：与 `DIAL02-main` 的完整差异说明 |
| `MANIFEST.txt` | 25 | 打包快照文件清单 |
| `.gitignore` | 11 | 忽略 `__pycache__/`、`*.py[cod]`、`outputs/`、`checkpoints/`、`runs/`、`*.log` 等 |
| `scripts/verify_state_future_fixes.py` | +289 | CPU 校验：mask 合并、decoder 形状、配置写回与加载 |
| `scripts/smoke_feed_sf_to_dit.py` | +82 | 回灌冒烟：17 个 SF token 进入 bridge pack |
| `scripts/diag_sf_per_frame_and_probe.py` | +335 | 逐帧 SF 损失 + 冻结表征线性探针诊断 |
| `scripts/run_mot_vs_dense_aligned_2k.sh` | +131 | Dense+Ŝ vs MoT 对齐 2k 对照（`SMOKE_STEPS=5` 可冒烟） |
| `scripts/run_sf_dual_channel_2k.sh` | +94 | `feed_state_future_to_dit` 双通道 2k |
| `scripts/run_mot_aligned_20k_pipeline.sh` | +194 | 24-task MoT：golden 20k → end2end 10k |
| `scripts/run_24task_pretrain_golden_sf_tune_llm_20k.sh` | +90 | 24-task SF golden 20k（`--tune-llm`） |
| `scripts/run_24task_finetune_end2end_sf_tune_llm_test.sh` | +89 | 24-task SF end2end 10k 验证 |
| `scripts/run_24task_sf_tune_llm_pipeline.sh` | +52 | 上述两个 24-task 脚本的流水线封装 |

## 5. 删除文件明细（74）

### 5.1 历史脚本（39 个）

| 类别 | 数量 | 代表文件 |
|---|---:|---|
| 4GPU / 4task | 6 | `run_4task_*_4gpu.sh`、`run_4task_*_end2end.sh`、`run_4task_pretrain_golden.sh` |
| 多机 / DDP 调试 | 12 | `run_2card_pretrain_*.sh`、`run_local/remote_{ddp,pretrain_ddp}*.sh`、`torchrun_smoke_*.sh`、`diag_nosync_*`、`diag2.py` |
| 中继 / 监控 / 状态 | 5 | `relay_eval.sh`、`relay_single.sh`、`status_repro.sh`、`monitor_libero_train.{py,sh}` |
| 一次性排查 / 修复 | 5 | `check_eval_tasks.py`、`check_ood_tasks.py`、`test_robocasa_{env,viz}.py`、`fix_episode_701.py` |
| 被取代的 1GPU | 9 | `run_1gpu_{finetune,pretrain}_{sim,single}*.sh`、`run_1gpu_{smoke_train,libero_*}.sh`、`env_1gpu.sh` |
| 其它 | 2 | `test_mem_bs2.sh`、`watch_sf.sh`（写死本机路径） |

<details>
<summary>展开完整 39 个删除脚本</summary>

```text
scripts/check_eval_tasks.py
scripts/check_ood_tasks.py
scripts/diag2.py
scripts/diag_nosync.py
scripts/diag_nosync_local.sh
scripts/diag_nosync_remote.sh
scripts/env_1gpu.sh
scripts/fix_episode_701.py
scripts/monitor_libero_train.py
scripts/monitor_libero_train.sh
scripts/relay_eval.sh
scripts/relay_single.sh
scripts/run_1gpu_finetune_sim.sh
scripts/run_1gpu_finetune_single.sh
scripts/run_1gpu_libero_pretrain.sh
scripts/run_1gpu_libero_smoke_train.sh
scripts/run_1gpu_pretrain_sim.sh
scripts/run_1gpu_pretrain_single.sh
scripts/run_1gpu_pretrain_single_80k.sh
scripts/run_1gpu_smoke_train.sh
scripts/run_2card_pretrain_local.sh
scripts/run_2card_pretrain_remote.sh
scripts/run_4task_finetune_end2end.sh
scripts/run_4task_finetune_end2end_4gpu.sh
scripts/run_4task_finetune_end2end_sf_4gpu.sh
scripts/run_4task_pretrain_golden.sh
scripts/run_4task_pretrain_golden_4gpu.sh
scripts/run_4task_pretrain_golden_sf_4gpu.sh
scripts/run_local_ddp0.sh
scripts/run_local_pretrain_ddp.sh
scripts/run_remote_ddp1.sh
scripts/run_remote_pretrain_ddp.sh
scripts/status_repro.sh
scripts/test_mem_bs2.sh
scripts/test_robocasa_env.py
scripts/test_robocasa_viz.py
scripts/torchrun_smoke_local.sh
scripts/torchrun_smoke_remote.sh
scripts/watch_sf.sh
```

</details>

### 5.2 版本库清理

- 删除 35 个被基线纳入版本管理的 `gr00t/**/__pycache__/*.pyc`。
- 新增根目录 `.gitignore`，避免再次提交字节码 / 训练产物。

## 6. CLI 与配置新增项

`gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json` 与 `dual_system_train.py` 新增：

| 开关 | 默认 | 含义 |
|---|---|---|
| `use_state_future` / `--use-state-future` | `false` | 启用状态未来监督 |
| `compute_state_future_loss` / `--compute-state-future-loss` | `false` | 计算 SF 损失 |
| `state_future_loss_weight` | `1.0` | SF 损失权重（对齐 / MoT 实验常用 `0.05`） |
| `state_future_horizon` | `null` → `action_horizon`(17) | 预测帧数 |
| `state_future_decoder` | `query_ca` | `query_ca` \| `mean_pool` |
| `state_future_dim` | `30`（wrist-only 为 18） | SF 标签维度 |
| `state_future_num_heads` | `8` | query_ca 注意力头数 |
| `state_future_wrist_only` / `--state-future-wrist-only` | `false` | 仅监督手腕状态 |
| `tune_state_future_projector` | `true` | decoder / 投影参与训练 |
| `loss_combine_mode` | `additive` | `additive` \| `legacy_avg` |
| `feed_state_future_to_dit` / `--feed-state-future-to-dit` | `false` | Ŝ 回灌 System-1 |
| `sf_dit_repr` | `shat` | `shat` \| `mid_hidden` |
| `use_mot_system1` / `--use-mot-system1` | `false` | Dense → MoT |
| `mot_include_sf` / `--mot-include-sf` | `true` | MoT 是否包含 SF 模态 |
| `mot_sf_num_tokens` / `--mot-sf-num-tokens` | `17` | SF token 数 |
| `mot_num_layers` / `--mot-num-layers` | `null` | MoT 层数（默认沿用 DiT 配置） |

## 7. 验证与训练入口

```bash
# CPU 校验（mask / decoder / 配置写回）
python scripts/verify_state_future_fixes.py

# 回灌冒烟（17 个 SF token 进入 bridge pack）
python scripts/smoke_feed_sf_to_dit.py

# 短训冒烟
SMOKE_STEPS=5 bash scripts/run_sf_dual_channel_2k.sh
SMOKE_STEPS=5 RUN_WHICH=mot bash scripts/run_mot_vs_dense_aligned_2k.sh

# 正式训练
bash scripts/run_24task_sf_tune_llm_pipeline.sh     # 24-task SF: 20k → 10k
bash scripts/run_mot_aligned_20k_pipeline.sh        # 24-task MoT: 20k → 10k
bash scripts/run_1gpu_pretrain_sf_80k.sh            # SF 80k 预训练
bash scripts/run_1gpu_finetune_sf_80k.sh            # SF 80k 微调（tune-llm）
```

## 8. 目录与脚本现状

保留的 `scripts/` 共 27 个文件，按用途：

- **核心训练 / 评测**：`dual_system_train.py`、`eval_policy_dial.py`、`inference_service_dial.py`、`simulation_service_dial_augPosRot.py`、`smoke_load_model.sh`、`compute_success_rate.py`
- **SF / MoT 实验**：`run_24task_*`、`run_mot_*`、`run_sf_dual_channel_2k.sh`、`run_1gpu_{pre,finetune}_sf_80k.sh`、`diag_sf_per_frame_and_probe.py`、`smoke_feed_sf_to_dit.py`、`verify_state_future_fixes.py`
- **数据 / 环境**：`install_dial_env.sh`、`setup_local_paths.sh`、`download_checkpoints.sh`、`download_gr1_raw.sh`、`preprocess_gr1_*.sh`、`preprocess_libero_dense_*.sh`
