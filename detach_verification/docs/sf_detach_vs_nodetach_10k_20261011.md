# detach vs 不 detach 总体总结（mean_pool / shat / 10k）

> 实验日期：2026-10-07 ~ 2026-10-10；总结日期：2026-10-11  
> 机制示意见 [`sf_tokens_detach_schematic_20261010.md`](sf_tokens_detach_schematic_20261010.md)；代码入口 [`gr00t/model/gr00t_n1_dial.py`](../gr00t/model/gr00t_n1_dial.py)（约 612–638 行 detach，644–674 行 SF 监督）。

---

## 1. 实验设置

两个 run 除 `sf_tokens_detach` 外完全相同：

| 项 | 值 |
|---|---|
| 任务 | 6 × `PosttrainPnPNovel...SplitA`（Placemat/…/Tray→Plate） |
| data config | `fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop` |
| bridge | `golden`，`select_layer=12`，`tune_llm=True`，`tune_visual=False` |
| SF 通路 | `use_state_future` + `compute_state_future_loss`，`decoder=mean_pool`，`dim=30`，`H=16`（+frame0=17） |
| SF→DiT | `feed_state_future_to_dit=True`，`sf_dit_repr=shat` |
| 损失 | `loss_combine_mode=additive`，`L = action + bridge + 0.05·sf` |
| 优化 | bs 2 × grad-accum 32，1×RTX 4090 48G，bf16 + gradient checkpointing，lr 1e-4 cosine，warmup 0.05，10k steps |
| 差异 | `nodetach: sf_tokens_detach=False`；`detach: sf_tokens_detach=True` |

- `nodetach`：`/media/weibin/weibin02/DIAL_MoT/sf_dual_channel_2k/mean_pool_shat_w0p05/`
- `detach`：`/media/weibin/weibin02/DIAL_MoT/sf_dual_channel_2k/mean_pool_shat_w0p05_detach/`
- 启动脚本：`scripts/run_sf_dual_channel_2k.sh`（`SF_TOKENS_DETACH=0/1`），中途 resume 到 10k。

---

## 2. 结论速览

1. **梯度路由验证通过**：detach 只切断 `action_loss → DiT → state_future_to_cond → Ŝ → SF 头` 这一条；前向数值、SF 自身监督、`state_future_to_cond` 的梯度都不受影响（详见 §3）。
2. **训练曲线几乎重合**：10k 时总 loss 0.7060 vs 0.7059，action 0.0835 vs 0.0823，bridge 0.5979 vs 0.5996；只有 `state_future_loss` detach 略低（0.0499 vs 0.0681）。
3. **SF 头质量 detach 略好**：probe（decoder 预测 vs GT）MSE 在 2k/5k/10k 都是 detach 更低（10k：0.0695 vs 0.0801），说明 action 回传对 SF 头有轻微扰动。
4. **Action 离线指标打平**：5k 时 detach 好 7.4%（0.1804 vs 0.1949），10k 时基本持平（0.1283 vs 0.1266，+1.4%）。
5. **线性探针方向不一致**：mean-pool 特征对 510 维 SF 的线性 R² 两边都仍为负，10k 时 nodetach 更好（-0.130 vs -0.224）。
6. **尚不能下"SF 无用/有用"结论**：golden 设置下动作头还能看到真实未来图，SF 通道贡献被稀释；且只有离线 action MSE、单 seed、无闭环成功率。

---

## 3. 机制与梯度验证（受控诊断，非训练曲线）

同一 checkpoint（detach 的 `checkpoint-5000`）、同一真实 batch，只切换 detach，报告见  
`diag_mean_pool_detach_ckpt5000_20261010_215745/sf_real_batch_diag.json`（脚本 `scripts/sf_real_batch_diag.py`）。

梯度只统计 SF 头（`state_future_projector`）与 `state_future_to_cond`：

| 量 | 不 detach | detach | 含义 |
|---|---|---|---|
| `action_to_decoder.norm` | **0.0360**（4/4 连着） | **0（0/4 断开）** | detach 生效 |
| `action_to_cond.norm` | 0.0195（4/4） | 0.0195（4/4） | cond 投影仍被 action 训练 |
| `sf_to_decoder.norm` | 0.6582（4/4） | 0.6582（4/4） | SF 监督两边都不断 |
| `action / (0.05×sf_to_decoder)` | **1.095** | — | 该 batch 上 action 回传与加权 SF 梯度同量级 |
| 前向预测最大差 | 0.0 | 0.0 | detach 不改数值 |
| `checks` / `gradient_routing_checks_passed` | 全部 True | | |

另一组独立对照（同一 checkpoint、同一 batch）：

```
detach=False: action_loss -> projector grad_norm = 3.39e-02
detach=True : action_loss -> projector grad_norm = 0.00e+00
两者        : state_future_loss -> projector grad_norm = 5.09e-01（完全一致）
两者        : action_loss -> state_future_to_cond grad_norm ≈ 7e-02（都非零）
```

要点：`action/(0.05·sf)` 比值随 batch 波动（0.07~1.1 都出现过），说明 action 回传强度不是恒定的小量，但方向与 SF 监督是否冲突无法由范数判断。

---

## 4. 训练 loss 曲线（tensorboard，按 step 内微批平均）

| step | loss nd / det | action nd / det | sf nd / det | bridge nd / det | grad_norm nd / det |
|---|---|---|---|---|---|
| 2000 | 1.6754 / 1.6729 | 0.9810 / 0.9759 | 0.1285 / 0.1074 | 0.6761 / 0.6765 | 0.660 / 0.610 |
| 5000 | 0.7686 / 0.7696 | 0.1351 / 0.1366 | 0.0590 / 0.0607 | 0.6733 / 0.6742 | 0.950 / 0.992 |
| 10000 | 0.7060 / 0.7059 | 0.0835 / 0.0823 | 0.0681 / 0.0499 | 0.5979 / 0.5996 | 0.795 / 0.887 |

观察：

- 总 loss / action / bridge 两条曲线基本重合，差异小于正常微批噪声。
- `state_future_loss` 全程 detach 略低（10k 低约 27%），与 SF 头不被 action 梯度"拉扯"一致。
- 单 step 内微批波动很大（sf 可 0.02~0.30），看趋势必须用平滑或按 step 平均。

数据源：`.../mean_pool_shat_w0p05/runs/*/events*`、`.../mean_pool_shat_w0p05_detach/runs/*/events*`。

---

## 5. SF 预测质量（probe）

`decoder_overall_mse_vs_gt`（预测未来状态 vs GT，越低越好）：

| checkpoint | nodetach | detach |
|---|---|---|
| 2k | 0.1410 | 0.1269 |
| 5k | 0.1027 | 0.0911 |
| 10k | 0.0801 | **0.0695** |

10k per-frame（nodetach / detach）：

| 分组 | nodetach | detach |
|---|---|---|
| frame0（当前帧） | 0.0550 | 0.0442 |
| future 1..H（未来帧） | 0.0582 | 0.0468 |
| near 1..4 | 0.0538 | 0.0429 |
| far last 4 | 0.0657 | 0.0525 |

线性探针（bridge mean-pool 特征 → 510 维 SF，`baseline_mean_r2=-0.016`）：

| checkpoint | nodetach R² | detach R² |
|---|---|---|
| 2k | -0.549 | -0.775 |
| 5k | -0.471 | -0.396 |
| 10k | **-0.130** | -0.224 |

要点：decoder MSE 一直下降且 future≈frame0（外推没有明显退化），detach 的 decoder 指标更好；但线性可解码性 10k 反而是 nodetach 更好。两个指标看的是不同东西（生成精度 vs 特征的线性可分性）。

路径：`checkpoint-*/sf_probe_r2/sf_per_frame_and_probe.json`。

---

## 6. 离线 action 评测（6 任务平均 action MSE）

| checkpoint | nodetach | detach | detach − nodetach |
|---|---|---|---|
| 2k | 0.808 | 0.823 | +1.9% |
| 5k | 0.1949 | **0.1804** | **−7.4%** |
| 10k | **0.1266** | 0.1283 | +1.4% |

bridge_loss（eval）：nodetach 0.7327 / 0.6976 / 0.6789，detach 0.7335 / 0.6973 / 0.6792，基本一致。

汇总 JSON：`.../compare_mean_pool_detach_vs_nodetach_10k.json`；原始：`checkpoint-*/eval_action_goal_loss/results.json`。

---

## 7. 如何解读

- **为什么曲线几乎一致**：detach 不改前向、不改 SF 监督，也共享全部主干/视觉梯度；只去掉一条 action→SF 头的回传。SF 头本身仍主要由 `state_future_loss`（w=0.05）塑形，所以两条训练轨迹高度接近是预期的。
- **detach 的 SF 头更好**：action 回传相当于一个额外任务在改 SF 表征，detach 后 SF 头更"纯"，decoder MSE 更低（10k 0.0695 vs 0.0801）。
- **action 指标没有稳定趋势**：5k detach 优、10k 打平，说明在当前 golden 设置下，是否 detach 对动作头影响在噪声范围内。
- **探针 R² 仍为负**：mean-pool 特征对 SF 的线性可解码性弱，但 10k 改善明显（-0.55→-0.13）；这不等于 SF 没信息，只是线性探针弱。

---

## 8. 建议的后续实验

1. **闭环评测**：离线 action MSE 不能替代成功率，建议对 5k/10k 两个 checkpoint 做闭环对比。
2. **换到 end2end（去掉 golden 未来图）再看 detach 差异**：golden 下动作头有真实未来图，SF 通道的贡献被稀释，detach 的影响自然小。
3. **若想放大 action→SF 梯度差异**：降低 `state_future_loss_weight`（如 0.01）或短暂冻结 SF 头只留 action 回传，观察 RF 头的偏移方向（是否与 SF 监督冲突）。
4. **统一结论口径**：目前 5k/10k 两个点的离线指标互有胜负，不宜只拿单点下结论；至少多 seed 或多种子平均。

---

## 9. 复现命令

训练（沿用原脚本，只差一个环境变量）：

```bash
SF_TOKENS_DETACH=0 bash scripts/run_sf_dual_channel_2k.sh   # nodetach
SF_TOKENS_DETACH=1 bash scripts/run_sf_dual_channel_2k.sh   # detach
```

梯度路由诊断（不训练，详见 `DIAL02-detach-diagnostics-20261009/README.md`）：

```bash
CHECKPOINT=/media/weibin/weibin02/DIAL_MoT/sf_dual_channel_2k/mean_pool_shat_w0p05_detach/checkpoint-5000
DIAG_OUT=$(mktemp -d /tmp/dial-sf-diag.XXXXXX)
DIAL_SF_DIAG=1 DIAL_SF_DIAG_INDICES=0,1 BASE_CFG="$CHECKPOINT" \
SF_DECODER=mean_pool SF_DIT_REPR=shat SF_DUAL_W=0.05 \
SF_TOKENS_DETACH=0 P_SF_DROP=0 P_SF_SWAP=0 \
NUM_GPUS=1 BATCH_SIZE=2 RUN_PROBE=0 \
SF_DUAL_OUT="$DIAG_OUT" SF_DUAL_LOG="$DIAG_OUT/run.log" \
bash scripts/run_sf_dual_channel_2k.sh
cat "$DIAG_OUT/sf_real_batch_diag.json"
```

---

## 附：一句话总结

detach 在代码和梯度层面已验证正确；两个 run 的训练/action 指标几乎重合，detach 让 SF 头拟合更好，但在 golden 设置下对 action 的影响在噪声内。要判断 SF 对未来决策是否有用，下一步应做闭环与 end2end 对照，而不是继续看训练 loss。
