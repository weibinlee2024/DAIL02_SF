# branch_probe_20261028

老师指定实验（同 seed / 同候选池 / score vs random / 同状态 K 分支）相关的**修改代码快照**，日期 2026-10-28。

> 说明：本目录是便于审阅/转发的拷贝。机器上实际跑的是  
> `DIAL02_SF20161018/` 与 `bon_attribution_4arm/` 下的同名文件（内容一致）。

---

## 目录

```
branch_probe_20261028/
├── README.md
├── gr00t/model/
│   ├── policy_dial.py                      # 完整 DialPolicy（含扩展）
│   └── propose_bon_with_meta_snippet.py     # 仅 propose_bon_with_meta 片段，方便快读
├── scripts/
│   └── probe_bon_branch_consistency.py     # teacher 协议主脚本（重写）
└── launch/
    ├── run_teacher_probe_potato_milk.sh    # Potato → Milk 启动队列
    └── wait_24task_then_rerun_branch_probe.sh
```

---

## 各文件改了什么

### 1. `gr00t/model/policy_dial.py`（扩展 `propose_bon_with_meta`）

- 相同候选噪声：`seed_base + 10007*(i+1)`
- 同池：一次采样返回全部 K 个 `actions_*`
- `best` = score 选；`random_pick = RandomState(seed_base+17)`（独立于分数）
- 返回 `s_t` / `s_hat` / `pred_sf`、`scores_pos/rot`、`dist_to_mean`、`smoothness`、`cand_dispersion`

### 2. `scripts/probe_bon_branch_consistency.py`（teacher 主实验）

- 固定 `episode_seed = seed_base + ep`
- Pass1 score：同状态 save/restore，分支执行 K 候选 + 存 pool
- Pass2 random：同 seed、回放 pool，只 commit `random_pick`
- 检查：dynamics、SF vs 进展、score vs oracle（pairwise/Spearman/Kendall）、grasp/transport/placement/close、平滑/均值捷径
- 输出：`bon_teacher_probe_*.json`、`bon_teacher_pools_*.json`
- 目标：解释离线 pairwise≈随机、Spearman≈0，而在线 score−random = +21.5pp

### 3. `launch/*.sh`

- 用对齐仿真 `DIAL-data/DIAL_TOOLS` 跑 Potato/Milk 各 **50 ep**（与 attr4/+21.5 协议对齐；`seed_base=20261028`, K=4, N=8）

---

## 正式评测 episode seed（已与 probe 对齐）

`gr00t/eval/simulation_augPosRot.py`：每个 episode  
`env.reset(seed=seed_base + episode_id)`，默认 `seed_base=20261028`。  
`scripts/simulation_service_dial_augPosRot.py --seed_base`、`examples/eval.sh` 的 `EVAL_SEED_BASE` 已接通。

未改：inference server 的 score 选择逻辑、WM/策略权重。

---

## 运行产物（不在本目录）

`/media/weibin/weibin01/DIAL_SF_20261018/bon_attribution_4arm/branch_probe/teacher_*/`
