# DIAL-SF-WM: State-Future and World-Model Extensions for DIAL

Also referred to as **DIAL-SF-WM**.

[Public repository](https://github.com/weibinlee2024/DAIL02_SF) · Research preview

DIAL02-SF is a research fork of **DIAL02**, built on **GR00T N1.5** and a dual-system robot policy. It extends the original DIAL pipeline with:

- a **State-Future (SF)** predictor for future end-effector states;
- an optional **SF-to-System-1** feedback path;
- an optional **Mixture-of-Transformers (MoT)** System-1;
- a **bridge / world-model** branch that learns from future visual goals;
- recipes for GR1/RoboCasa-style tabletop manipulation and an experimental LIBERO-plus pipeline.

> **Status:** research code, not a production library and not an official NVIDIA or DIAL release. Interfaces, scripts, and defaults may change. Reproduction requires a local GPU environment and external datasets/checkpoints.

---

## Contents

- [What This Repository Provides](#what-this-repository-provides)
- [Architecture](#architecture)
- [Status](#status)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints](#checkpoints)
- [Configuration Reference](#configuration-reference)
- [Verification and Smoke Tests](#verification-and-smoke-tests)
- [Relationship to DIAL02](#relationship-to-dial02)
- [Troubleshooting](#troubleshooting)
- [License and Acknowledgements](#license-and-acknowledgements)
- [Contact](#contact)

---

## What This Repository Provides

DIAL02-SF keeps the original DIAL dual-system design:

- **System 2:** a VLM backbone that encodes images, language, and proprioception, and produces bridge tokens.
- **System 1:** a diffusion/flow-matching action expert that predicts an action chunk.
- **Bridge / world model:** a future-image latent objective that trains the bridge representation.

This fork adds three research directions:

1. **State-Future prediction in System 2**
   - `query_ca`: a cross-attention decoder that predicts a sequence of future states.
   - `mean_pool`: the earlier MLP/mean-pool path, retained for checkpoint compatibility.
   - SF parameters are written into the checkpoint configuration, so `from_pretrained` can reconstruct the model without manual flags.

2. **Optional SF feedback to System 1**
   - `feed_state_future_to_dit=True` appends predicted future-state tokens to the System-1 bridge pack.
   - `sf_dit_repr` selects the predicted state (`shat`) or a decoder hidden representation (`mid_hidden`).

3. **Optional MoT System 1**
   - `use_mot_system1=True` switches the dense DiT action expert to a Mixture-of-Transformers variant with modality-specific QKV, output projections, FFNs, and layer norms.
   - The default is still the dense DiT path.

The repository also contains the engineering work needed to run these variants: data transforms, checkpoint/config plumbing, training entry points, evaluation services, and smoke tests.

---

## Architecture

```text
images (front / wrist), language, proprioception
                         |
                System-2 VLM backbone
                  /                 \
          bridge tokens        State-Future decoder
                  |             (query_ca / mean_pool)
                  |                    |
                  |          optional SF-to-System-1
                  |                    |
                  +--------> System-1 action expert
                                     |
                              action chunk
```

### Dual-system policy

- The VLM backbone processes visual observations and language instructions.
- The bridge branch learns a latent representation aligned with a future goal frame.
- The action expert consumes bridge tokens and predicts a chunk of robot actions.
- The action expert can be a dense DiT or the optional MoT DiT.

### State-Future head

- Predicts future end-effector state over the action horizon.
- Supports full-state and wrist-only variants.
- Uses frame and dimension masks for padded trajectory tails.
- Can be trained as an auxiliary loss or fed back into System 1.

### MoT option

- `MoTDiT`, `MoTTransformerBlock`, and `MoTGlobalSelfAttention` are implemented in the action-head code.
- Modalities are `vis`, `sf`, and `act`.
- The dense path remains the reference configuration.

### Training objective

```text
L = action_loss + bridge_weight * bridge_loss + sf_weight * state_future_loss
```

- `loss_combine_mode=additive` is the default.
- `legacy_avg` is retained for comparison.
- The bridge loss can be disabled for action-only fine-tuning.
- The SF loss is optional and is usually enabled only for SF experiments.

---

## Status

| Area | Status |
|---|---|
| 24-task GR1/RoboCasa-style training pipeline | Available; research recipe |
| Dense System-1 + bridge / world-model | Available |
| State-Future with `query_ca` or `mean_pool` | Available |
| SF-to-System-1 feedback | Available; optional |
| MoT System-1 | Available; experimental |
| GR1 simulation evaluation | Available; environment setup required |
| LIBERO-plus integration | Experimental; not a supported benchmark yet |
| Public pretrained checkpoints | Download script provided for released DIAL checkpoints |
| Datasets | Not included; download/preprocess scripts provided |

### Reference result (internal, not a controlled benchmark)

A 36-layer mean-pool SF 20k recipe reached approximately **61.8%** success in internal GR1 simulation logs. This number is a reference point for reproduction, not a claim of state-of-the-art performance. Re-run the scripts in your environment before citing it.

### LIBERO-plus note

The LIBERO-plus branch is included for transparency and continued research. Current transfer pilots from the GR1/RoboCasa 20k checkpoint are not reliable; the action head and action-interface mismatch are under investigation. See `DIAL-LIBERO-PLUS/` for code links, interface checks, and the failure analysis. Do not treat current LIBERO-plus numbers as benchmark results.

---

## Repository Layout

```text
.
├── gr00t/
│   ├── model/                 # VLM backbone, DiT/MoT action heads, policy, transforms
│   ├── experiment/            # training loop, data configs, trainer
│   ├── data/                  # LeRobot datasets, transforms, embodiment tags
│   ├── eval/                  # observation/action adapters
│   └── rlvr/                  # optional RLVR utilities
├── scripts/
│   ├── dual_system_train.py   # main training entry point
│   ├── inference_service_dial.py
│   ├── simulation_service_dial_augPosRot.py
│   ├── eval_policy_dial.py
│   ├── compute_success_rate.py
│   ├── run_24task_*.sh        # 24-task training recipes
│   ├── run_mot_*.sh           # MoT recipes
│   ├── run_libero_*.sh        # LIBERO-plus experimental pipeline
│   └── preprocess_*.sh        # dataset preparation
├── preprocessing/             # HDF5/LeRobot preprocessing utilities
├── examples/                  # shell entry points and task lists
├── detach_verification/       # SF gradient-detach ablations and patches
├── DIAL-LIBERO-PLUS/          # LIBERO-plus documentation and code snapshot
├── docs/                      # design notes
└── launch/ probe/ rlvr/       # experiment utilities on the public branch
```

`outputs/`, `runs/`, checkpoints, raw datasets, and generated logs are intentionally not part of the source tree. The provided scripts create them under configurable external roots.

---

## Installation

### Requirements

- Linux
- NVIDIA GPU; 48 GB is recommended for full `--tune-llm` training
- CUDA 12.1-compatible PyTorch (the setup script uses PyTorch 2.5.1 / torchvision 0.20.1)
- Conda or Miniconda
- Python 3.10 (recommended)
- Optional: FlashAttention for faster training
- Optional: a second environment for LIBERO-plus evaluation

### Setup

```bash
git clone https://github.com/weibinlee2024/DAIL02_SF.git
cd DAIL02_SF

conda create -n dial python=3.10 -y
conda activate dial

# The installer assumes the conda environment is already named "dial".
bash scripts/install_dial_env.sh

# This repository is used as a source tree.
export PYTHONPATH="$PWD:$PYTHONPATH"
```

If the installer's editable-install step is unavailable in your checkout, skip it and keep the source tree on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
python -c "import gr00t; print(gr00t.__file__)"
```

FlashAttention is optional. If it is not available, use the non-flash-attention configuration supported by the model.

### External storage

The provided helper links `data/`, `checkpoints/`, and `outputs/` to an external directory:

```bash
export DIAL_EXT_ROOT=/path/to/large/disk
bash scripts/setup_local_paths.sh
```

The helper requires those target names not to exist as regular directories. Move existing data aside before linking.

---

## Data Preparation

Datasets and pretrained checkpoints are not redistributed in this repository.

### GR1 / RoboCasa-style teleoperation data

```bash
# Download the NVIDIA PhysicalAI GR00T Teleop Sim package.
bash scripts/download_gr1_raw.sh

# Preprocess the 24-task GR1 dataset.
bash scripts/preprocess_gr1_all_tasks.sh
```

The raw package is large. The download script writes to `data/raw/` (or the external root configured by `setup_local_paths.sh`).

The 24-task training scripts expect a list of LeRobot task directories. Review each script and set the dataset root/path variables for your machine. For example:

```bash
DATA_ROOT=/path/to/lerobot/tasks
OUTPUT_DIR=/path/to/outputs/24task_sf
```

### LIBERO-plus data (experimental)

The LIBERO-plus pipeline uses LeRobot datasets and the `libero-plus` environment:

```bash
bash scripts/preprocess_libero_dense_all.sh
python scripts/prepare_libero_plus_lerobot.py --help
python scripts/build_libero_plus_heldout_manifest.py --help
```

LIBERO-plus videos are AV1 encoded. Use the `torchvision_av` backend (or another AV1-capable decoder); `decord` does not handle these files.

### Expected data conventions

- LeRobot v2.1-style datasets with images, language annotations, state, and action keys.
- Data configs live in `gr00t/experiment/data_config_dial.py`.
- Normalization statistics must come from the target dataset. Do not reuse GR1 metadata for a new embodiment.
- Keep `action_horizon` consistent with the checkpoint unless you intentionally rebuild the action head.

---

## Training

### Two-stage bridge recipe

The main recipe follows the DIAL two-stage pattern:

1. **Golden / decoupled stage**: learn the bridge/world-model representation.
2. **End-to-end stage**: fine-tune the full policy with the action loss and optional bridge/SF losses.

Common wrappers:

```bash
# Stage 1: 24-task SF golden 20k
bash scripts/run_24task_pretrain_golden_sf_tune_llm_20k.sh

# Stage 2: 24-task SF end-to-end 10k
bash scripts/run_24task_finetune_end2end_sf_tune_llm_test.sh

# Both stages
bash scripts/run_24task_sf_tune_llm_pipeline.sh
```

MoT recipe:

```bash
bash scripts/run_mot_aligned_20k_pipeline.sh
```

Single-GPU SF recipes:

```bash
bash scripts/run_1gpu_pretrain_sf_80k.sh
bash scripts/run_1gpu_finetune_sf_80k.sh
```

The generic wrapper maps the stage name to `--bridge_type`:

```bash
bash examples/train.sh decoupled [training arguments]
bash examples/train.sh end2end  [training arguments]
```

### Minimal direct training command

```bash
python -u scripts/dual_system_train.py \
  --bridge_type golden \
  --dataset-path /path/to/lerobot/task_a /path/to/lerobot/task_b \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop_sf128 \
  --embodiment_tag gr1 \
  --base_model_path /path/to/base/checkpoint \
  --output_dir /path/to/output \
  --num-gpus 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 32 \
  --max-steps 20000 \
  --save-steps 1000 \
  --tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --select_layer 36 \
  --use-state-future \
  --compute-state-future-loss \
  --state-future-decoder mean_pool \
  --state-future-dim 128 \
  --state-future-horizon 16 \
  --state-future-loss-weight 0.5 \
  --compute-bridge-loss \
  --use-separate-projector-for-loss
```

The exact task list, base checkpoint, and output paths are machine-specific. Review the shell recipes before launching; they are intended as reproducible starting points rather than portable one-click scripts.

### Effective batch size

`--batch-size` and `--gradient-accumulation-steps` determine the effective batch:

```text
effective_batch = batch_size * gradient_accumulation_steps * num_gpus
```

The 24-task recipes typically use an effective batch of 64. When comparing runs, compare effective samples/steps, not only the nominal step count.

### Memory guidance

For a 48 GB GPU with full `--tune-llm` training:

- use `MODEL_BF16=1` when the recipe supports it;
- keep `--no-tune-visual`;
- reduce `batch_size` and increase gradient accumulation;
- enable gradient checkpointing;
- do not enable extra losses until the baseline action-only run is stable.

---

## Evaluation

### In-distribution and OOD GR1 simulation

`examples/eval.sh` launches the policy server and the simulation client for the GR1 task sets:

```bash
bash examples/eval.sh /path/to/checkpoint id
bash examples/eval.sh /path/to/checkpoint ood_object_appearance
bash examples/eval.sh /path/to/checkpoint ood_container_combination
bash examples/eval.sh /path/to/checkpoint ood_object_type
```

The script uses local RoboCasa/robosuite paths and a local conda environment. Update those paths for your machine before running.

Useful components:

```bash
# Policy server only
python scripts/inference_service_dial.py \
  --server \
  --model_path /path/to/checkpoint \
  --embodiment_tag gr1 \
  --data_config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --port 5555

# Offline dataset evaluation
python scripts/eval_policy_dial.py \
  --dataset-path /path/to/lerobot/tasks \
  --model-path /path/to/checkpoint \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment-tag gr1 \
  --video-backend torchvision_av

# Aggregate success rates
python scripts/compute_success_rate.py --help
```

### LIBERO-plus (experimental)

The manifest-based evaluation path is:

```bash
MODEL_PATH=/path/to/checkpoint \
MANIFEST=/path/to/pilot_v1.json \
OUT=/path/to/eval_output \
PORT=5582 \
CUDA_DEVICE=1 \
  bash scripts/run_libero_plus_manifest_eval.sh
```

Use the `libero-plus` Python environment and OSMesa for headless rendering. The current LIBERO-plus transfer results are experimental; the `DIAL-LIBERO-PLUS/` package contains the interface checks, pilot results, and failure analysis.

---

## Checkpoints

Download the released DIAL checkpoints:

```bash
bash scripts/download_checkpoints.sh fewshot
bash scripts/download_checkpoints.sh fulldata
bash scripts/download_checkpoints.sh both
```

The script uses `xpeng-robotics/DIAL_checkpoints` on Hugging Face and writes under `checkpoints/`.

Checkpoint loading notes:

- The checkpoint configuration is authoritative for `select_layer`, action horizon, SF decoder, and embodiment settings.
- Do not force a different `select_layer` from the one used during pretraining.
- `bridge_goal_model.*` may be absent from shared-bridge checkpoints by design; this is expected for inference-only use.
- Do not enable bridge loss with a randomly initialized goal encoder unless you intentionally want a random-target ablation.
- Prefer a weight-loading check that reports missing/unexpected keys. Silent fallback to random initialization is a common source of invalid experiments.

---

## Configuration Reference

Key training and model flags:

| Flag | Default | Description |
|---|---|---|
| `--bridge_type` | `golden` in examples | `golden` (stage 1) or `end2end` (stage 2) |
| `--compute-bridge-loss` | off | Enable bridge/world-model loss |
| `--use-separate-projector-for-loss` | off | Use a separate bridge loss projector |
| `--use-state-future` | off | Enable state-future data/model path |
| `--compute-state-future-loss` | off | Add the SF loss |
| `--state-future-loss-weight` | `1.0` | SF loss weight; experiments often use `0.05` |
| `--state-future-decoder` | `query_ca` | `query_ca` or `mean_pool` |
| `--state-future-dim` | model/config | SF label width |
| `--state-future-horizon` | action horizon | Number of future frames |
| `--state-future-num-heads` | `8` | `query_ca` attention heads |
| `--state-future-wrist-only` | off | Restrict SF to wrist state |
| `--feed-state-future-to-dit` | off | Feed predicted SF tokens into System 1 |
| `--sf-dit-repr` | `shat` | `shat` or `mid_hidden` |
| `--use-mot-system1` | off | Enable MoT System 1 |
| `--mot-include-sf` | on | Include SF modality in MoT |
| `--mot-sf-num-tokens` | `17` | Number of SF tokens |
| `--mot-num-layers` | model/config | MoT layer count |
| `--loss-combine-mode` | `additive` | `additive` or `legacy_avg` |
| `--select_layer` | model/config | VLM layer used by the action expert |
| `--unified-embodiment-tag` | model/config | Shared action-head category |
| `--tune-llm` | off | Train the language-model backbone |
| `--tune-diffusion-model` | off | Train the action expert |
| `--no-tune-visual` | off | Freeze the visual encoder |

The public code supports both `query_ca` and `mean_pool` SF decoders. The 36-layer mean-pool SF recipe uses `mean_pool`; check the checkpoint configuration before overriding it.

---

## Verification and Smoke Tests

Run these before a long training job:

```bash
# CPU checks for SF masks, decoder shapes, and config round-trip
python scripts/verify_state_future_fixes.py

# SF-to-System-1 smoke test
python scripts/smoke_feed_sf_to_dit.py

# MoT smoke tests
python scripts/smoke_mot_system1.py
python scripts/smoke_mot_cond_experts.py

# Short training smoke runs
SMOKE_STEPS=5 bash scripts/run_sf_dual_channel_2k.sh
SMOKE_STEPS=5 RUN_WHICH=mot bash scripts/run_mot_vs_dense_aligned_2k.sh
```

For SF diagnostics:

```bash
python scripts/diag_sf_per_frame_and_probe.py --help
python scripts/diag_sf_probe_normalized.py --help
```

---

## Relationship to DIAL02

This repository is a research fork of **DIAL02**.

- Baseline: [weibinlee2024/DIAL02](https://github.com/weibinlee2024/DIAL02) at commit `986a7dd`.
- This branch: SF decoder, optional SF feedback, optional MoT, data/config plumbing, and experiment scripts.

If the baseline commit is present in your checkout, inspect the delta with:

```bash
git diff --stat 986a7dd HEAD
git diff 986a7dd HEAD -- gr00t/
```

Major changes:

| Area | Change |
|---|---|
| System-2 SF | `StateFutureDecoder` with `query_ca`; `mean_pool` retained |
| SF-to-System-1 | `state_future_to_cond` and `feed_state_future_to_dit` |
| System-1 | Optional `MoTDiT` / `MoTTransformerBlock` |
| Losses | Additive and `legacy_avg` combination modes |
| Data | `state_future` modality, masks, padding, and transforms |
| Training | SF/MoT CLI plumbing and checkpoint config persistence |
| Engineering | AdamW `foreach=False` OOM fix and trainable-parameter logging |
| Cleanup | Removed stale experiment scripts and versioned bytecode |

The complete change history relative to the baseline remains available in Git.

---

## Troubleshooting

### The installer cannot find packaging metadata

This is a source checkout. If `pip install -e .` is unavailable, skip the editable install and add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### OOM during full fine-tuning

- Use `MODEL_BF16=1` when supported.
- Reduce micro-batch size and increase gradient accumulation.
- Keep the visual encoder frozen.
- Enable gradient checkpointing.
- Use the `AdamW(foreach=False)` behavior implemented in `gr00t/experiment/trainer.py`.

### Video decoding fails on LIBERO-plus

LIBERO-plus videos are AV1. Use `torchvision_av`/PyAV; `decord` is not sufficient for these files.

### `bridge_goal_model` missing-weight warning

Shared bridge checkpoints intentionally omit duplicated `bridge_goal_model` weights. This is expected for inference. If you train with `--compute-bridge-loss`, make sure the goal encoder is trained or intentionally frozen; otherwise the bridge loss target is random.

### Checkpoint config mismatch

If `select_layer`, action horizon, SF decoder, or embodiment tag does not match the checkpoint, the loader may warn or rebuild modules. Use the checkpoint configuration and avoid silent `ignore_mismatched_sizes` fallbacks in training.

### Hard-coded local paths

Some experiment scripts and `examples/eval.sh` contain local paths from the development machine. Replace them with your dataset, checkpoint, RoboCasa, robosuite, and conda paths before running.

### LIBERO-plus action saturation

The LIBERO-plus branch is experimental. The documented pilot analysis found severe action saturation and a GR1-to-LIBERO action-head mismatch. Treat those runs as debugging experiments, not benchmark results.

---

## License and Acknowledgements

The source files retain their upstream SPDX headers, including Apache-2.0 notices where applicable. Upstream components, datasets, and checkpoints keep their original licenses and terms.

This work builds on:

- **DIAL02**: the dual-system policy baseline.
- **NVIDIA Isaac GR00T N1.5**: the robot foundation-model stack.
- **RoboCasa / GR1 tabletop environments** for simulation data and evaluation.
- **LIBERO-plus** for the experimental lifelong manipulation benchmark.

Datasets and pretrained checkpoints are not redistributed by this repository. Please follow the upstream licenses when downloading or using them.

---

## Contact

For questions, bug reports, or reproduction notes, open an issue in the public repository:

https://github.com/weibinlee2024/DAIL02_SF

When reporting a training or evaluation issue, include the commit, checkpoint path, data config, `select_layer`, effective batch, GPU type, and the exact command used.
