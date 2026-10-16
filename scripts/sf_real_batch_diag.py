"""用训练入口的模型和 batch 检查 SF 梯度，不更新参数或保存模型。"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


def _grad_stats(loss, named_parameters):
    if not torch.is_tensor(loss) or loss.numel() != 1 or not loss.requires_grad:
        raise RuntimeError("Expected a scalar differentiable loss")
    grads = torch.autograd.grad(
        loss, [p for _, p in named_parameters], allow_unused=True, retain_graph=True
    )
    rows = []
    for (name, _), grad in zip(named_parameters, grads):
        if grad is None:
            rows.append({"name": name, "connected": False, "norm": 0.0})
        else:
            if not torch.isfinite(grad).all():
                raise RuntimeError(f"Non-finite gradient: {name}")
            rows.append({"name": name, "connected": True,
                         "norm": float(grad.detach().float().norm())})
    return {"norm": sum(r["norm"] ** 2 for r in rows) ** 0.5,
            "connected": sum(r["connected"] for r in rows),
            "parameters": len(rows), "details": rows}


def diagnose(model, batch, *, seed=123, context_factory=nullcontext):
    """固定一个 batch，在 eval 下保留梯度，比较 detach 开关两边的结果。

    为省显存，临时冻结 SF 预测头和 cond 投影以外的参数。
    DiT 仍向 SF 输入传梯度；这里不测共享 backbone 的梯度。
    """
    for key in ("use_state_future", "compute_state_future_loss", "feed_state_future_to_dit"):
        if not getattr(model, key, False):
            raise RuntimeError(f"Required flag is false: {key}")
    if not hasattr(model, "sf_tokens_detach"):
        raise RuntimeError("Model has no sf_tokens_detach; apply student's detach implementation first")
    if getattr(model, "sf_dit_repr", None) != "shat":
        raise RuntimeError("This diagnostic supports shat only, not mid_hidden")
    if getattr(model, "loss_combine_mode", None) != "additive":
        raise RuntimeError("Expected additive loss; do not silently compare different loss definitions")
    decoder = [(n, p) for n, p in model.named_parameters()
               if n.startswith(("state_future_projector.", "state_future_decoder."))]
    cond = [(n, p) for n, p in model.named_parameters() if n.startswith("state_future_to_cond.")]
    if not decoder or not cond or any(not p.requires_grad for _, p in decoder + cond):
        raise RuntimeError("SF decoder/cond parameters missing or frozen; check training flags")
    modes = [(m, m.training) for m in model.modules()]
    trainable = [(p, p.requires_grad) for p in model.parameters()]
    original_detach = model.sf_tokens_detach
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    allowed = {id(p) for _, p in decoder + cond}
    report = {"mode": "eval_with_grad", "seed": seed,
              "gradient_scope": "SF decoder and state_future_to_cond only",
              "original_detach": original_detach,
              "bridge_type": getattr(model, "bridge_type", None),
              "state_future_loss_weight": float(model.state_future_loss_weight),
              "action_metric": "actual forward action_loss (not rollout action MSE)",
              "runs": {}, "checks": {}}

    def one_run(detach, intervention=None, gradients=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.sf_tokens_detach = detach
        captured = {}

        def capture_input(module, args):
            value = args[0]
            captured["prediction"] = value.detach().float().cpu().clone()
            captured["sf_input_requires_grad"] = value.requires_grad

        def alter_output(module, args, value):
            captured["calls"] = captured.get("calls", 0) + 1
            if intervention == "zero_projected_sf":
                modified = torch.zeros_like(value)
            elif intervention == "swap_projected_sf":
                if value.shape[0] < 2:
                    raise RuntimeError("Swap requires batch size >= 2")
                modified = value.roll(1, dims=0)
            else:
                modified = value
            captured["cond_delta_max"] = float((modified - value).detach().abs().max())
            return modified

        handles = [model.state_future_to_cond.register_forward_pre_hook(capture_input),
                   model.state_future_to_cond.register_forward_hook(alter_output)]
        try:
            # forward 会改动输入字典，每次用一份副本。
            with torch.set_grad_enabled(gradients), context_factory():
                output = model.forward(copy.deepcopy(batch), action_mode=False)
            if captured.get("calls") != 1:
                raise RuntimeError(f"Expected one SF->cond invocation, got {captured.get('calls', 0)}")
            for key in ("action_loss", "state_future_loss"):
                if key not in output or not torch.isfinite(output[key]).all():
                    raise RuntimeError(f"Missing/nonfinite {key}")
            row = {key: float(output[key].detach()) for key in
                   ("action_loss", "bridge_loss", "state_future_loss") if key in output}
            row.update({k: v for k, v in captured.items() if k != "prediction"})
            if gradients:
                row["action_to_decoder"] = _grad_stats(output["action_loss"], decoder)
                row["action_to_cond"] = _grad_stats(output["action_loss"], cond)
                row["sf_to_decoder"] = _grad_stats(output["state_future_loss"], decoder)
                denom = abs(model.state_future_loss_weight) * row["sf_to_decoder"]["norm"]
                row["action_over_weighted_sf_decoder_grad_norm"] = (
                    row["action_to_decoder"]["norm"] / denom if denom > 0 else None)
            return row, captured["prediction"]
        finally:
            for handle in handles:
                handle.remove()

    try:
        model.eval()
        for p, _ in trainable:
            p.requires_grad_(id(p) in allowed)
        off, pred_off = one_run(False)
        on, pred_on = one_run(True)
        report["runs"].update(no_detach=off, detach=on)
        report["prediction_shape"] = list(pred_off.shape)
        report["prediction_detach_max_diff"] = float((pred_off - pred_on).abs().max())
        checks = report["checks"]
        checks["no_detach_action_reaches_decoder"] = off["action_to_decoder"]["norm"] > 0
        checks["detach_action_disconnected_from_decoder"] = on["action_to_decoder"]["connected"] == 0
        checks["cond_receives_action_grad_both"] = all(r["action_to_cond"]["norm"] > 0 for r in (off, on))
        checks["decoder_receives_sf_grad_both"] = all(r["sf_to_decoder"]["norm"] > 0 for r in (off, on))
        checks["same_forward_prediction"] = torch.allclose(pred_off, pred_on, rtol=1e-5, atol=1e-6)
        checks["same_forward_losses"] = all(
            abs(off[k] - on[k]) <= 1e-6 + 1e-5 * abs(off[k])
            for k in ("action_loss", "state_future_loss", "bridge_loss") if k in off)
        # CUDA/AMP 可能有小的数值差异，检查失败后再看具体差值。
        for intervention in ("zero_projected_sf", "swap_projected_sf"):
            if intervention.startswith("swap") and pred_off.shape[0] < 2:
                report["runs"][intervention] = {"skipped": "batch size < 2"}
                continue
            row, _ = one_run(False, intervention, gradients=False)
            row["action_loss_delta_from_normal"] = row["action_loss"] - off["action_loss"]
            report["runs"][intervention] = row
        report["gradient_routing_checks_passed"] = all(checks.values())
        return report
    finally:
        model.sf_tokens_detach = original_detach
        for p, value in trainable:
            p.requires_grad_(value)
        for module, value in modes:
            module.training = value
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def validate_loading(info):
    if info is None:
        raise RuntimeError("No checkpoint loading_info; do not audit a silently random-initialized model")
    # DIAL save path drops bridge_goal_model.* when goal shares bridge_vision_model
    # (see set_trainable_parameters: del bridge_goal_model; alias to vision).
    # Those keys show up as missing on reload before aliasing; allow only that prefix.
    missing = list(info.get("missing_keys") or [])
    allowed_missing_prefix = "bridge_goal_model."
    benign_missing = [k for k in missing if k.startswith(allowed_missing_prefix)]
    real_missing = [k for k in missing if not k.startswith(allowed_missing_prefix)]
    if benign_missing:
        print(
            f"[SF DIAG] allowing {len(benign_missing)} missing keys under "
            f"{allowed_missing_prefix} (shared with bridge_vision_model at save time)",
            flush=True,
        )
    errors = {}
    if real_missing:
        errors["missing_keys"] = real_missing
    for key in ("unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            errors[key] = info.get(key)
    if errors:
        raise RuntimeError(f"Checkpoint does not strictly match this run: {errors}")


def validate_launch(config):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or config.num_gpus != 1:
        raise RuntimeError("Run this diagnostic on a single GPU, without distributed launch")
    if config.resume:
        raise RuntimeError("Do not use --resume: pass the existing checkpoint via --base-model-path")
    if config.resume_pretrained_option != "all" or config.resume_vlm_path or config.resume_action_head_path:
        raise RuntimeError("Diagnostic requires loading one complete checkpoint, not replacing model components")
    checkpoint = Path(config.base_model_path)
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise RuntimeError("--base-model-path must be a trained checkpoint directory, not base JSON")
    if not any(checkpoint.glob("*.safetensors")) and not any(checkpoint.glob("pytorch_model*.bin")):
        raise RuntimeError("No checkpoint weight files found")
    output = Path(config.output_dir)
    if (output / "experiment_cfg").exists() or any(output.glob("checkpoint-*")):
        raise RuntimeError("Use a NEW diagnostic output directory, not an existing training run")
    if config.lora_rank > 0:
        raise RuntimeError("LoRA wrapper is outside this diagnostic's tested scope")


def run_from_trainer(trainer, config):
    checkpoint = Path(config.base_model_path)
    model = trainer.model
    validate_loading(getattr(model, "_sf_diag_loading_info", None))
    if config.lora_rank > 0:
        raise RuntimeError("LoRA wrapper is outside this diagnostic's tested scope")
    # 直接用训练入口的 dataset 和 collator 取样。
    indices = [int(i) for i in os.environ.get("DIAL_SF_DIAG_INDICES", "0,1").split(",")]
    if not indices or min(indices) < 0:
        raise ValueError("DIAL_SF_DIAG_INDICES must contain nonnegative sample indices")
    samples = [trainer.train_dataset[i] for i in indices]
    batch = trainer.data_collator(samples)
    batch = trainer._prepare_inputs(batch)
    _, action_inputs = model.prepare_input(copy.deepcopy(batch))
    target = action_inputs.get("state_future")
    mask = action_inputs.get("state_future_mask")
    if target is None or mask is None or target.shape != mask.shape:
        raise RuntimeError("Missing SF target/mask or mismatched shapes")
    if not torch.isfinite(target).all() or not torch.isfinite(mask).all() or (mask < 0).any():
        raise RuntimeError("Invalid SF target/mask")
    if (mask.flatten(1).sum(1) <= 0).any():
        raise RuntimeError("An SF sample has zero valid supervision")
    batch_info = {"target_shape": list(target.shape), "mask_sum": float(mask.sum()),
                  "mask_sum_per_sample": mask.flatten(1).sum(1).float().cpu().tolist(),
                  "mask_sum_per_time": mask.sum(-1).float().cpu().tolist(),
                  "sample_indices": indices}
    del action_inputs, target, mask, samples
    device_type = next(model.parameters()).device.type
    if device_type == "cuda" and trainer.args.bf16:
        # 还没进入 train()，Accelerator 没有包装 forward，这里手动开启 AMP。
        context_factory = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    elif device_type == "cuda" and trainer.args.fp16:
        context_factory = lambda: torch.autocast("cuda", dtype=torch.float16)
    else:
        context_factory = trainer.compute_loss_context_manager
    report = diagnose(model, batch, context_factory=context_factory)
    report["batch"] = batch_info
    report["checkpoint"] = str(checkpoint.resolve())
    report["checkpoint_config_sha256"] = hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest()
    report["effective_model_config"] = model.config.to_dict()
    report["launch_config"] = vars(config)
    source_paths = [Path(__file__), Path(inspect.getfile(type(model)))]
    report["source_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source_paths}
    report["torch_version"] = torch.__version__
    report["checkpoint_weight_files"] = [
        {"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in checkpoint.iterdir() if p.suffix in (".safetensors", ".bin")]
    dest = Path(config.output_dir) / "sf_real_batch_diag.json"
    # 不覆盖上一次结果。
    with dest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str, allow_nan=False)
    print(f"[SF DIAG] report: {dest}", flush=True)
    print(f"[SF DIAG] routing checks passed: {report['gradient_routing_checks_passed']}", flush=True)
    if not report["gradient_routing_checks_passed"]:
        raise RuntimeError("SF routing diagnostic flagged a failure; inspect JSON before training")
