# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import torchvision.transforms.functional as TF
from copy import deepcopy
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import timm

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .action_head.cross_attention_dit import BridgeCrossAttentionTransformer
from .action_head.flow_matching_action_head_dial import (
    FlowmatchingActionHeadDIAL,
    FlowmatchingActionHeadDIALConfig,
)
from .backbone import VLMBackbone
from gr00t.model.transforms import BRIDGE_TOKENS

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3

# Default SF target width: dual-arm wrist pos+rot6d + 6-DoF hands (no padding).
# 2026-10-11: default SF target dim 30 -> 60 (+10 fingertip pos).
# Pre-change backup: bake20261011_sf_d30/
DEFAULT_STATE_FUTURE_DIM = 30  # mainline mean_pool+motcond; set SF_FORCE_KEY_DIM=60 for fingertips
DEFAULT_STATE_FUTURE_NUM_HEADS = 8
VALID_STATE_FUTURE_DECODERS = ("query_ca", "mean_pool")


class StateFutureDecoder(nn.Module):
    """Per-frame learnable queries over bridge tokens (replaces mean-pool MLP).

    queries: (num_frames, hidden) attend to (B, n_tokens, hidden) and each
    query is projected to one frame of the SF target (D_sf dims).
    """

    def __init__(
        self,
        hidden: int,
        num_frames: int,
        out_dim: int,
        num_heads: int = DEFAULT_STATE_FUTURE_NUM_HEADS,
    ):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(
                f"hidden={hidden} must be divisible by state_future_num_heads={num_heads}"
            )
        self.num_frames = int(num_frames)
        self.out_dim = int(out_dim)
        self.queries = nn.Parameter(torch.empty(self.num_frames, hidden))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_out = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.proj = nn.Linear(hidden, self.out_dim)

    def forward(self, tokens: torch.Tensor, return_hidden: bool = False):
        # tokens: [B, n_bridge, hidden]  ->  [B, num_frames, out_dim]
        # mid_hidden (pre-proj): [B, num_frames, hidden] — used as DiT condition tokens
        batch_size = tokens.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.norm_q(queries)
        attended, _ = self.attn(queries, tokens, tokens, need_weights=False)
        hidden_states = attended + queries
        hidden_states = hidden_states + self.ffn(self.norm_out(hidden_states))
        pred = self.proj(hidden_states)
        if return_hidden:
            return pred, hidden_states
        return pred


def compute_global_mean_std_patch(feat: torch.Tensor):
    """
    feat: [B, P, D]
    Returns:
        global_mean: [P, D]
        global_std:  [P, D]
    """
    x = feat.float()
    B, P, D = x.shape
    device = x.device

    distributed = dist.is_available() and dist.is_initialized()

    # ------ local statistics ------
    local_count = torch.tensor([B], device=device, dtype=torch.float32)  # batch size on current GPU
    local_sum = x.sum(dim=0)               # [P, D]
    local_sum_sq = (x * x).sum(dim=0)      # [P, D]

    # ------ distributed reduce ------
    if distributed:
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM)

    # ------ compute global mean & variance ------
    global_mean = local_sum / local_count
    global_var = local_sum_sq / local_count - global_mean ** 2
    global_std = torch.sqrt(global_var)

    global_mean, global_std = global_mean.to(feat.device).detach(), global_std.to(feat.device).detach()

    return global_mean, global_std


@dataclass
class GR00T_N1_5_DIAL_Config(PretrainedConfig):
    model_type = "gr00t_n1_5_dial"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    bridge_cfg: dict = field(init=False, metadata={"help": "Bridge configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    bridge_cross_attention_cfg: dict = field(default=None)

    classifier_free_guidance_cfg: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5_DIAL(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_DIAL_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_DIAL_Config,
        local_model_path: str,
        tokenizer_len: int=None,
        bridge_type: str="end2end",
        compute_bridge_loss: bool=False,
        select_layer: int=None,
        goal_image_type: str="future",
        bridge_loss_type: str="mse",
        use_image_type_embedding: bool=False,
        use_vl_mask: bool=False,
        correct_vl_mask: bool=False,
        action_only_one_obs: bool=False,

        noise_tau: float=0,
        omit_image_type_embedding_for_goal: bool=False,
        reweight_noise: bool=False,

        unified_embodiment_id: int=None,
        use_separate_projector_for_loss: bool=False,
        vlm_small_lr: float=None,
        select_layer_for_bridge: int=None,
        matching_coeff: float = None,

        use_state_future: bool = False,
        compute_state_future_loss: bool = False,
        state_future_loss_weight: float = 1.0,
        state_future_horizon: int = None,
        tune_state_future_projector: bool = True,
        state_future_decoder: str = "query_ca",
        state_future_dim: int = None,
        state_future_num_heads: int = DEFAULT_STATE_FUTURE_NUM_HEADS,
        # "additive": L = action + bridge + w*sf  (teacher SF-weight debug)
        # "legacy_avg": L = ((a+b)/2 + w*sf)/(1+w)  (old 6task10k formula)
        loss_combine_mode: str = "additive",
        # Concat SF into bridge pack → DiT encoder_hidden_states
        feed_state_future_to_dit: bool = False,
        # "shat": project predicted Ŝ [B,T,D_sf] → condition tokens (teacher default)
        # "mid_hidden": use query_ca pre-proj hidden (legacy 2A experiment)
        sf_dit_repr: str = "shat",
        # Ablation: stop-grad Ŝ before state_future_to_cond so action_loss cannot reshape SF decoder
        sf_tokens_detach: bool = False,
        # Train-time dropout: replace SF cond tokens with learnable null (CFG-style)
        p_sf_drop: float = 0.0,
        # Train-time intent-swap: with this prob, roll SF tokens across batch (wrong foresight)
        p_sf_swap: float = 0.0,
        # Mixture-of-Transformers System-1 (action head)
        use_mot_system1: bool = False,
        # Dense-CA MoT cond experts (teacher口径); mutually exclusive with use_mot_system1
        use_mot_cond_experts: bool = False,
        mot_include_sf: bool = True,
        mot_sf_num_tokens: int = 17,
        mot_num_layers: int = None,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        if select_layer is not None:
            config.backbone_cfg['select_layer'] = select_layer
        if select_layer_for_bridge is not None:
            config.backbone_cfg['select_layer_for_bridge'] = select_layer_for_bridge

        config.action_head_cfg['use_vl_mask'] = use_vl_mask
        config.action_head_cfg['correct_vl_mask'] = correct_vl_mask
        if matching_coeff is not None:
            config.action_head_cfg['matching_coeff'] = matching_coeff
        if use_mot_system1 and use_mot_cond_experts:
            raise ValueError(
                "use_mot_system1 and use_mot_cond_experts are mutually exclusive"
            )
        config.action_head_cfg['use_mot_system1'] = bool(use_mot_system1)
        config.action_head_cfg['use_mot_cond_experts'] = bool(use_mot_cond_experts)
        config.action_head_cfg['mot_include_sf'] = bool(mot_include_sf)
        config.action_head_cfg['mot_sf_num_tokens'] = int(mot_sf_num_tokens)
        if use_mot_cond_experts:
            config.action_head_cfg.setdefault("diffusion_model_cfg", {})
            config.action_head_cfg["diffusion_model_cfg"] = dict(
                config.action_head_cfg["diffusion_model_cfg"]
            )
            config.action_head_cfg["diffusion_model_cfg"]["use_mot_cond_experts"] = True
        if use_mot_system1 and mot_num_layers is not None:
            config.action_head_cfg.setdefault("diffusion_model_cfg", {})
            config.action_head_cfg["diffusion_model_cfg"] = dict(
                config.action_head_cfg["diffusion_model_cfg"]
            )
            config.action_head_cfg["diffusion_model_cfg"]["num_layers"] = int(mot_num_layers)
            print(f"MoT override num_layers → {mot_num_layers}")

        super().__init__(config)
        self.local_model_path = local_model_path

        if tokenizer_len is None:
            tokenizer_len = config.bridge_cfg['tokenizer_len']
        else:
            if 'tokenizer_len' in config.bridge_cfg:
                assert tokenizer_len == config.bridge_cfg['tokenizer_len']
                
        self.backbone = VLMBackbone(**config.backbone_cfg, tokenizer_len=tokenizer_len)
        action_head_cfg = FlowmatchingActionHeadDIALConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHeadDIAL(action_head_cfg, backbone_hidden_size=self.backbone.vlm_model.config.hidden_size)

        print(f"Use bridge: {self.config.bridge_cfg['use_bridge']}")
        self.use_bridge = self.config.bridge_cfg['use_bridge']
        if self.config.bridge_cfg['use_bridge']:
            self.use_dino_vision = self.config.bridge_cfg.get('use_dino_vision', False)
            self.use_dino_goal = self.config.bridge_cfg.get('use_dino_goal', False)
            self.dino_path = self.config.bridge_cfg.get('dino_path', None)

            if self.use_dino_vision or self.use_dino_goal:
                print(f"Loading DINO from {self.dino_path}")
                self.dino_model = timm.create_model(
                    model_name=self.dino_path,
                    pretrained=True,
                    num_classes=0,
                    img_size=224,
                )

                if not hasattr(self.dino_model, '_initialize_weights'):
                    self.dino_model._initialize_weights = lambda *args, **kwargs: None

                dino_dim = self.dino_model.num_features
                llm_hidden_size = self.backbone.vlm_model.config.hidden_size
                # Dimension alignment projection layer
                self.dino_proj = nn.Linear(dino_dim, llm_hidden_size) if dino_dim != llm_hidden_size else nn.Identity()
                # DINO normalization
                self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            if not (self.use_dino_vision and self.use_dino_goal):
                if hasattr(self.backbone.vlm_model, "vision_model"):
                    self.bridge_vision_model = deepcopy(self.backbone.vlm_model.vision_model)
                    self.bridge_goal_model = deepcopy(self.backbone.vlm_model.vision_model)
                else:
                    self.bridge_vision_model = deepcopy(self.backbone.vlm_model.visual)
                    self.bridge_goal_model = deepcopy(self.backbone.vlm_model.visual)

            print(f"GR00T_N1_5_DIAL num_bridge_tokens: {len(BRIDGE_TOKENS)}")
            self.num_bridge_tokens = len(BRIDGE_TOKENS)

            self.bridge_projector = nn.Sequential(
                nn.Linear(self.backbone.vlm_model.config.hidden_size, self.backbone.vlm_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.backbone.vlm_model.config.hidden_size, self.backbone.vlm_model.config.hidden_size),
            )

            self.use_separate_projector_for_loss = use_separate_projector_for_loss
            if self.use_separate_projector_for_loss:
                self.bridge_projector_for_loss = deepcopy(self.bridge_projector)

            if config.bridge_cross_attention_cfg is not None:
                print(f"config.bridge_cross_attention_cfg: {config.bridge_cross_attention_cfg}")
                self.bridge_cross_attention = BridgeCrossAttentionTransformer(**config.bridge_cross_attention_cfg)
            else:
                self.bridge_cross_attention = None

            if config.classifier_free_guidance_cfg is not None:
                print(f"config.classifier_free_guidance_cfg: {config.classifier_free_guidance_cfg}")
                self.null_subgoal_embed = nn.Parameter(torch.randn(1, self.num_bridge_tokens, self.backbone.vlm_model.config.hidden_size))
                self.p_uncond = config.classifier_free_guidance_cfg['p_uncond']
                self.register_buffer("null_initialized", torch.tensor(False))
                assert 0 < self.p_uncond < 1
            else:
                self.p_uncond = None

            self.vlm_small_lr = vlm_small_lr
            print(f"vlm_small_lr: {self.vlm_small_lr}")

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.bridge_type = bridge_type
        self.compute_bridge_loss = compute_bridge_loss
        self.goal_image_type = goal_image_type
        self.bridge_loss_type = bridge_loss_type

        self.use_image_type_embedding = use_image_type_embedding
        self.omit_image_type_embedding_for_goal = omit_image_type_embedding_for_goal
        if use_image_type_embedding:
            self.image_type_embedding = nn.Embedding(3, self.backbone.vlm_model.config.hidden_size)
            nn.init.normal_(self.image_type_embedding.weight, mean=0.0, std=0.02)
        
        self.action_only_one_obs = action_only_one_obs
        self.noise_tau = noise_tau
        self.reweight_noise = reweight_noise
        self.action_head.unified_embodiment_id = unified_embodiment_id

        # Physical State Future Foresight (auxiliary head; default off — bypasses the Action Head)
        self.use_state_future = bool(use_state_future)
        self.compute_state_future_loss = bool(compute_state_future_loss)
        self.state_future_loss_weight = float(state_future_loss_weight)
        mode = (loss_combine_mode or "additive").strip().lower()
        if mode not in ("additive", "legacy_avg"):
            raise ValueError(
                f"loss_combine_mode={loss_combine_mode!r}; expected 'additive' or 'legacy_avg'"
            )
        self.loss_combine_mode = mode
        self.feed_state_future_to_dit = bool(feed_state_future_to_dit)
        repr_mode = (sf_dit_repr or "shat").strip().lower()
        if repr_mode not in ("shat", "mid_hidden"):
            raise ValueError(f"sf_dit_repr={sf_dit_repr!r}; expected 'shat' or 'mid_hidden'")
        self.sf_dit_repr = repr_mode
        self.sf_tokens_detach = bool(sf_tokens_detach)
        self.p_sf_drop = float(p_sf_drop)
        self.p_sf_swap = float(p_sf_swap)
        if not (0.0 <= self.p_sf_drop <= 1.0):
            raise ValueError(f"p_sf_drop={p_sf_drop!r}; expected in [0, 1]")
        if not (0.0 <= self.p_sf_swap <= 1.0):
            raise ValueError(f"p_sf_swap={p_sf_swap!r}; expected in [0, 1]")
        # Online / eval SR ablations (deterministic). Train path still uses p_sf_drop/swap.
        # correct | zero | swap  — see set_sf_eval_mode()
        self.sf_eval_mode = "correct"
        self._sf_token_bank = []  # past SF cond tokens for intent-swap at B=1
        self._sf_token_bank_max = 64
        sf_horizon = state_future_horizon
        if sf_horizon is None:
            sf_horizon = config.action_horizon
        self.state_future_horizon = int(sf_horizon)
        self.tune_state_future_projector = tune_state_future_projector
        decoder_type = (state_future_decoder or "query_ca").strip().lower()
        if decoder_type not in VALID_STATE_FUTURE_DECODERS:
            raise ValueError(
                f"state_future_decoder={state_future_decoder!r}; "
                f"expected one of {VALID_STATE_FUTURE_DECODERS}"
            )
        self.state_future_decoder_type = decoder_type
        self.state_future_num_heads = int(state_future_num_heads)
        if state_future_dim is not None:
            self.state_dim_for_future = int(state_future_dim)
        elif decoder_type == "mean_pool":
            self.state_dim_for_future = int(
                config.action_head_cfg.get("max_state_dim", config.action_dim)
            )
        else:
            self.state_dim_for_future = int(DEFAULT_STATE_FUTURE_DIM)
        if self.feed_state_future_to_dit and self.use_state_future:
            if self.sf_dit_repr == "mid_hidden" and decoder_type != "query_ca":
                raise ValueError(
                    "sf_dit_repr='mid_hidden' requires state_future_decoder='query_ca'"
                )
        if self.use_state_future and self.use_bridge:
            hidden = self.backbone.vlm_model.config.hidden_size
            num_frames = self.state_future_horizon + 1
            if decoder_type == "query_ca":
                self.state_future_decoder = StateFutureDecoder(
                    hidden=hidden,
                    num_frames=num_frames,
                    out_dim=self.state_dim_for_future,
                    num_heads=self.state_future_num_heads,
                )
                self.state_future_decoder.requires_grad_(self.tune_state_future_projector)
                n_sf = sum(p.numel() for p in self.state_future_decoder.parameters())
                print(
                    f"State future decoder: query_ca  H={self.state_future_horizon}, "
                    f"out_frames={num_frames}, dim={self.state_dim_for_future}, "
                    f"heads={self.state_future_num_heads}, params={n_sf}, "
                    f"compute_loss={self.compute_state_future_loss}, "
                    f"feed_to_dit={self.feed_state_future_to_dit}, "
                    f"sf_dit_repr={self.sf_dit_repr}"
                )
            else:
                self.state_future_projector = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, num_frames * self.state_dim_for_future),
                )
                self.state_future_projector.requires_grad_(self.tune_state_future_projector)
                print(
                    f"State future projector: mean_pool  H={self.state_future_horizon}, "
                    f"out_frames={num_frames}, dim={self.state_dim_for_future}, "
                    f"compute_loss={self.compute_state_future_loss}"
                )
            if self.feed_state_future_to_dit:
                # Type emb to distinguish proprio foresight tokens from visual bridge pack
                self.proprio_foresight_type_emb = nn.Parameter(torch.zeros(1, 1, hidden))
                nn.init.normal_(self.proprio_foresight_type_emb, mean=0.0, std=0.02)
                # One null token per foresight frame. Checkpoints store [1, T, H]
                # (T = horizon + 1). A length-1 parameter cannot load those weights.
                self.null_sf_embed = nn.Parameter(torch.zeros(1, num_frames, hidden))
                nn.init.normal_(self.null_sf_embed, mean=0.0, std=0.02)
                self.null_sf_embed.requires_grad_(self.tune_state_future_projector)
                if self.sf_dit_repr == "shat":
                    # Project raw predicted Ŝ [..., D_sf] → condition token dim
                    self.state_future_to_cond = nn.Sequential(
                        nn.Linear(self.state_dim_for_future, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, hidden),
                    )
                    self.state_future_to_cond.requires_grad_(self.tune_state_future_projector)
                    print(
                        f"SF→DiT cond projector: Ŝ dim={self.state_dim_for_future} → {hidden}"
                    )
                print(
                    f"SF cond ablations: detach={self.sf_tokens_detach} "
                    f"p_sf_drop={self.p_sf_drop} p_sf_swap={self.p_sf_swap}"
                )
        else:
            print("State future projector: DISABLED")
        self.register_buffer("bridge_training_steps", torch.tensor(0, dtype=torch.float32), persistent=False)

    def _extract_dino(self, pixel_values, downsample_to_n=None):
        """
        Processing pipeline: Resize(224) -> Norm -> DINO -> (Optional) Pool -> Proj
        pixel_values: [B, C, H, W] or [B*V, C, H, W]
        """
        if pixel_values.shape[-1] != 224 or pixel_values.shape[-2] != 224:
            x = F.interpolate(pixel_values.float(), size=(224, 224), mode='bicubic', align_corners=False)
        else:
            x = pixel_values
        if x.max() > 1.0: x = x / 255.0
        x = (x - self.dino_mean) / self.dino_std

        with torch.set_grad_enabled(getattr(self, 'tune_bridge_visual', False) and self.training):
            feats = self.dino_model.forward_features(x)
            patch_feats = feats[:, 5:, :]

        if downsample_to_n is not None:
            B_V, N, D = patch_feats.shape
            side = int(np.sqrt(N)) # 16
            target_side = int(np.sqrt(downsample_to_n)) # 8
            stride = side // target_side # 2
            
            grid = patch_feats.transpose(1, 2).reshape(B_V, D, side, side)
            grid = F.avg_pool2d(grid, kernel_size=stride, stride=stride)
            patch_feats = grid.flatten(2).transpose(1, 2)

        return self.dino_proj(patch_feats)

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
        action_mode: bool=False,
        backbone_features: torch.Tensor = None,
    ) -> BatchFeature:
        guidance_scale = inputs.pop("guidance_scale", 1.0)

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        if action_mode and (backbone_features is not None):
            backbone_outputs = BatchFeature(data={"backbone_features": backbone_features})
        else:
            backbone_outputs = self.backbone(backbone_inputs)

        if action_mode:
            raw_backbone_features = deepcopy(backbone_outputs["backbone_features"])

        batch_size = inputs['state'].shape[0]

        output_dict = {}
        pred_state_future = None
        if self.use_bridge:
            if self.vlm_small_lr is not None:
                backbone_outputs['backbone_features'] = backbone_outputs['backbone_features'] * self.vlm_small_lr + backbone_outputs['backbone_features'].detach() * (1-self.vlm_small_lr)

            backbone_outputs.data = {k: v[:, -self.num_bridge_tokens:] for k, v in backbone_outputs.items()}

            # Extract raw backbone features (before being overwritten by projector)
            raw_backbone_feat = backbone_outputs['backbone_features']
            # If backbone provides dedicated bridge layer features, use them for loss computation
            raw_feat_for_loss = backbone_outputs.get('backbone_features_for_bridge', raw_backbone_feat)

            # Physical State Future: predict early so Ŝ (or mid-hidden) can enter DiT cond pack
            pred_state_future = None
            proprio_foresight_tokens = None
            if self.use_state_future:
                need_mid = (
                    self.feed_state_future_to_dit and self.sf_dit_repr == "mid_hidden"
                )
                if self.feed_state_future_to_dit or self.compute_state_future_loss:
                    if need_mid:
                        pred_state_future, sf_mid_hidden = self._predict_state_future(
                            raw_backbone_feat, batch_size, return_hidden=True
                        )
                        sf_for_cond = (
                            sf_mid_hidden.detach()
                            if self.sf_tokens_detach
                            else sf_mid_hidden
                        )
                        proprio_foresight_tokens = (
                            sf_for_cond + self.proprio_foresight_type_emb
                        )
                    else:
                        pred_state_future = self._predict_state_future(
                            raw_backbone_feat, batch_size, return_hidden=False
                        )
                        if (
                            self.feed_state_future_to_dit
                            and pred_state_future is not None
                        ):
                            # Teacher default: project raw Ŝ → condition tokens
                            # Optional detach: action_loss must not reshape SF decoder
                            sf_for_cond = (
                                pred_state_future.detach()
                                if self.sf_tokens_detach
                                else pred_state_future
                            )
                            proprio_foresight_tokens = (
                                self.state_future_to_cond(sf_for_cond)
                                + self.proprio_foresight_type_emb
                            )
                    if proprio_foresight_tokens is not None:
                        proprio_foresight_tokens = self._apply_sf_cond_ablations(
                            proprio_foresight_tokens
                        )

            # SF MSE before matching/CFG doubles the batch (keeps shapes aligned)
            if (
                pred_state_future is not None
                and self.compute_state_future_loss
                and not action_mode
            ):
                gt = action_inputs.get("state_future")
                if gt is None:
                    print(
                        "WARNING: compute_state_future_loss=True but batch has no "
                        "state_future — skipping state_future_loss "
                        "(enable use_state_future on data config)."
                    )
                else:
                    gt = gt.to(pred_state_future.dtype)
                    if gt.shape != pred_state_future.shape:
                        raise ValueError(
                            "state_future GT/pred shape mismatch: "
                            f"gt={tuple(gt.shape)} pred={tuple(pred_state_future.shape)} "
                            f"(decoder={self.state_future_decoder_type}, "
                            f"dim={self.state_dim_for_future})"
                        )
                    mask = action_inputs.get("state_future_mask")
                    if mask is not None:
                        mask = mask.to(pred_state_future.dtype)
                    diff = (pred_state_future - gt) ** 2
                    if mask is not None:
                        state_future_loss = (diff * mask).sum() / mask.sum().clamp_min(1.0)
                    else:
                        state_future_loss = diff.mean()
                    output_dict["state_future_loss"] = state_future_loss

            # Compute features for Action Head (main Projector)
            projected_feat_action = self.bridge_projector(raw_backbone_feat)
            
            # Compute features for Bridge Loss
            if getattr(self, 'use_separate_projector_for_loss', False):
                projected_feat_loss = self.bridge_projector_for_loss(raw_feat_for_loss)
            else:
                projected_feat_loss = self.bridge_projector(raw_feat_for_loss)

            # Put loss features back into backbone_outputs for subsequent bridge_loss computation
            backbone_outputs['backbone_features'] = projected_feat_loss

            # Process current observation (supports multi-view)
            if self.use_dino_vision:
                # dino_img: [Batch, Views, C, H, W]  (C=3, H=224, W=224, uint8)
                dino_img = backbone_inputs['dino_obs_images']
                B, V, C, H, W = dino_img.shape
                
                # Flatten Batch and Views dimensions to feed into DINO
                dino_img_flat = dino_img.view(B * V, C, H, W)

                raw_dino_obs = self._extract_dino(dino_img_flat, downsample_to_n=None)
                num_views = raw_dino_obs.shape[0] // batch_size
                obs_embeds = raw_dino_obs.reshape(batch_size, num_views * 256, -1)
                obs_tokens_per_img = 256
            else:
                if "act_obs_image_pixel_values" in backbone_inputs:
                    obs_image_prefix = "act_obs_image_"
                    obs_input = {
                        k.removeprefix(obs_image_prefix): v
                        for k, v in backbone_inputs.items()
                        if k.startswith(obs_image_prefix)
                    }
                else:
                    # Original VLM extraction logic
                    obs_input = {k.removeprefix("vlm_"): v for k, v in backbone_inputs.items() if k.startswith("vlm_")}
                    obs_input.pop("image_sizes", None); obs_input.pop("input_ids", None); obs_input.pop("attention_mask", None)

                obs_embeds = self.bridge_vision_model(*obs_input.values())
                if self.action_only_one_obs:
                    obs_embeds = obs_embeds.reshape(batch_size, -1, self.num_bridge_tokens, obs_embeds.shape[-1])
                    obs_embeds = obs_embeds[:, -1]
                else:
                    obs_embeds = obs_embeds.reshape(batch_size, -1, obs_embeds.shape[-1])
                obs_tokens_per_img = self.num_bridge_tokens
                num_views = obs_embeds.shape[1] // obs_tokens_per_img

            if self.bridge_loss_type == "cosine":
                obs_embeds = F.normalize(obs_embeds, p=2, dim=-1)

            if self.bridge_loss_type == "cosine":
                backbone_outputs['backbone_features'] = F.normalize(backbone_outputs['backbone_features'], p=2, dim=-1)
                projected_feat_action = F.normalize(projected_feat_action, p=2, dim=-1) # synchronize normalization

            goal_image_embeds = None
            if self.compute_bridge_loss or self.bridge_type == "golden":
                if self.goal_image_type == "future":
                    has_dino_goal = "dino_goal_images" in backbone_inputs
                    has_goal_pixels = any(
                        k.startswith("goal_image_") for k in backbone_inputs
                    )
                    if self.use_dino_goal and has_dino_goal:
                        dino_goal_img = backbone_inputs['dino_goal_images'] # [Batch, 1, C, H, W]
                        B, V_goal, C, H, W = dino_goal_img.shape
                        
                        goal_img_flat = dino_goal_img.view(B * V_goal, C, H, W)
                        # Extract and downsample to 64 tokens
                        raw_dino_goal = self._extract_dino(goal_img_flat, downsample_to_n=self.num_bridge_tokens)
                        
                        goal_image_embeds = raw_dino_goal.view(B, V_goal * self.num_bridge_tokens, -1)
                    elif (not self.use_dino_goal) and has_goal_pixels:
                        goal_image_prefix = "goal_image_"
                        goal_image_input = {
                            k.removeprefix(goal_image_prefix): v
                            for k, v in backbone_inputs.items()
                            if k.startswith(goal_image_prefix)
                        }
                        goal_image_embeds = self._run_bridge_vision(
                            self.bridge_goal_model, goal_image_input, batch_size
                        )
                    # else: online/action_mode without future GT frame → leave None
                    # and fall back to predicted bridge below.

                    if goal_image_embeds is not None and self.bridge_loss_type == "cosine":
                        goal_image_embeds = F.normalize(goal_image_embeds, p=2, dim=-1)

                elif self.goal_image_type == "current":
                    goal_image_embeds = obs_embeds
                else:
                    raise NotImplementedError

            if self.compute_bridge_loss and goal_image_embeds is not None:
                if self.bridge_loss_type == "mse":
                    bridge_loss = F.mse_loss(backbone_outputs['backbone_features'], goal_image_embeds)
                elif self.bridge_loss_type == "cosine":
                    cos_sim = (backbone_outputs['backbone_features'] * goal_image_embeds).sum(dim=-1)
                    bridge_loss = 1 - cos_sim.mean()
                elif self.bridge_loss_type == "mse_cosine":
                    mse_loss = F.mse_loss(backbone_outputs['backbone_features'], goal_image_embeds)
                    cosine_loss = 1 - F.cosine_similarity(backbone_outputs['backbone_features'], goal_image_embeds, dim=-1).mean()
                    bridge_loss = mse_loss + cosine_loss
                    output_dict['mse_loss'] = mse_loss
                    output_dict['cosine_loss'] = cosine_loss
                else:
                    raise NotImplementedError
                output_dict['bridge_loss'] = bridge_loss

            if "end2end" in self.bridge_type:
                # Now use features dedicated to Action instead
                goal_image_embeds = projected_feat_action
            elif self.bridge_type == "golden":
                if self.goal_image_type == "future" and "act_goal_image_pixel_values" in backbone_inputs:
                    if self.use_dino_goal:
                        dino_goal_img = backbone_inputs['dino_act_goal_images'] # [Batch, 1, C, H, W]
                        B, V_goal, C, H, W = dino_goal_img.shape
                        
                        goal_img_flat = dino_goal_img.view(B * V_goal, C, H, W)
                        # Extract and downsample to 64 tokens
                        raw_dino_goal = self._extract_dino(goal_img_flat, downsample_to_n=self.num_bridge_tokens)
                        
                        goal_image_embeds = raw_dino_goal.view(B, V_goal * self.num_bridge_tokens, -1)
                    
                    else:
                        goal_image_prefix = "act_goal_image_"
                        goal_image_input = {
                            k.removeprefix(goal_image_prefix): v
                            for k, v in backbone_inputs.items()
                            if k.startswith(goal_image_prefix)
                        }
                        goal_image_embeds = self._run_bridge_vision(
                            self.bridge_goal_model, goal_image_input, batch_size
                        )
                        
                    if self.bridge_loss_type == "cosine":
                        goal_image_embeds = F.normalize(goal_image_embeds, p=2, dim=-1)
                elif goal_image_embeds is None:
                    # Online sim / action_mode: no future GT → predicted visual foresight
                    goal_image_embeds = projected_feat_action

                if self.training and self.noise_tau > 0:
                    noise_sigma = self.noise_tau * torch.rand((goal_image_embeds.size(0),) + (1,) * (len(goal_image_embeds.shape) - 1), device=goal_image_embeds.device)
                    noise = torch.randn_like(goal_image_embeds)
                    if self.reweight_noise:
                        mean, std = compute_global_mean_std_patch(goal_image_embeds)
                        noise = noise * std
                    noise = noise_sigma * noise
                    goal_image_embeds = goal_image_embeds + noise
            else:
                raise NotImplementedError

            if self.bridge_cross_attention is not None:
                goal_image_embeds = self.bridge_cross_attention(
                    hidden_states=obs_embeds,
                    encoder_hidden_states=goal_image_embeds,
                )

            if self.training and self.p_uncond is not None:
                if not self.null_initialized:
                    with torch.no_grad():
                        # 1. Compute local batch mean on current GPU
                        local_mean = goal_image_embeds.mean(dim=0, keepdim=True) # [1, L, D]
                        
                        # 2. Global sum
                        dist.all_reduce(local_mean, op=dist.ReduceOp.SUM)
                        
                        # 3. Divide by total number of GPUs to get global mean
                        world_size = dist.get_world_size()
                        global_mean = local_mean / world_size
                        
                        # 4. Assign
                        self.null_subgoal_embed.data.copy_(global_mean)
                        self.null_initialized.fill_(True)

                # Randomly generate mask, B is batch_size
                mask = torch.bernoulli(torch.full((batch_size, 1, 1), self.p_uncond)).to(goal_image_embeds.device)
                # If mask=1, use null_subgoal_embed; if mask=0, use normal goal_image_embeds
                goal_image_embeds = (1 - mask) * goal_image_embeds + mask * self.null_subgoal_embed

            # Construct negative samples for discriminative task (only during training when enabled)
            use_matching = self.config.bridge_cfg.get('use_matching_loss', False)
            if self.training and use_matching:
                assert self.p_uncond is None
                batch_size = goal_image_embeds.shape[0]
                assert batch_size > 1
                # Use roll to construct misaligned negative sample indices: [0,1,2,3] -> [3,0,1,2]
                idx = torch.arange(batch_size, device=goal_image_embeds.device).roll(shifts=1, dims=0)
                neg_goal_embeds = goal_image_embeds[idx]
                
                # Construct double-batch features
                # First half is positive samples (Obs_i, Goal_i), second half is negative samples (Obs_i, Goal_j)
                goal_image_embeds = torch.cat([goal_image_embeds, neg_goal_embeds], dim=0)
                obs_embeds = torch.cat([obs_embeds, obs_embeds], dim=0)
                
                # Construct labels: 1 for match, 0 for mismatch
                match_labels = torch.cat([
                    torch.ones(batch_size, 1, device=goal_image_embeds.device),
                    torch.zeros(batch_size, 1, device=goal_image_embeds.device)
                ], dim=0)
                action_inputs["match_labels"] = match_labels
                
                # Important: must also double the other items in action_inputs
                for k in ["state", "action", "action_mask", "embodiment_id",
                          "state_future", "state_future_mask"]:
                    if k in action_inputs:
                        action_inputs[k] = torch.cat([action_inputs[k], action_inputs[k]], dim=0)
                if proprio_foresight_tokens is not None:
                    proprio_foresight_tokens = torch.cat(
                        [proprio_foresight_tokens, proprio_foresight_tokens], dim=0
                    )

            if action_mode and guidance_scale > 1.0 and self.p_uncond is not None:
                # Construct double batch: [Cond, Uncond]
                uncond_goal = self.null_subgoal_embed.expand(batch_size, -1, -1)
                goal_image_embeds = torch.cat([goal_image_embeds, uncond_goal], dim=0)
                obs_embeds = torch.cat([obs_embeds, obs_embeds], dim=0)
                if proprio_foresight_tokens is not None:
                    # SF-CFG: uncond arm uses learnable null SF (not a copy of cond SF)
                    if hasattr(self, "null_sf_embed"):
                        uncond_sf = self.null_sf_embed.to(
                            dtype=proprio_foresight_tokens.dtype,
                            device=proprio_foresight_tokens.device,
                        ).expand_as(proprio_foresight_tokens)
                    else:
                        uncond_sf = proprio_foresight_tokens
                    proprio_foresight_tokens = torch.cat(
                        [proprio_foresight_tokens, uncond_sf], dim=0
                    )

            if self.use_image_type_embedding:
                if not self.omit_image_type_embedding_for_goal:
                    goal_image_embeds = goal_image_embeds + self.image_type_embedding.weight[0]

                # Add multi-view encoding in a loop (Index 1, 2...)
                for i in range(num_views):
                    start_i, end_i = i * obs_tokens_per_img, (i + 1) * obs_tokens_per_img
                    obs_embeds[:, start_i:end_i] += self.image_type_embedding.weight[i+1]

            if proprio_foresight_tokens is not None:
                proprio_foresight_tokens = proprio_foresight_tokens.to(
                    dtype=goal_image_embeds.dtype, device=goal_image_embeds.device
                )
                bridge_features = torch.cat(
                    [goal_image_embeds, obs_embeds, proprio_foresight_tokens], dim=1
                )
            else:
                bridge_features = torch.cat([goal_image_embeds, obs_embeds], dim=1)
            bridge_attention_mask = torch.ones(bridge_features.shape[:-1], device=bridge_features.device)

            bridge_outputs = BatchFeature(
                data={
                    "backbone_features": bridge_features,
                    "backbone_attention_mask": bridge_attention_mask
                }
            )
        else:
            bridge_outputs = backbone_outputs

        if not action_mode:
            action_head_outputs = self.action_head(bridge_outputs, action_inputs)
            self.validate_data(action_head_outputs, backbone_outputs, is_training=True)

            output_dict['action_loss'] = action_head_outputs['loss']
            if "match_loss" in action_head_outputs:
                output_dict['match_loss'] = action_head_outputs['match_loss']
                output_dict['raw_action_loss'] = action_head_outputs['action_loss']

            # state_future_loss already computed earlier (before matching/CFG doubles)

            action_loss = action_head_outputs['loss']
            bridge_loss = output_dict.get('bridge_loss')
            sf_loss = output_dict.get('state_future_loss')
            w_f = float(getattr(self, "state_future_loss_weight", 1.0))
            combine_mode = getattr(self, "loss_combine_mode", "additive")

            # Bridge weight (1.0 unless decay schedule is configured)
            bridge_w = 1.0
            if bridge_loss is not None:
                bridge_cfg = self.config.bridge_cfg
                if bridge_cfg.get('bridge_loss_decay_steps', None) is not None:
                    if self.training:
                        self.bridge_training_steps += 1
                    start_w = bridge_cfg.get('bridge_loss_start_w', 1.0)
                    end_w = bridge_cfg.get('bridge_loss_end_w', 0.1)
                    decay_steps = bridge_cfg['bridge_loss_decay_steps']
                    progress = min(1.0, self.bridge_training_steps.item() / max(1, decay_steps))
                    bridge_w = start_w - (start_w - end_w) * progress
                    output_dict['bridge_w'] = torch.tensor(bridge_w).to(action_loss.device)

            if combine_mode == "additive":
                # Teacher SF-weight debug: L = action + bridge + w·sf
                loss = action_loss
                if bridge_loss is not None:
                    loss = loss + bridge_w * bridge_loss
                if sf_loss is not None:
                    loss = loss + w_f * sf_loss
                # Effective coeffs on (action, bridge, sf) before any averaging
                output_dict['loss_w_action'] = torch.tensor(1.0).to(action_loss.device)
                output_dict['loss_w_bridge'] = torch.tensor(
                    bridge_w if bridge_loss is not None else 0.0
                ).to(action_loss.device)
                output_dict['loss_w_sf'] = torch.tensor(
                    w_f if sf_loss is not None else 0.0
                ).to(action_loss.device)
            else:
                # legacy_avg: ((a + w_b·b)/(1+w_b) + w_f·sf)/(1+w_f)
                # with w_b=1 → (a+b)/2 when no decay schedule
                loss = action_loss
                if bridge_loss is not None:
                    loss = (loss + bridge_w * bridge_loss) / (1.0 + bridge_w)
                if sf_loss is not None:
                    loss = (loss + w_f * sf_loss) / (1.0 + w_f)
                # Log effective share of each term in the final scalar (approx)
                # After (a+b)/2 then /(1+w): action gets 1/(2(1+w)), bridge 1/(2(1+w)), sf w/(1+w)
                denom_b = 1.0 + (bridge_w if bridge_loss is not None else 0.0)
                denom_f = 1.0 + (w_f if sf_loss is not None else 0.0)
                w_a_eff = (1.0 / denom_b) / denom_f if bridge_loss is not None else (1.0 / denom_f)
                w_b_eff = (bridge_w / denom_b) / denom_f if bridge_loss is not None else 0.0
                w_f_eff = w_f / denom_f if sf_loss is not None else 0.0
                output_dict['loss_w_action'] = torch.tensor(w_a_eff).to(action_loss.device)
                output_dict['loss_w_bridge'] = torch.tensor(w_b_eff).to(action_loss.device)
                output_dict['loss_w_sf'] = torch.tensor(w_f_eff).to(action_loss.device)

            output_dict['loss_combine_mode'] = combine_mode
            output_dict['loss'] = loss
            return BatchFeature(data=output_dict)

        else:
            action_head_outputs = self.action_head.get_action(bridge_outputs, action_inputs, guidance_scale=guidance_scale)
            self.validate_data(action_head_outputs, backbone_outputs, is_training=False)

            action_head_outputs.update(output_dict)
            action_head_outputs["backbone_features"] = raw_backbone_features
            # Inference only. Training loss must not see this tensor (.item() logging).
            if pred_state_future is not None:
                action_head_outputs["pred_state_future"] = pred_state_future
            return action_head_outputs

    def _run_bridge_vision(self, vision_model, inputs_dict, batch_size: int):
        """Call Qwen vision tower with processor keys mapped to (hidden_states, grid_thw)."""
        if not inputs_dict:
            raise ValueError("bridge vision inputs_dict is empty")
        if "pixel_values" in inputs_dict and "image_grid_thw" in inputs_dict:
            out = vision_model(
                hidden_states=inputs_dict["pixel_values"],
                grid_thw=inputs_dict["image_grid_thw"],
            )
        elif "hidden_states" in inputs_dict and "grid_thw" in inputs_dict:
            out = vision_model(
                hidden_states=inputs_dict["hidden_states"],
                grid_thw=inputs_dict["grid_thw"],
            )
        else:
            # Legacy: rely on insertion order of values()
            out = vision_model(*inputs_dict.values())
        return out.reshape(batch_size, -1, out.shape[-1])

    def _has_state_future_module(self) -> bool:
        return hasattr(self, "state_future_decoder") or hasattr(self, "state_future_projector")

    def set_sf_eval_mode(self, mode: str = "correct", bank_max: int = 64) -> None:
        """Set deterministic online-SR SF ablation mode.

        Modes (eval / action_mode):
          - correct: use predicted SF cond as-is (default)
          - zero:    replace SF cond with learnable null_sf_embed (or zeros)
          - swap:    replace with a past SF from the token bank (wrong foresight).
                     At B>1 also supports batch roll. Bank empty → null fallback.

        Train-time stochastic p_sf_drop / p_sf_swap are unchanged and only run
        when self.training is True.
        """
        m = (mode or "correct").strip().lower()
        aliases = {
            "correct": "correct",
            "none": "correct",
            "identity": "correct",
            "zero": "zero",
            "null": "zero",
            "drop": "zero",
            "swap": "swap",
            "shuffle": "swap",
        }
        if m not in aliases:
            raise ValueError(
                f"sf_eval_mode={mode!r}; expected one of "
                f"{sorted(set(aliases.values()))} (aliases: {sorted(aliases)})"
            )
        self.sf_eval_mode = aliases[m]
        self._sf_token_bank = []
        self._sf_token_bank_max = max(1, int(bank_max))
        print(
            f"SF eval mode: {self.sf_eval_mode} "
            f"(bank_max={self._sf_token_bank_max}; "
            f"null_sf={'yes' if hasattr(self, 'null_sf_embed') else 'no'})"
        )

    def _null_sf_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "null_sf_embed"):
            return self.null_sf_embed.to(
                dtype=tokens.dtype, device=tokens.device
            ).expand_as(tokens)
        return torch.zeros_like(tokens)

    def _push_sf_token_bank(self, tokens: torch.Tensor) -> None:
        """Store detached per-sample SF cond for later intent-swap."""
        if tokens is None:
            return
        with torch.no_grad():
            for i in range(tokens.shape[0]):
                self._sf_token_bank.append(tokens[i].detach().cpu().clone())
        overflow = len(self._sf_token_bank) - int(self._sf_token_bank_max)
        if overflow > 0:
            del self._sf_token_bank[:overflow]

    def _sample_sf_token_bank(self, tokens: torch.Tensor) -> torch.Tensor | None:
        """Sample wrong SF from bank; shape-matched to tokens. None if bank empty."""
        if not self._sf_token_bank:
            return None
        B = tokens.shape[0]
        out = []
        n_bank = len(self._sf_token_bank)
        for _ in range(B):
            idx = int(torch.randint(0, n_bank, (1,)).item())
            sample = self._sf_token_bank[idx].to(
                device=tokens.device, dtype=tokens.dtype
            )
            if sample.shape != tokens.shape[1:]:
                # Shape drift (e.g. horizon change) — skip this sample
                return None
            out.append(sample)
        return torch.stack(out, dim=0)

    def _apply_sf_eval_mode(self, proprio_foresight_tokens: torch.Tensor) -> torch.Tensor:
        """Deterministic online-SR SF ablations (correct / zero / swap)."""
        tokens = proprio_foresight_tokens
        mode = getattr(self, "sf_eval_mode", "correct") or "correct"
        if mode == "correct":
            self._push_sf_token_bank(tokens)
            return tokens
        if mode == "zero":
            # Do not push null into the bank — keep bank for optional later swap runs
            return self._null_sf_tokens(tokens)
        if mode == "swap":
            B = tokens.shape[0]
            if B > 1:
                wrong = tokens.roll(1, dims=0)
            else:
                wrong = self._sample_sf_token_bank(tokens)
                if wrong is None:
                    wrong = self._null_sf_tokens(tokens)
            # Push the *correct* current tokens so the bank holds real foresight
            self._push_sf_token_bank(tokens)
            return wrong
        return tokens

    def _apply_sf_cond_ablations(self, proprio_foresight_tokens: torch.Tensor) -> torch.Tensor:
        """SF cond ablations: train-time stochastic drop/swap; eval-time sf_eval_mode."""
        if proprio_foresight_tokens is None:
            return proprio_foresight_tokens
        tokens = proprio_foresight_tokens
        if self.training:
            B = tokens.shape[0]
            if self.p_sf_drop > 0.0 and hasattr(self, "null_sf_embed"):
                mask = torch.bernoulli(
                    torch.full((B, 1, 1), self.p_sf_drop, device=tokens.device)
                ).to(dtype=tokens.dtype)
                null_feat = self.null_sf_embed.to(
                    dtype=tokens.dtype, device=tokens.device
                ).expand_as(tokens)
                tokens = (1.0 - mask) * tokens + mask * null_feat
            if self.p_sf_swap > 0.0 and B > 1:
                mask = torch.bernoulli(
                    torch.full((B, 1, 1), self.p_sf_swap, device=tokens.device)
                ).to(dtype=tokens.dtype)
                swapped = tokens.roll(1, dims=0)
                tokens = (1.0 - mask) * tokens + mask * swapped
            return tokens
        # Eval / online: deterministic modes for SR (correct / zero / swap)
        return self._apply_sf_eval_mode(tokens)

    def _predict_state_future(self, raw_backbone_feat, batch_size, return_hidden: bool = False):
        """Decode SF targets from raw bridge tokens. None if the aux head is off.

        If return_hidden=True (query_ca only), returns (pred, mid_hidden) where
        mid_hidden is [B, H+1, hidden] before the final proj to D_sf.
        """
        if not (self.use_bridge and self.use_state_future):
            return (None, None) if return_hidden else None
        fut_len = self.state_future_horizon + 1
        decoder_type = getattr(self, "state_future_decoder_type", "query_ca")
        if decoder_type == "query_ca" and hasattr(self, "state_future_decoder"):
            if return_hidden:
                pred, mid_hidden = self.state_future_decoder(
                    raw_backbone_feat, return_hidden=True
                )
            else:
                pred = self.state_future_decoder(raw_backbone_feat, return_hidden=False)
                mid_hidden = None
            if pred.shape != (batch_size, fut_len, self.state_dim_for_future):
                raise ValueError(
                    f"StateFutureDecoder output {tuple(pred.shape)} != "
                    f"({batch_size}, {fut_len}, {self.state_dim_for_future})"
                )
            if return_hidden:
                if mid_hidden.shape[:2] != (batch_size, fut_len):
                    raise ValueError(
                        f"SF mid_hidden shape {tuple(mid_hidden.shape)} incompatible "
                        f"with batch={batch_size} frames={fut_len}"
                    )
                return pred, mid_hidden
            return pred
        if hasattr(self, "state_future_projector"):
            if return_hidden:
                raise ValueError(
                    "return_hidden=True is not supported for mean_pool SF decoder"
                )
            pooled = raw_backbone_feat.mean(dim=1)
            pred_flat = self.state_future_projector(pooled)
            return pred_flat.view(batch_size, fut_len, self.state_dim_for_future)
        return (None, None) if return_hidden else None

    def get_action(
        self,
        inputs: dict,
        backbone_features=None
    ) -> BatchFeature:
        return self.forward(inputs=inputs, action_mode=True, backbone_features=backbone_features)

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        for key in ["raw_obs_images", "raw_goal_images", "raw_act_goal_images"]:
            if key in inputs and isinstance(inputs[key], list):
                tensors = [TF.to_tensor(img) for img in inputs[key]]
                inputs[key] = torch.stack(tensors).to(self.device, dtype=torch.float32)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    def state_dict(self, *args, **kwargs):
        full_dict = super().state_dict(*args, **kwargs)
        if getattr(self, '_shared_bridge_models', False):
            full_dict = {k: v for k, v in full_dict.items() if not k.startswith("bridge_goal_model.")}
        return full_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        goal_prefix = prefix + "bridge_goal_model."
        has_goal_keys = any(k.startswith(goal_prefix) for k in state_dict)
        if not has_goal_keys:
            missing_keys[:] = [k for k in missing_keys if not k.startswith(goal_prefix)]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, resume_pretrained_option: str="all", **kwargs):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        tune_visual = kwargs.pop("tune_visual", model_config.backbone_cfg['tune_visual'])
        tune_llm = kwargs.pop("tune_llm", model_config.backbone_cfg['tune_llm'])
        tune_bridge_embedding = kwargs.pop("tune_bridge_embedding", model_config.backbone_cfg['tune_bridge_embedding'])
        tokenizer_len = kwargs.pop("tokenizer_len", None)
        tune_projector = kwargs.pop("tune_projector", model_config.action_head_cfg['tune_projector'])
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", model_config.action_head_cfg['tune_diffusion_model'])

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune backbone bridge embedding: {tune_bridge_embedding}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        try:
            bridge_type = kwargs.pop("bridge_type", model_config.bridge_cfg.get('bridge_type', "end2end"))
            compute_bridge_loss = kwargs.pop("compute_bridge_loss", model_config.bridge_cfg.get('compute_bridge_loss', False))
            goal_image_type = kwargs.pop("goal_image_type", model_config.bridge_cfg.get('goal_image_type', 'future'))
            bridge_loss_type = kwargs.pop("bridge_loss_type", model_config.bridge_cfg.get('bridge_loss_type', 'mse'))
            tune_all_llm_embedding = kwargs.pop("tune_all_llm_embedding", model_config.bridge_cfg.get('tune_all_llm_embedding', False))
            use_image_type_embedding = kwargs.pop("use_image_type_embedding", model_config.bridge_cfg.get('use_image_type_embedding', False))
            omit_image_type_embedding_for_goal = kwargs.pop("omit_image_type_embedding_for_goal", model_config.bridge_cfg.get('omit_image_type_embedding_for_goal', False))
            action_only_one_obs = kwargs.pop("action_only_one_obs", model_config.bridge_cfg.get('action_only_one_obs', False))
            noise_tau = kwargs.pop("noise_tau", model_config.bridge_cfg.get('noise_tau', 0))
            reweight_noise = kwargs.pop("reweight_noise", model_config.bridge_cfg.get('reweight_noise', None))
            unified_embodiment_id = kwargs.pop("unified_embodiment_id", model_config.bridge_cfg.get('unified_embodiment_id', None))

            bridge_loss_end_w = kwargs.pop("bridge_loss_end_w", model_config.bridge_cfg.get('bridge_loss_end_w', None))
            bridge_loss_decay_steps = kwargs.pop("bridge_loss_decay_steps", model_config.bridge_cfg.get('bridge_loss_decay_steps', None))
            use_separate_projector_for_loss = kwargs.pop("use_separate_projector_for_loss", model_config.bridge_cfg.get('use_separate_projector_for_loss', False))
            vlm_small_lr = kwargs.pop("vlm_small_lr", model_config.bridge_cfg.get('vlm_small_lr', False))
            matching_coeff = kwargs.pop("matching_coeff", model_config.action_head_cfg.get('matching_coeff', None))
            use_state_future = kwargs.pop("use_state_future", model_config.bridge_cfg.get('use_state_future', False))
            compute_state_future_loss = kwargs.pop(
                "compute_state_future_loss", model_config.bridge_cfg.get('compute_state_future_loss', False)
            )
            state_future_loss_weight = kwargs.pop(
                "state_future_loss_weight", model_config.bridge_cfg.get('state_future_loss_weight', 1.0)
            )
            state_future_horizon = kwargs.pop(
                "state_future_horizon", model_config.bridge_cfg.get('state_future_horizon', None)
            )
            tune_state_future_projector = kwargs.pop(
                "tune_state_future_projector", model_config.bridge_cfg.get('tune_state_future_projector', True)
            )
            state_future_decoder = kwargs.pop(
                "state_future_decoder",
                model_config.bridge_cfg.get("state_future_decoder", "query_ca"),
            )
            state_future_dim = kwargs.pop(
                "state_future_dim", model_config.bridge_cfg.get("state_future_dim", None)
            )
            state_future_num_heads = kwargs.pop(
                "state_future_num_heads",
                model_config.bridge_cfg.get(
                    "state_future_num_heads", DEFAULT_STATE_FUTURE_NUM_HEADS
                ),
            )
            loss_combine_mode = kwargs.pop(
                "loss_combine_mode",
                model_config.bridge_cfg.get("loss_combine_mode", "additive"),
            )
            feed_state_future_to_dit = kwargs.pop(
                "feed_state_future_to_dit",
                model_config.bridge_cfg.get("feed_state_future_to_dit", False),
            )
            sf_dit_repr = kwargs.pop(
                "sf_dit_repr",
                model_config.bridge_cfg.get("sf_dit_repr", "shat"),
            )
            sf_tokens_detach = kwargs.pop(
                "sf_tokens_detach",
                model_config.bridge_cfg.get("sf_tokens_detach", False),
            )
            p_sf_drop = kwargs.pop(
                "p_sf_drop",
                model_config.bridge_cfg.get("p_sf_drop", 0.0),
            )
            p_sf_swap = kwargs.pop(
                "p_sf_swap",
                model_config.bridge_cfg.get("p_sf_swap", 0.0),
            )
            use_mot_system1 = kwargs.pop(
                "use_mot_system1",
                model_config.action_head_cfg.get("use_mot_system1", False),
            )
            use_mot_cond_experts = kwargs.pop(
                "use_mot_cond_experts",
                model_config.action_head_cfg.get("use_mot_cond_experts", False),
            )
            mot_include_sf = kwargs.pop(
                "mot_include_sf",
                model_config.action_head_cfg.get("mot_include_sf", True),
            )
            mot_sf_num_tokens = kwargs.pop(
                "mot_sf_num_tokens",
                model_config.action_head_cfg.get("mot_sf_num_tokens", 17),
            )
            mot_num_layers = kwargs.pop("mot_num_layers", None)
        except Exception as e:
            print(kwargs)
            raise e
        print(f"Bridge type: {bridge_type}")
        print(f"Compute bridge loss: {compute_bridge_loss}")
        print(f"Goal image type: {goal_image_type}")
        print(f"Bridge loss type: {bridge_loss_type}")
        print(f"Tune all llm token embeddings: {tune_all_llm_embedding}")
        print(f"Use image type embeddings: {use_image_type_embedding}")
        print(f"Omit image type embeddings for goal images: {omit_image_type_embedding_for_goal}")
        print(f"State future: use={use_state_future} compute_loss={compute_state_future_loss} "
              f"weight={state_future_loss_weight} horizon={state_future_horizon} "
              f"decoder={state_future_decoder} dim={state_future_dim} heads={state_future_num_heads}")
        print(f"Loss combine mode: {loss_combine_mode}")
        print(f"Feed state future to DiT: {feed_state_future_to_dit}")
        print(f"SF DiT repr: {sf_dit_repr}")
        print(
            f"SF cond ablations: detach={sf_tokens_detach} "
            f"p_sf_drop={p_sf_drop} p_sf_swap={p_sf_swap}"
        )
        print(
            f"MoT System-1: use={use_mot_system1} include_sf={mot_include_sf} "
            f"sf_tokens={mot_sf_num_tokens} num_layers={mot_num_layers}"
        )
        print(f"MoT cond experts (Dense CA): use={use_mot_cond_experts}")
        if use_mot_system1 and use_mot_cond_experts:
            raise ValueError(
                "use_mot_system1 and use_mot_cond_experts are mutually exclusive"
            )
        if use_mot_cond_experts and not feed_state_future_to_dit:
            print(
                "WARNING: use_mot_cond_experts=True but feed_state_future_to_dit=False "
                "— SF segment will be empty; vis expert only."
            )
        print(f"Action head using only one obs: {action_only_one_obs}")
        print(f"Noise Tau: {noise_tau}")
        print(f"Reweight Noise: {reweight_noise}")
        print(f"unified_embodiment_id: {unified_embodiment_id}")
        print(f"bridge_loss_end_w: {bridge_loss_end_w}")
        print(f"bridge_loss_decay_steps: {bridge_loss_decay_steps}")
        print(f"use_separate_projector_for_loss: {use_separate_projector_for_loss}")
        print(f"vlm_small_lr: {vlm_small_lr}")
        print(f"matching_coeff: {matching_coeff}")

        select_layer = kwargs.pop("select_layer", None)
        select_layer_for_bridge = kwargs.pop("select_layer_for_bridge", None)

        tune_bridge_visual = kwargs.pop("tune_bridge_visual", model_config.bridge_cfg['tune_bridge_visual'])
        tune_bridge_goal = kwargs.pop("tune_bridge_goal", model_config.bridge_cfg['tune_bridge_goal'])
        assert tune_bridge_visual == False and tune_bridge_goal == False
        tune_image_type_embedding = kwargs.pop("tune_image_type_embedding", model_config.bridge_cfg.get('tune_image_type_embedding', True))
        print(f"Tune bridge vision model: {tune_bridge_visual}")
        print(f"Tune bridge goal model: {tune_bridge_goal}")
        print(f"Tune image type embeddings: {tune_image_type_embedding}")

        use_vl_mask = kwargs.pop("use_vl_mask", model_config.action_head_cfg.get('use_vl_mask', False))
        print(f"Use VL mask: {use_vl_mask}")

        correct_vl_mask = kwargs.pop("correct_vl_mask", model_config.action_head_cfg.get('correct_vl_mask', False))
        print(f"Correct VL mask: {correct_vl_mask}")

        resume_vlm_path = kwargs.pop("resume_vlm_path", model_config.bridge_cfg.get('resume_vlm_path', None))
        resume_action_head_path = kwargs.pop("resume_action_head_path", model_config.bridge_cfg.get('resume_action_head_path', None))
        print(f"resume_vlm_path: {resume_vlm_path}")
        print(f"resume_action_head_path: {resume_action_head_path}")

        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        customized_kwargs = {
            "tokenizer_len": tokenizer_len,
            "bridge_type": bridge_type,
            "compute_bridge_loss": compute_bridge_loss,
            "select_layer": select_layer,
            "goal_image_type": goal_image_type,
            "bridge_loss_type": bridge_loss_type,
            "use_image_type_embedding": use_image_type_embedding,
            "use_vl_mask": use_vl_mask,
            "correct_vl_mask": correct_vl_mask,
            "action_only_one_obs": action_only_one_obs,
            "noise_tau": noise_tau,
            "omit_image_type_embedding_for_goal": omit_image_type_embedding_for_goal,
            "reweight_noise": reweight_noise,
            "unified_embodiment_id": unified_embodiment_id,
            "use_separate_projector_for_loss": use_separate_projector_for_loss,
            "vlm_small_lr": vlm_small_lr,
            "select_layer_for_bridge": select_layer_for_bridge,
            "matching_coeff": matching_coeff,
            "use_state_future": use_state_future,
            "compute_state_future_loss": compute_state_future_loss,
            "state_future_loss_weight": state_future_loss_weight,
            "state_future_horizon": state_future_horizon,
            "tune_state_future_projector": tune_state_future_projector,
            "state_future_decoder": state_future_decoder,
            "state_future_dim": state_future_dim,
            "state_future_num_heads": state_future_num_heads,
            "loss_combine_mode": loss_combine_mode,
            "feed_state_future_to_dit": feed_state_future_to_dit,
            "sf_dit_repr": sf_dit_repr,
            "sf_tokens_detach": sf_tokens_detach,
            "p_sf_drop": p_sf_drop,
            "p_sf_swap": p_sf_swap,
            "use_mot_system1": use_mot_system1,
            "use_mot_cond_experts": use_mot_cond_experts,
            "mot_include_sf": mot_include_sf,
            "mot_sf_num_tokens": mot_sf_num_tokens,
            "mot_num_layers": mot_num_layers,
        }

        kwargs["output_loading_info"] = True

        try:
            pretrained_model, loading_info = super().from_pretrained(
                local_model_path, local_model_path=local_model_path, 
                **customized_kwargs,
                **kwargs
            )

            if resume_pretrained_option != "all" or resume_vlm_path or resume_action_head_path:
                assert not (resume_vlm_path and resume_action_head_path)
                # The logic here is not compatible with bridge_projector_for_loss
                if resume_vlm_path is not None:
                    resume_model, loading_info = super().from_pretrained(
                        resume_vlm_path, local_model_path=resume_vlm_path, 
                        **customized_kwargs,
                        **kwargs
                    )
                    pretrained_model.backbone = resume_model.backbone
                    pretrained_model.bridge_projector = resume_model.bridge_projector

                elif resume_action_head_path is not None:
                    resume_model, loading_info = super().from_pretrained(
                        resume_action_head_path, local_model_path=resume_action_head_path, 
                        **customized_kwargs,
                        **kwargs
                    )
                    pretrained_model.action_head = resume_model.action_head
                    if hasattr(pretrained_model, "image_type_embedding"):
                        pretrained_model.image_type_embedding = resume_model.image_type_embedding

                else:
                    config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
                    scratch_model = cls(
                        config, local_model_path=None, 
                        **customized_kwargs,
                    )
                    print(f"Resuming only {resume_pretrained_option} ...")
                    if resume_pretrained_option == "backbone+bridge_projector":
                        pretrained_model.action_head = scratch_model.action_head
                        if hasattr(pretrained_model, "image_type_embedding"):
                            pretrained_model.image_type_embedding = scratch_model.image_type_embedding
                    elif resume_pretrained_option == "action_head":
                        pretrained_model.backbone = scratch_model.backbone
                        pretrained_model.bridge_projector = scratch_model.bridge_projector
                    else:
                        raise NotImplementedError

        except Exception as e:
            if kwargs.get("ignore_mismatched_sizes"):
                raise
            print(f"Load pretrained model error!!!! {e}")
            config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
            pretrained_model = cls(
                config, local_model_path=None, 
                **customized_kwargs,
            )
            loading_info = None

        if use_separate_projector_for_loss and (loading_info is not None):
            # Check if missing keys include bridge_projector_for_loss
            is_missing = any("bridge_projector_for_loss" in k for k in loading_info["missing_keys"])
            
            if is_missing:
                print("### [Init] bridge_projector_for_loss not found in checkpoint. Copying from bridge_projector.")
                pretrained_model.bridge_projector_for_loss = deepcopy(pretrained_model.bridge_projector)
            else:
                print("### [Resume] bridge_projector_for_loss loaded from checkpoint. Skipping deepcopy.")

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm, 
            tune_bridge_embedding=tune_bridge_embedding,
            tokenizer_len=tokenizer_len,
            tune_all_llm_embedding=tune_all_llm_embedding,
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )

        if pretrained_model.use_bridge:
            pretrained_model.set_trainable_parameters(
                tune_bridge_visual=tune_bridge_visual,
                tune_bridge_goal=tune_bridge_goal,
                tune_image_type_embedding=tune_image_type_embedding
            )

        pretrained_model.config.backbone_cfg['tune_visual'] = tune_visual
        pretrained_model.config.backbone_cfg['tune_llm'] = tune_llm
        pretrained_model.config.backbone_cfg['tune_bridge_embedding'] = tune_bridge_embedding
        pretrained_model.config.action_head_cfg['tune_projector'] = tune_projector
        pretrained_model.config.action_head_cfg['tune_diffusion_model'] = tune_diffusion_model
        pretrained_model.config.bridge_cfg['tune_bridge_visual'] = tune_bridge_visual
        pretrained_model.config.bridge_cfg['tune_bridge_goal'] = tune_bridge_goal
        pretrained_model.config.bridge_cfg['tokenizer_len'] = tokenizer_len
        pretrained_model.config.bridge_cfg['bridge_type'] = bridge_type
        pretrained_model.config.bridge_cfg['compute_bridge_loss'] = compute_bridge_loss
        pretrained_model.config.bridge_cfg['goal_image_type'] = goal_image_type
        pretrained_model.config.bridge_cfg['bridge_loss_type'] = bridge_loss_type
        pretrained_model.config.backbone_cfg['tune_all_llm_embedding'] = tune_all_llm_embedding
        pretrained_model.config.bridge_cfg['use_image_type_embedding'] = use_image_type_embedding
        pretrained_model.config.bridge_cfg['action_only_one_obs'] = action_only_one_obs
        pretrained_model.config.bridge_cfg['noise_tau'] = noise_tau
        pretrained_model.config.bridge_cfg['reweight_noise'] = reweight_noise
        pretrained_model.config.bridge_cfg['omit_image_type_embedding_for_goal'] = omit_image_type_embedding_for_goal
        pretrained_model.config.bridge_cfg['tune_image_type_embedding'] = tune_image_type_embedding
        pretrained_model.config.bridge_cfg['unified_embodiment_id'] = unified_embodiment_id
        pretrained_model.config.bridge_cfg['bridge_loss_end_w'] = bridge_loss_end_w
        pretrained_model.config.bridge_cfg['bridge_loss_decay_steps'] = bridge_loss_decay_steps
        pretrained_model.config.bridge_cfg['use_separate_projector_for_loss'] = use_separate_projector_for_loss
        pretrained_model.config.bridge_cfg['vlm_small_lr'] = vlm_small_lr
        # Persist state_future knobs so from_pretrained can recreate the projector
        pretrained_model.config.bridge_cfg['use_state_future'] = use_state_future
        pretrained_model.config.bridge_cfg['compute_state_future_loss'] = compute_state_future_loss
        pretrained_model.config.bridge_cfg['state_future_loss_weight'] = state_future_loss_weight
        pretrained_model.config.bridge_cfg['state_future_horizon'] = state_future_horizon
        pretrained_model.config.bridge_cfg['tune_state_future_projector'] = tune_state_future_projector
        pretrained_model.config.bridge_cfg['state_future_decoder'] = state_future_decoder
        pretrained_model.config.bridge_cfg['state_future_dim'] = (
            pretrained_model.state_dim_for_future if use_state_future else state_future_dim
        )
        pretrained_model.config.bridge_cfg['state_future_num_heads'] = state_future_num_heads
        pretrained_model.config.bridge_cfg['loss_combine_mode'] = loss_combine_mode
        pretrained_model.config.bridge_cfg['feed_state_future_to_dit'] = feed_state_future_to_dit
        pretrained_model.config.bridge_cfg['sf_dit_repr'] = sf_dit_repr
        pretrained_model.config.bridge_cfg['sf_tokens_detach'] = sf_tokens_detach
        pretrained_model.config.bridge_cfg['p_sf_drop'] = p_sf_drop
        pretrained_model.config.bridge_cfg['p_sf_swap'] = p_sf_swap
        pretrained_model.config.action_head_cfg['use_mot_system1'] = use_mot_system1
        pretrained_model.config.action_head_cfg['use_mot_cond_experts'] = use_mot_cond_experts
        pretrained_model.config.action_head_cfg['mot_include_sf'] = mot_include_sf
        pretrained_model.config.action_head_cfg['mot_sf_num_tokens'] = mot_sf_num_tokens
        if use_mot_cond_experts:
            pretrained_model.config.action_head_cfg.setdefault("diffusion_model_cfg", {})
            pretrained_model.config.action_head_cfg["diffusion_model_cfg"] = dict(
                pretrained_model.config.action_head_cfg["diffusion_model_cfg"]
            )
            pretrained_model.config.action_head_cfg["diffusion_model_cfg"][
                "use_mot_cond_experts"
            ] = True

        # Loading diagnostics for state_future recoverability (query_ca or mean_pool)
        if loading_info is not None:
            sf_needles = ("state_future_projector", "state_future_decoder")
            sf_missing = [
                k for k in loading_info.get("missing_keys", [])
                if any(n in k for n in sf_needles)
            ]
            sf_unexpected = [
                k for k in loading_info.get("unexpected_keys", [])
                if any(n in k for n in sf_needles)
            ]
            has_sf_module = pretrained_model._has_state_future_module()
            if use_state_future and not has_sf_module:
                print(
                    "WARNING: use_state_future=True but no state_future decoder/projector "
                    "was created (requires use_bridge=True). Weights in checkpoint would be ignored."
                )
            if use_state_future and has_sf_module and sf_missing:
                print(
                    f"WARNING: SF module created but checkpoint is missing "
                    f"{len(sf_missing)} keys (random init for those params): {sf_missing[:5]}..."
                )
                # transformers (low_cpu_mem_usage) leaves missing keys uninitialized
                # (empty device memory can contain NaN) -> explicitly re-initialize them.
                reinit_failed = []
                for key in sf_missing:
                    mod_path, _, param_name = key.rpartition(".")
                    mod = pretrained_model
                    for part in mod_path.split("."):
                        if part.isdigit() and isinstance(mod, (nn.Sequential, nn.ModuleList)):
                            mod = mod[int(part)]
                        elif hasattr(mod, part):
                            mod = getattr(mod, part)
                        else:
                            mod = None
                            break
                    if mod is None:
                        reinit_failed.append(key)
                        continue
                    with torch.no_grad():
                        if param_name in ("weight", "bias") and hasattr(mod, "reset_parameters"):
                            mod.reset_parameters()
                        else:
                            p = getattr(mod, param_name, None)
                            if isinstance(p, torch.nn.Parameter):
                                nn.init.normal_(p, std=0.02)
                            else:
                                reinit_failed.append(key)
                print(
                    f"[Init] Explicitly re-initialized {len(sf_missing) - len(reinit_failed)} SF params"
                    + (f"; could not init: {reinit_failed}" if reinit_failed else "")
                )

        # Fail-fast audit: trainable params must be finite right after loading.
        bad_params = [
            n for n, p in pretrained_model.named_parameters()
            if p.requires_grad and not torch.isfinite(p).all()
        ]
        if bad_params:
            raise RuntimeError(
                f"Non-finite trainable parameters after loading: {bad_params[:10]}"
            )
            if sf_unexpected:
                print(
                    f"WARNING: checkpoint has {len(sf_unexpected)} unused SF keys "
                    f"(decoder={state_future_decoder}, use_state_future={use_state_future}): "
                    f"{sf_unexpected[:5]}..."
                )

        if matching_coeff is not None:
            if hasattr(pretrained_model, 'config'):
                if 'action_head_cfg' not in pretrained_model.config.__dict__:
                    pretrained_model.config.action_head_cfg['matching_coeff'] = matching_coeff
            if hasattr(pretrained_model, 'action_head'):
                pretrained_model.action_head.matching_coeff = matching_coeff
        
        return pretrained_model

    def set_trainable_parameters(self, tune_bridge_visual: bool, tune_bridge_goal: bool, tune_image_type_embedding: bool):
        self.tune_bridge_visual = tune_bridge_visual
        self.tune_bridge_goal = tune_bridge_goal
        self.tune_image_type_embedding = tune_image_type_embedding

        if hasattr(self, 'dino_model'):
            self.dino_model.requires_grad_(self.tune_bridge_visual)

        if hasattr(self, 'bridge_vision_model') and hasattr(self, 'bridge_goal_model'):
            if not tune_bridge_visual and not tune_bridge_goal:
                del self.bridge_goal_model
                self.bridge_goal_model = self.bridge_vision_model
                self._shared_bridge_models = True
            else:
                self._shared_bridge_models = False
                self.bridge_vision_model.requires_grad_(tune_bridge_visual)
                self.bridge_goal_model.requires_grad_(tune_bridge_goal)

        if self.use_image_type_embedding:
            self.image_type_embedding.requires_grad_(self.tune_image_type_embedding)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if hasattr(self, 'dino_model'):
                self.dino_model.requires_grad_(self.tune_bridge_visual)

            if not self.tune_bridge_visual and hasattr(self, 'bridge_vision_model'):
                self.bridge_vision_model.eval()
            if not getattr(self, '_shared_bridge_models', False):
                if not self.tune_bridge_goal and hasattr(self, 'bridge_goal_model'):
                    self.bridge_goal_model.eval()
            if self.use_image_type_embedding and not self.tune_image_type_embedding:
                self.image_type_embedding.eval()

# register
AutoConfig.register("gr00t_n1_5_dial", GR00T_N1_5_DIAL_Config)
AutoModel.register(GR00T_N1_5_DIAL_Config, GR00T_N1_5_DIAL)
