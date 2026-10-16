"""Reference extract: detach / drop / swap code from gr00t/model/gr00t_n1_dial.py
This is NOT a runnable module — see the live file for full context.
Baseline without these blocks: /home/weibin/projects3/DIAL02-upload/gr00t/model/gr00t_n1_dial.py
"""

# ---- ctor kwargs  (gr00t_n1_dial.py L215-L220) ----
        # Ablation: stop-grad Ŝ before state_future_to_cond so action_loss cannot reshape SF decoder
        sf_tokens_detach: bool = False,
        # Train-time dropout: replace SF cond tokens with learnable null (CFG-style)
        p_sf_drop: float = 0.0,
        # Train-time intent-swap: with this prob, roll SF tokens across batch (wrong foresight)
        p_sf_swap: float = 0.0,

# ---- store flags  (gr00t_n1_dial.py L375-L380) ----
            raise ValueError(f"sf_dit_repr={sf_dit_repr!r}; expected 'shat' or 'mid_hidden'")
        self.sf_dit_repr = repr_mode
        self.sf_tokens_detach = bool(sf_tokens_detach)
        self.p_sf_drop = float(p_sf_drop)
        self.p_sf_swap = float(p_sf_swap)
        if not (0.0 <= self.p_sf_drop <= 1.0):

# ---- null_sf_embed + ablation print  (gr00t_n1_dial.py L442-L464) ----
            if self.feed_state_future_to_dit:
                # Type emb to distinguish proprio foresight tokens from visual bridge pack
                self.proprio_foresight_type_emb = nn.Parameter(torch.zeros(1, 1, hidden))
                nn.init.normal_(self.proprio_foresight_type_emb, mean=0.0, std=0.02)
                # Learnable null SF for p_sf_drop / CFG uncond arm / ablation
                self.null_sf_embed = nn.Parameter(torch.zeros(1, 1, hidden))
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

# ---- forward: detach before state_future_to_cond  (gr00t_n1_dial.py L600-L642) ----
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

# ---- CFG uncond uses null_sf_embed  (gr00t_n1_dial.py L870-L890) ----
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

# ---- _apply_sf_cond_ablations  (gr00t_n1_dial.py L1001-L1021) ----
    def _apply_sf_cond_ablations(self, proprio_foresight_tokens: torch.Tensor) -> torch.Tensor:
        """Train-time SF cond dropout / intent-swap (no-op at eval unless called)."""
        if proprio_foresight_tokens is None or not self.training:
            return proprio_foresight_tokens
        tokens = proprio_foresight_tokens
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

# ---- from_pretrained: pop + log  (gr00t_n1_dial.py L1175-L1225) ----
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

# ---- ctor kwargs dict  (gr00t_n1_dial.py L1305-L1315) ----
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

# ---- persist bridge_cfg  (gr00t_n1_dial.py L1438-L1446) ----
        )
        pretrained_model.config.bridge_cfg['state_future_num_heads'] = state_future_num_heads
        pretrained_model.config.bridge_cfg['loss_combine_mode'] = loss_combine_mode
        pretrained_model.config.bridge_cfg['feed_state_future_to_dit'] = feed_state_future_to_dit
        pretrained_model.config.bridge_cfg['sf_dit_repr'] = sf_dit_repr
        pretrained_model.config.bridge_cfg['sf_tokens_detach'] = sf_tokens_detach
        pretrained_model.config.bridge_cfg['p_sf_drop'] = p_sf_drop
        pretrained_model.config.bridge_cfg['p_sf_swap'] = p_sf_swap
        pretrained_model.config.action_head_cfg['use_mot_system1'] = use_mot_system1

