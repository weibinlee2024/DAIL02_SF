# Extracted detach / drop / swap wiring from scripts/dual_system_train.py
# Live source of truth: scripts/dual_system_train.py

# ---- CLI config fields (L230-L245) ----
    loss_combine_mode: str = "additive"
    # Concat SF into DiT encoder_hidden_states (proprio foresight channel)
    feed_state_future_to_dit: bool = False
    # shat: project Ŝ; mid_hidden: query_ca pre-proj (legacy)
    sf_dit_repr: str = "shat"
    # Teacher ablations: stop-grad SF→DiT; null-drop; batch intent-swap
    sf_tokens_detach: bool = False
    p_sf_drop: float = 0.0
    p_sf_swap: float = 0.0
    # Mixture-of-Transformers System-1
    use_mot_system1: bool = False
    # Dense-CA MoT cond experts (teacher口径); exclusive with use_mot_system1
    use_mot_cond_experts: bool = False
    mot_include_sf: bool = True
    mot_sf_num_tokens: int = 17
    # None → keep diffusion_model_cfg.num_layers; set lower to fit VRAM

# ---- log line (L365-L375) ----
            sf_dim = data_config_cls.get_state_future_dim()
        except Exception:
            sf_dim = 18 if config.state_future_wrist_only else 30
    if getattr(config, "use_state_future", False):
        print(
            f"state_future: decoder={config.state_future_decoder} dim={sf_dim} "
            f"detach={config.sf_tokens_detach} p_drop={config.p_sf_drop} "
            f"p_swap={config.p_sf_swap} "
            f"heads={config.state_future_num_heads} wrist_only={config.state_future_wrist_only}"
        )


# ---- pass into from_pretrained (L425-L440) ----
        tune_state_future_projector=config.tune_state_future_projector,
        state_future_decoder=config.state_future_decoder,
        state_future_dim=sf_dim,
        state_future_num_heads=config.state_future_num_heads,
        loss_combine_mode=config.loss_combine_mode,
        feed_state_future_to_dit=config.feed_state_future_to_dit,
        sf_dit_repr=config.sf_dit_repr,
        sf_tokens_detach=config.sf_tokens_detach,
        p_sf_drop=config.p_sf_drop,
        p_sf_swap=config.p_sf_swap,
        use_mot_system1=config.use_mot_system1,
        use_mot_cond_experts=config.use_mot_cond_experts,
        mot_include_sf=config.mot_include_sf,
        mot_sf_num_tokens=config.mot_sf_num_tokens,
        mot_num_layers=config.mot_num_layers,
    )

