"""Snippet from policy_dial.py — propose_bon_with_meta.py"""

    def propose_bon_with_meta(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Sample K BoN candidates and return scores + unnormalized action dicts.

        Used by same-state branch probes: caller save/restores sim and executes
        each candidate chunk to compare verifier ranking vs real outcomes.
        """
        if self.bon_k <= 1:
            raise RuntimeError("propose_bon_with_meta requires bon_k > 1")

        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        normalized_input["guidance_scale"] = self._guidance_scale

        s_t = None
        if self._wm_bon is not None:
            s_t = self._wm_bon.extract_s_t_from_normalized(normalized_input)

        def _clone(value):
            if isinstance(value, torch.Tensor):
                return value.clone()
            if isinstance(value, np.ndarray):
                return value.copy()
            if isinstance(value, dict):
                return {k: _clone(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_clone(v) for v in value]
            return value

        if self._wm_bon is not None:
            seed_base = self._wm_bon.next_seed_base(self.bon_seed)
        else:
            self._bon_call_idx += 1
            seed_base = int(self.bon_seed) + 100003 * self._bon_call_idx

        cands = []
        pred_sf = None
        backbone_feat = None  # cand0 full VLM; cand1..K-1 reuse (skip backbone)
        for i in range(self.bon_k):
            torch.manual_seed(seed_base + 10007 * (i + 1))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base + 10007 * (i + 1))
            copied = _clone(normalized_input)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                model_pred = self.model.get_action(
                    copied, backbone_features=backbone_feat
                )
            a = model_pred["action_pred"].float()
            if a.ndim == 3:
                a = a[0]
            cands.append(a.detach())
            if backbone_feat is None and model_pred.get("backbone_features") is not None:
                backbone_feat = model_pred["backbone_features"]
            if pred_sf is None and model_pred.get("pred_state_future") is not None:
                psf = model_pred["pred_state_future"].detach().float()
                if psf.ndim == 3:
                    psf = psf[0]
                pred_sf = psf
        stacked = torch.stack(cands, dim=0)

        scores = None
        scores_pos = None
        scores_rot = None
        best = 0
        if self._wm_bon is not None and s_t is not None and pred_sf is not None:
            from gr00t.model.wm_verifier.action_conditioned_dynamics import POS_ROT_DIM

            wm = self._wm_bon
            k = min(self.bon_k, int(stacked.shape[0]))
            a = stacked[:k, : wm.action_horizon, : wm.action_dim].float().to(wm.device)
            st = s_t.float().to(wm.device)
            if st.ndim == 1:
                st = st.unsqueeze(0)
            st = st[:, :POS_ROT_DIM].expand(k, -1)
            sf = pred_sf.float().to(wm.device)
            if sf.ndim == 3:
                sf = sf[0]
            sf = sf[:, :POS_ROT_DIM].unsqueeze(0).expand(k, -1, -1)
            s_hat = wm.wm(st, a)
            # Per-sample pos/rot distances on score window (same as distance()).
            crit = wm.crit
            sl = crit.score_slice
            p = s_hat[:, sl]
            g = sf[:, sl]
            m = torch.ones(*p.shape[:2], 1, device=p.device, dtype=p.dtype)
            w = m.sum(dim=(1, 2)).clamp_min(1.0)
            std = crit.dim_std()

            def _one(idx: torch.Tensor) -> torch.Tensor:
                gstd = std.index_select(0, idx).view(1, 1, -1)
                err = (p.index_select(-1, idx) - g.index_select(-1, idx)) / gstd
                return (err.square() * m).sum(dim=(1, 2)) / (w * idx.numel())

            d_pos = _one(crit.pos_idx)
            d_rot = _one(crit.rot_idx)
            d_tot = d_pos + d_rot
            scores = (-d_tot).detach().float().cpu().numpy()
            scores_pos = (-d_pos).detach().float().cpu().numpy()
            scores_rot = (-d_rot).detach().float().cpu().numpy()
            best = int(np.argmax(scores))
            s_hat_np = s_hat.detach().float().cpu().numpy()
            pred_sf_np = sf[0].detach().float().cpu().numpy()
            s_t_np = st[0].detach().float().cpu().numpy()
        else:
            best = 0
            s_hat_np = None
            pred_sf_np = None
            s_t_np = s_t.detach().float().cpu().numpy() if s_t is not None else None

        # Random index independent of scores, but tied to the same seed_base
        # so score/random share the identical candidate pool.
        rng = np.random.RandomState(int(seed_base) + 17)
        random_pick = int(rng.randint(0, self.bon_k))

        # Mean-proximity / smoothness diagnostics on normalized actions.
        stacked_np = stacked.detach().float().cpu().numpy()
        mean_a = stacked_np.mean(axis=0, keepdims=True)
        dist_to_mean = np.sqrt(((stacked_np - mean_a) ** 2).mean(axis=(1, 2)))
        # Temporal smoothness: mean L2 of consecutive action deltas.
        if stacked_np.shape[1] > 1:
            deltas = stacked_np[:, 1:] - stacked_np[:, :-1]
            smoothness = np.sqrt((deltas**2).mean(axis=(1, 2)))
        else:
            smoothness = np.zeros((stacked_np.shape[0],), dtype=np.float32)
        # Pairwise candidate dispersion (mean pairwise L2 over flattened chunk).
        flat = stacked_np.reshape(stacked_np.shape[0], -1)
        if flat.shape[0] > 1:
            diffs = flat[:, None, :] - flat[None, :, :]
            pairwise = np.sqrt((diffs**2).sum(axis=-1))
            # upper triangle mean
            iu = np.triu_indices(flat.shape[0], k=1)
            cand_dispersion = float(pairwise[iu].mean()) if iu[0].size else 0.0
        else:
            cand_dispersion = 0.0

        unnorm_list = []
        for i in range(self.bon_k):
            un = self._get_unnormalized_action(stacked[i].unsqueeze(0), normalized_input)
            if not is_batch:
                un = squeeze_dict_values(un)
            unnorm_list.append(un)

        return {
            "best": best,  # score pick
            "random_pick": random_pick,  # independent of scores; same pool
            "scores": scores,
            "scores_pos": scores_pos,
            "scores_rot": scores_rot,
            "dist_to_mean": dist_to_mean,
            "smoothness": smoothness,
            "cand_dispersion": cand_dispersion,
            "actions_norm": stacked_np,
            "actions_unnorm": unnorm_list,
            "seed_base": seed_base,
            "s_t": s_t_np,  # current wrist pos+rot6d (normalized)
            "s_hat": s_hat_np,  # WM dynamics pred [K, T, 18]
            "pred_sf": pred_sf_np,  # s* foresight target [T, 18]
        }
