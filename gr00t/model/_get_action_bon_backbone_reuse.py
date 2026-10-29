"""Snippet from policy_dial.py — _get_action_bon_backbone_reuse.py"""

    def _get_action_bon(
        self, normalized_input: Dict[str, Any], s_t: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Sample K action chunks; pick by WM score or random (same candidate seeds)."""

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
            # pred_sf / backbone are obs-conditioned (independent of action noise).
            if backbone_feat is None and model_pred.get("backbone_features") is not None:
                backbone_feat = model_pred["backbone_features"]
            if pred_sf is None and model_pred.get("pred_state_future") is not None:
                psf = model_pred["pred_state_future"].detach().float()
                if psf.ndim == 3:
                    psf = psf[0]
                pred_sf = psf
        stacked = torch.stack(cands, dim=0)  # [K,H,D]

        scores = None
        if self.bon_select == "score":
            if self._wm_bon is None or s_t is None:
                raise RuntimeError("bon_select=score requires WM and s_t")
            if pred_sf is None:
                raise RuntimeError("WM-BoN needs pred_state_future from policy get_action")
            best, scores = self._wm_bon.select(stacked, s_t, pred_sf)
            tag = "score"
        else:
            # Deterministic among the same K pool (reproducible random arm).
            rng = np.random.RandomState(seed_base + 17)
            best = int(rng.randint(0, self.bon_k))
            tag = "random"
            if self._wm_bon is not None and s_t is not None and pred_sf is not None:
                _, scores = self._wm_bon.select(stacked, s_t, pred_sf)

        if scores is not None:
            print(
                f"[WM-BoN] select={tag} pick={best}/{self.bon_k} "
                f"scores={np.array2string(scores, precision=4)}",
                flush=True,
            )
        else:
            print(f"[WM-BoN] select={tag} pick={best}/{self.bon_k}", flush=True)
        return stacked[best].unsqueeze(0)
