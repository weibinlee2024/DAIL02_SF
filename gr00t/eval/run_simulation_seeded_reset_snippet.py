"""Snippet: seeded env.reset in SimulationInferenceClient.run_simulation"""

        # Main simulation loop — one seeded reset per episode (same formula as teacher probe).
        print(
            f"Episode seeds: env_seed = seed_base + episode_id "
            f"(seed_base={int(config.seed_base)}, n_episodes={config.n_episodes})",
            flush=True,
        )
        for ep_id in range(config.n_episodes):
            env_seed = int(config.seed_base) + int(ep_id)
            obs, _ = self.env.reset(seed=env_seed)
            while True:
                # Process observations and get actions from the server
                obs = {k: v[None, :] if not type(v) == str else (v,) for k, v in obs.items()}
                actions = self._get_actions_from_server(obs)
                # Step the environment
                actions = {k: v.squeeze(0) for k, v in actions.items()}
                next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
                # Update episode tracking
                current_successes[ep_id] |= bool(env_infos["success"][0])
                current_rewards[ep_id] += rewards
                current_lengths[ep_id] += 1
                obs = next_obs

                # If episode ended, store results (next ep resets with its own seed above)
                if terminations or truncations:
                    episode_lengths.append(current_lengths[ep_id])
                    episode_successes.append(current_successes[ep_id])
                    completed_episodes += 1
                    break
