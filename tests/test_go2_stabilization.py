"""Regression tests for the bounded-policy Go2 stabilization path."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.vec_env import DummyVecEnv

from sac_experiments.config import load_config
from sac_experiments.env_utils import (
    FixedSeedResetWrapper,
    best_eval_reward,
)
from sac_experiments.go2_env import Go2LocomotionEnv
from sac_experiments.model_factory import build_model
from sac_experiments.policies import TanhActorCriticPolicy
from sac_experiments.runtime_benchmark import (
    _benchmark_config,
    _config_sha256,
    _host_info,
    apply_runtime_profile,
)
from sac_experiments.training import paired_bootstrap_interval


REPO_ROOT = Path(__file__).resolve().parents[1]


class TanhPolicyTests(unittest.TestCase):
    def _policy(self) -> TanhActorCriticPolicy:
        return TanhActorCriticPolicy(
            gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            lambda _: 3e-4,
            net_arch=[16, 16],
            log_std_init=-1.0,
            log_std_min=-5.0,
            log_std_max=0.0,
        )

    def test_policy_uses_squashed_distribution_and_consistent_log_prob(self) -> None:
        policy = self._policy()
        self.assertIsInstance(
            policy.action_dist,
            SquashedDiagGaussianDistribution,
        )
        self.assertTrue(policy.squash_output)

        observations = torch.randn(128, 4)
        actions, _, forward_log_prob = policy(observations)
        _, evaluated_log_prob, _ = policy.evaluate_actions(observations, actions)

        self.assertTrue(torch.all(actions <= 1.0))
        self.assertTrue(torch.all(actions >= -1.0))
        torch.testing.assert_close(forward_log_prob, evaluated_log_prob)

    def test_effective_std_is_clamped(self) -> None:
        policy = self._policy()
        with torch.no_grad():
            policy.log_std.fill_(10.0)
        self.assertTrue(torch.all(policy.effective_log_std() == 0.0))
        distribution = policy.get_distribution(torch.zeros(3, 4))
        torch.testing.assert_close(
            distribution.distribution.scale,
            torch.ones(3, 2),
        )

    def test_ppo_save_load_preserves_bounded_policy(self) -> None:
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        model = PPO(
            TanhActorCriticPolicy,
            env,
            n_steps=16,
            batch_size=8,
            n_epochs=1,
            ent_coef=0.0,
            policy_kwargs={
                "net_arch": [16, 16],
                "log_std_init": -1.0,
                "log_std_min": -5.0,
                "log_std_max": 0.0,
            },
            device="cpu",
        )
        observation = env.reset()
        action_before, _ = model.predict(observation, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            model.save(path)
            loaded = PPO.load(path, env=env, device="cpu")
        action_after, _ = loaded.predict(observation, deterministic=True)
        self.assertIsInstance(loaded.policy, TanhActorCriticPolicy)
        np.testing.assert_allclose(action_before, action_after)
        env.close()


class RewardAndEvaluationTests(unittest.TestCase):
    def test_smooth_reward_is_zero_at_standstill_and_improves_toward_target(self) -> None:
        stand_error = 0.25
        at_standstill = Go2LocomotionEnv._smooth_baseline_improvement(
            stand_error, stand_error, 0.25
        )
        closer = Go2LocomotionEnv._smooth_baseline_improvement(
            stand_error, 0.0625, 0.25
        )
        perfect = Go2LocomotionEnv._smooth_baseline_improvement(
            stand_error, 0.0, 0.25
        )
        worse = Go2LocomotionEnv._smooth_baseline_improvement(
            stand_error, 1.0, 0.25
        )
        self.assertEqual(at_standstill, 0.0)
        self.assertLess(worse, at_standstill)
        self.assertLess(at_standstill, closer)
        self.assertLess(closer, perfect)

    def test_best_eval_uses_episode_mean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.npz"
            np.savez(
                path,
                results=np.asarray([[100.0, 0.0], [60.0, 60.0]]),
            )
            self.assertEqual(best_eval_reward(path), 60.0)

    def test_paired_bootstrap_detects_consistent_improvement(self) -> None:
        lower, upper = paired_bootstrap_interval(
            np.asarray([1.0, 2.0, 1.5, 2.5]),
            samples=2000,
        )
        self.assertGreater(lower, 0.0)
        self.assertGreater(upper, lower)

    def test_fixed_seed_wrapper_replays_initial_state(self) -> None:
        env = FixedSeedResetWrapper(
            gym.make(
                "Go2Locomotion-v0",
                command_x_range=(0.5, 0.5),
                command_y_range=(0.0, 0.0),
                command_yaw_range=(0.0, 0.0),
                domain_rand_mass=0.0,
                domain_rand_friction=0.0,
                domain_rand_kp=0.0,
            ),
            reset_seeds=(123, 456),
        )
        first, _ = env.reset()
        env.reset()
        env.rewind_seed_cycle()
        repeated, _ = env.reset()
        np.testing.assert_allclose(first, repeated)
        env.close()

    def test_nonfinite_physics_state_returns_finite_terminal_sample(self) -> None:
        env = Go2LocomotionEnv(
            command_x_range=(0.5, 0.5),
            command_y_range=(0.0, 0.0),
            command_yaw_range=(0.0, 0.0),
            domain_rand_mass=0.0,
            domain_rand_friction=0.0,
            domain_rand_kp=0.0,
        )
        env.reset(seed=123)
        env.data.qvel[0] = np.nan
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "nonfinite")
        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue(np.isfinite(reward))
        for key in ("episode_vx_rmse", "episode_torque_sq", "termination_code"):
            self.assertTrue(np.isfinite(info[key]))
        env.close()

    def test_stabilized_config_builds_bounded_policy(self) -> None:
        config = load_config(
            REPO_ROOT / "configs" / "go2" / "ppo_stabilized_forward.yaml"
        )
        ppo = dict(config.ppo)
        ppo["batch_size"] = 128
        config = replace(
            config,
            n_envs=1,
            ppo=MappingProxyType(ppo),
        )
        env = DummyVecEnv([
            lambda: gym.make(config.env_id, **config.env_kwargs)
        ])
        model = build_model(config, env, device="cpu", tensorboard_log=None)
        self.assertIsInstance(model.policy, TanhActorCriticPolicy)
        self.assertEqual(config.eval_env_kwargs["domain_rand_mass"], 0.0)
        env.close()

    def test_runtime_profiles_keep_matched_rollout_size(self) -> None:
        config = load_config(
            REPO_ROOT / "configs" / "go2" / "ppo_stabilized_forward.yaml"
        )
        for n_envs in (8, 16, 32):
            with self.subTest(n_envs=n_envs):
                resolved = _benchmark_config(
                    config,
                    n_envs=n_envs,
                    device="cpu",
                    transitions=131_072,
                )
                self.assertEqual(
                    resolved.n_envs * resolved.ppo["n_steps"],
                    2048,
                )

    def test_selected_runtime_profile_is_applied_without_editing_yaml(self) -> None:
        config = load_config(
            REPO_ROOT / "configs" / "go2" / "ppo_stabilized_forward.yaml"
        )
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            candidate = {
                "n_envs": 8,
                "n_steps": 256,
                "device": "cpu",
                "status": "completed",
                "eligible": True,
            }
            profile.write_text(json.dumps({
                "profile_version": 1,
                "config_sha256": _config_sha256(config),
                "host": _host_info(),
                "runs": [candidate],
                "selected": candidate,
            }), encoding="utf-8")
            resolved = apply_runtime_profile(config, profile)
        self.assertEqual(resolved.n_envs, 8)
        self.assertEqual(resolved.ppo["n_steps"], 256)
        self.assertEqual(resolved.device, "cpu")


if __name__ == "__main__":
    unittest.main()
