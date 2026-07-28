"""Focused contract tests for the config-driven search layer."""

from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from sac_experiments.config import load_config
from sac_experiments.hyperparameter_search import (
    TrialEvaluationReporter,
    _confidence_bounds,
    _paired_lower_bound,
    _render_config,
    load_search_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class HyperparameterSearchConfigTests(unittest.TestCase):
    def test_sac_balanced_config_is_valid_and_aligned(self) -> None:
        config = load_search_config(
            REPO_ROOT / "configs/search/sac_mlp_4070_balanced.yaml"
        )
        self.assertEqual(config.base_experiment.algorithm, "SAC")
        self.assertEqual(config.base_experiment.n_envs, 8)
        self.assertEqual(config.resource_rungs, (100_000, 200_000, 400_000))

    def test_ppo_balanced_config_uses_complete_rollouts(self) -> None:
        config = load_search_config(
            REPO_ROOT / "configs/search/ppo_mlp_4070_balanced.yaml"
        )
        rollout_size = config.base_experiment.n_envs * config.base_experiment.ppo["n_steps"]
        self.assertEqual(config.resource_rungs, (262_144, 524_288))
        self.assertTrue(all(rung % rollout_size == 0 for rung in config.resource_rungs))

    def test_resolved_trial_config_reuses_normal_validation(self) -> None:
        config = load_search_config(
            REPO_ROOT / "configs/search/sac_mlp_4070_balanced.yaml"
        )
        parameters = {
            "sac.learning_rate": 0.0005,
            "sac.tau": 0.01,
            "sac.gamma": 0.99,
            "sac.batch_size": 256,
            "sac.ent_coef": "auto",
        }
        with tempfile.TemporaryDirectory() as directory:
            temporary_config = replace(config, output_root=Path(directory))
            resolved_path = _render_config(
                temporary_config,
                parameters,
                training_seed=123,
                trial_root=Path(directory) / "trial",
            )
            resolved = load_config(resolved_path)
        self.assertEqual(resolved.seed, 123)
        self.assertEqual(resolved.sac["batch_size"], 256)


class HyperparameterSearchStatisticsTests(unittest.TestCase):
    def test_confidence_bound_penalizes_variance(self) -> None:
        stable = _confidence_bounds([200.0, 201.0, 199.0, 200.0, 200.0])
        noisy = _confidence_bounds([220.0, 180.0, 220.0, 180.0, 200.0])
        self.assertGreater(stable[2], noisy[2])

    def test_paired_lower_bound_detects_consistent_lead(self) -> None:
        leader = [210.0, 212.0, 211.0, 213.0, 212.0]
        challenger = [200.0, 201.0, 199.0, 202.0, 201.0]
        self.assertGreater(_paired_lower_bound(leader, challenger), 0.0)

    def test_reporter_uses_last_three_evaluations(self) -> None:
        reporter = TrialEvaluationReporter(trial=object(), rungs=(100,), max_resource=200)
        reporter.evaluation_means.extend([10.0, 20.0, 40.0])
        self.assertEqual(reporter.objective_value, 70.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
