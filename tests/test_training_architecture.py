"""Fast tests for model construction, CLI routing, and training orchestration."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import main as entrypoint
from sac_experiments.config import load_config
from sac_experiments.model_factory import build_model
from sac_experiments.reporting import write_experiment_summary
from sac_experiments.training import run_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_cli_exposes_training_search_and_revalidation(self) -> None:
        training = entrypoint.parse_args(["--config", "train.yaml"])
        multiseed = entrypoint.parse_args([
            "--config",
            "train.yaml",
            "--seeds",
            "0",
            "1",
            "2",
        ])
        benchmark = entrypoint.parse_args([
            "--config",
            "train.yaml",
            "--benchmark-runtime",
        ])
        search = entrypoint.parse_args(["--search-config", "search.yaml"])
        revalidation = entrypoint.parse_args(
            ["--search-config", "search.yaml", "--revalidate"]
        )
        self.assertEqual(training.config, Path("train.yaml"))
        self.assertEqual(multiseed.seeds, [0, 1, 2])
        self.assertTrue(benchmark.benchmark_runtime)
        self.assertEqual(search.search_config, Path("search.yaml"))
        self.assertTrue(revalidation.revalidate)


class ModelFactoryTests(unittest.TestCase):
    def test_factory_passes_validated_settings_to_sb3(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "go2" / "sac_baseline.yaml")
        constructor = Mock(return_value=object())
        train_env = object()
        with (
            patch(
                "sac_experiments.model_factory.algorithm_class",
                return_value=constructor,
            ),
            patch(
                "sac_experiments.model_factory.linear_schedule",
                return_value="schedule",
            ),
        ):
            model = build_model(
                config,
                train_env,
                device="cpu",
                tensorboard_log=Path("runs"),
            )

        self.assertIsNotNone(model)
        args, kwargs = constructor.call_args
        self.assertEqual(kwargs["learning_rate"], 0.0003)
        self.assertEqual(kwargs["policy_kwargs"]["net_arch"], [256, 256])
        self.assertEqual(kwargs["tensorboard_log"], "runs")


class TrainingOrchestrationTests(unittest.TestCase):
    def test_run_experiment_single_variant_without_training(self) -> None:
        base = load_config(REPO_ROOT / "configs" / "smoke" / "go2_sac.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                base,
                output_dir=root / "outputs",
                tensorboard_log=root / "runs",
            )

            def fake_train(_config, device, options):
                self.assertEqual(device, "cpu")
                return {
                    "after_training": {"mean_reward": 1.0},
                    "best_eval_mean_reward": 1.0,
                }

            with (
                patch("sac_experiments.training.configure_torch", return_value="cpu"),
                patch("sac_experiments.training.set_random_seed"),
                patch(
                    "sac_experiments.training.train_variant",
                    side_effect=fake_train,
                ) as train,
            ):
                summary_path = run_experiment(config)

            self.assertEqual(train.call_count, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("run", summary)

    def test_single_variant_summary_filename(self) -> None:
        base = load_config(REPO_ROOT / "configs" / "go2" / "sac_baseline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            config = replace(base, output_dir=Path(directory))
            path = write_experiment_summary(config, [{"algorithm": "SAC"}])
            self.assertEqual(path.name, "experiment_summary.json")


if __name__ == "__main__":
    unittest.main()
