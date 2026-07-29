"""Contract tests for every checked-in experiment YAML."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from sac_experiments.config import DEFAULT_CONFIG, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigTests(unittest.TestCase):
    def test_every_experiment_yaml_parses(self) -> None:
        config_paths = [
            *sorted((REPO_ROOT / "configs" / "go2").rglob("*.yaml")),
            *sorted((REPO_ROOT / "configs" / "smoke").rglob("*.yaml")),
        ]
        self.assertGreater(len(config_paths), 0)
        for path in config_paths:
            with self.subTest(config=path.relative_to(REPO_ROOT)):
                config = load_config(path)
                self.assertGreater(config.timesteps, 0)
                self.assertEqual(config.env_id, "Go2Locomotion-v0")

    def test_unknown_nested_key_is_rejected(self) -> None:
        data = deepcopy(DEFAULT_CONFIG)
        data["training"]["typo"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training.typo"):
                load_config(path)

    def test_unsafe_run_tag_is_rejected(self) -> None:
        data = deepcopy(DEFAULT_CONFIG)
        data["output"]["run_tag"] = "../overwrite"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safe path segment"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
