"""Tests for the explicit, learning-oriented variant registry."""

from __future__ import annotations

from pathlib import Path
import unittest

from sac_experiments.config import load_config
from sac_experiments.variants import (
    ALL_VARIANTS,
    VARIANT_SPECS,
    canonical_variant,
    feature_extractor_name,
    tensorboard_run_name,
    uses_action_history,
    variant_policy_kwargs,
    variant_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class VariantRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ltc_config = load_config(
            REPO_ROOT / "configs" / "sac" / "ltc_comparison.yaml"
        )
        cls.rbf_config = load_config(
            REPO_ROOT / "configs" / "sac" / "rbf_comparison.yaml"
        )

    def test_registry_covers_all_variants_once(self) -> None:
        self.assertEqual(tuple(VARIANT_SPECS), ALL_VARIANTS)
        self.assertEqual(len(VARIANT_SPECS), len(set(VARIANT_SPECS)))

    def test_aliases_resolve_to_registered_specs(self) -> None:
        variant = canonical_variant("stacked_ltc_circuit")
        self.assertEqual(variant, "ltc")
        self.assertEqual(variant_spec(variant).family, "ltc_circuit")

    def test_action_history_metadata_is_centralized(self) -> None:
        self.assertTrue(uses_action_history("ltc_residual_action"))
        self.assertFalse(uses_action_history("ltc_residual"))
        self.assertEqual(tensorboard_run_name("ltc_residual_action"), "ltc_act")
        self.assertEqual(
            feature_extractor_name("ltc_residual_action"),
            "ResidualCircuitLTCFeaturesExtractor",
        )

    def test_policy_kwargs_follow_registry_family(self) -> None:
        mlp = variant_policy_kwargs(self.ltc_config, "mlp")
        rbf = variant_policy_kwargs(self.rbf_config, "rbf_64")
        self.assertEqual(mlp["net_arch"], list(self.ltc_config.policy_net_arch))
        self.assertEqual(rbf["net_arch"], [])
        self.assertEqual(
            rbf["rbf_num_centers"],
            self.rbf_config.rbf["small_num_centers"],
        )


if __name__ == "__main__":
    unittest.main()
