"""Variant registry shared by LunarLander SAC and PPO experiments."""

from __future__ import annotations

from typing import Any, Literal

from sac_experiments.ltc_features import (
    CircuitLTCTemporalFeaturesExtractor,
    LTCTemporalFeaturesExtractor,
    ResidualCircuitLTCFeaturesExtractor,
)
Variant = Literal["mlp", "ltc", "ltc_residual", "ltc_residual_action", "ltc_simple"]
DEFAULT_VARIANTS: tuple[Variant, ...] = (
    "mlp",
    "ltc",
    "ltc_residual",
    "ltc_residual_action",
)
LEGACY_VARIANTS: tuple[Variant, ...] = ("ltc_simple",)
ALL_VARIANTS: tuple[Variant, ...] = DEFAULT_VARIANTS + LEGACY_VARIANTS
LEGACY_VARIANT_ALIASES = {
    "stacked_mlp": "mlp",
    "stacked_ltc_circuit": "ltc",
    "stacked_ltc": "ltc_simple",
    "stacked_ltc_simple": "ltc_simple",
}


def canonical_variant(variant: str) -> Variant:
    canonical = LEGACY_VARIANT_ALIASES.get(variant, variant)
    if canonical not in ALL_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return canonical  # type: ignore[return-value]


def base_policy_kwargs(config) -> dict[str, Any]:
    return {"net_arch": list(config.policy_net_arch)}


def uses_action_history(variant: Variant) -> bool:
    return variant == "ltc_residual_action"


def tensorboard_run_name(variant: Variant) -> str:
    if variant == "ltc_residual":
        return "ltc_res"
    if variant == "ltc_residual_action":
        return "ltc_act"
    return variant


def circuit_ltc_kwargs(config) -> dict[str, Any]:
    ltc = config.ltc
    return {
        "liquid_hidden_dim": ltc["liquid_hidden_dim"],
        "features_dim": ltc["features_dim"],
        "dt": ltc["dt"],
        "tau_min": ltc["tau_min"],
        "ode_unfolds": ltc["ode_unfolds"],
        "reversal_init_scale": ltc["reversal_init_scale"],
    }


def variant_policy_kwargs(
    config,
    variant: Variant,
    raw_obs_dim: int | None = None,
) -> dict[str, Any]:
    if variant == "mlp":
        return base_policy_kwargs(config)
    if variant == "ltc_simple":
        ltc = config.ltc
        return {
            **base_policy_kwargs(config),
            "features_extractor_class": LTCTemporalFeaturesExtractor,
            "features_extractor_kwargs": {
                "liquid_hidden_dim": ltc["liquid_hidden_dim"],
                "features_dim": ltc["features_dim"],
                "dt": ltc["dt"],
            },
        }
    if variant == "ltc":
        return {
            **base_policy_kwargs(config),
            "features_extractor_class": CircuitLTCTemporalFeaturesExtractor,
            "features_extractor_kwargs": circuit_ltc_kwargs(config),
        }
    if variant in {"ltc_residual", "ltc_residual_action"}:
        ltc = config.ltc
        extractor_kwargs = {
            **circuit_ltc_kwargs(config),
            "raw_features_dim": ltc["raw_features_dim"],
            "fusion_hidden_dim": ltc["fusion_hidden_dim"],
        }
        if uses_action_history(variant):
            if raw_obs_dim is None:
                raise ValueError("raw_obs_dim is required for the action-history variant.")
            extractor_kwargs["raw_obs_dim"] = raw_obs_dim
        return {
            **base_policy_kwargs(config),
            "features_extractor_class": ResidualCircuitLTCFeaturesExtractor,
            "features_extractor_kwargs": extractor_kwargs,
        }
    raise ValueError(f"Unknown variant: {variant}")


def feature_extractor_name(variant: Variant) -> str:
    if variant == "mlp":
        return "FlattenExtractor"
    if variant == "ltc_simple":
        return "LTCTemporalFeaturesExtractor"
    if variant == "ltc":
        return "CircuitLTCTemporalFeaturesExtractor"
    if variant in {"ltc_residual", "ltc_residual_action"}:
        return "ResidualCircuitLTCFeaturesExtractor"
    raise ValueError(f"Unknown variant: {variant}")
