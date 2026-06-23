"""Variant registry for LunarLander SAC comparison experiments."""

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


def base_policy_kwargs() -> dict[str, Any]:
    return {"net_arch": [400, 300]}


def uses_action_history(variant: Variant) -> bool:
    return variant == "ltc_residual_action"


def tensorboard_run_name(variant: Variant) -> str:
    if variant == "ltc_residual":
        return "ltc_res"
    if variant == "ltc_residual_action":
        return "ltc_act"
    return variant


def circuit_ltc_kwargs(args) -> dict[str, Any]:
    ltc = args.ltc
    return {
        "liquid_hidden_dim": ltc["liquid_hidden_dim"],
        "features_dim": ltc["features_dim"],
        "dt": ltc["dt"],
        "tau_min": ltc["tau_min"],
        "ode_unfolds": ltc["ode_unfolds"],
        "reversal_init_scale": ltc["reversal_init_scale"],
    }


def variant_policy_kwargs(args, variant: Variant) -> dict[str, Any]:
    if variant == "mlp":
        return base_policy_kwargs()
    if variant == "ltc_simple":
        ltc = args.ltc
        return {
            **base_policy_kwargs(),
            "features_extractor_class": LTCTemporalFeaturesExtractor,
            "features_extractor_kwargs": {
                "liquid_hidden_dim": ltc["liquid_hidden_dim"],
                "features_dim": ltc["features_dim"],
                "dt": ltc["dt"],
            },
        }
    if variant == "ltc":
        return {
            **base_policy_kwargs(),
            "features_extractor_class": CircuitLTCTemporalFeaturesExtractor,
            "features_extractor_kwargs": circuit_ltc_kwargs(args),
        }
    if variant in {"ltc_residual", "ltc_residual_action"}:
        ltc = args.ltc
        extractor_kwargs = {
            **circuit_ltc_kwargs(args),
            "raw_features_dim": ltc["raw_features_dim"],
            "fusion_hidden_dim": ltc["fusion_hidden_dim"],
        }
        if uses_action_history(variant):
            extractor_kwargs["raw_obs_dim"] = args.action_history["raw_obs_dim"]
        return {
            **base_policy_kwargs(),
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
