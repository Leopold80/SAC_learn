"""Variant registry for LunarLander SAC comparison experiments."""

from __future__ import annotations

from typing import Any, Literal

from sac_experiments.ltc_features import (
    CircuitLTCTemporalFeaturesExtractor,
    LTCTemporalFeaturesExtractor,
)


Variant = Literal["stacked_mlp", "stacked_ltc_simple", "stacked_ltc_circuit"]
DEFAULT_VARIANTS: tuple[Variant, ...] = (
    "stacked_mlp",
    "stacked_ltc_simple",
    "stacked_ltc_circuit",
)
ALL_VARIANTS: tuple[Variant, ...] = DEFAULT_VARIANTS
LEGACY_VARIANT_ALIASES = {"stacked_ltc": "stacked_ltc_simple"}


def canonical_variant(variant: str) -> Variant:
    canonical = LEGACY_VARIANT_ALIASES.get(variant, variant)
    if canonical not in ALL_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return canonical  # type: ignore[return-value]


def base_policy_kwargs() -> dict[str, Any]:
    return {"net_arch": [400, 300]}


def variant_policy_kwargs(args, variant: Variant) -> dict[str, Any]:
    if variant == "stacked_mlp":
        return base_policy_kwargs()
    if variant == "stacked_ltc_simple":
        return {
            **base_policy_kwargs(),
            "features_extractor_class": LTCTemporalFeaturesExtractor,
            "features_extractor_kwargs": {
                "liquid_hidden_dim": args.liquid_hidden_dim,
                "features_dim": args.features_dim,
                "dt": args.ltc_dt,
            },
        }
    if variant == "stacked_ltc_circuit":
        return {
            **base_policy_kwargs(),
            "features_extractor_class": CircuitLTCTemporalFeaturesExtractor,
            "features_extractor_kwargs": {
                "liquid_hidden_dim": args.liquid_hidden_dim,
                "features_dim": args.features_dim,
                "dt": args.ltc_dt,
                "tau_min": args.ltc_tau_min,
                "ode_unfolds": args.ltc_ode_unfolds,
                "reversal_init_scale": args.ltc_reversal_init_scale,
            },
        }
    raise ValueError(f"Unknown variant: {variant}")


def feature_extractor_name(variant: Variant) -> str:
    if variant == "stacked_mlp":
        return "FlattenExtractor"
    if variant == "stacked_ltc_simple":
        return "LTCTemporalFeaturesExtractor"
    if variant == "stacked_ltc_circuit":
        return "CircuitLTCTemporalFeaturesExtractor"
    raise ValueError(f"Unknown variant: {variant}")
