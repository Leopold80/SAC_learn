"""Variant registry shared by LunarLander SAC and PPO experiments."""

from __future__ import annotations

from typing import Any, Literal

from sac_experiments.ltc_features import (
    CircuitLTCTemporalFeaturesExtractor,
    LTCTemporalFeaturesExtractor,
    ResidualCircuitLTCFeaturesExtractor,
)
from sac_experiments.rbf_policies import (
    RBFActorCriticPolicy,
    RBFActorMLPCriticPolicy,
)

Variant = Literal[
    "mlp",
    "ltc",
    "ltc_residual",
    "ltc_residual_action",
    "ltc_simple",
    "rbf_64",
    "rbf_192",
    "rbf_actor_mlp_critic_64",
    "rbf_actor_mlp_critic_192",
]
DEFAULT_VARIANTS: tuple[Variant, ...] = (
    "mlp",
    "ltc",
    "ltc_residual",
    "ltc_residual_action",
)
LEGACY_VARIANTS: tuple[Variant, ...] = ("ltc_simple",)
RBF_VARIANTS: tuple[Variant, ...] = (
    "rbf_64",
    "rbf_192",
    "rbf_actor_mlp_critic_64",
    "rbf_actor_mlp_critic_192",
)
ALL_VARIANTS: tuple[Variant, ...] = DEFAULT_VARIANTS + LEGACY_VARIANTS + RBF_VARIANTS
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


def is_ltc_variant(variant: Variant) -> bool:
    return variant in {"ltc_simple", "ltc", "ltc_residual", "ltc_residual_action"}


def is_rbf_variant(variant: Variant) -> bool:
    return variant in RBF_VARIANTS


def uses_action_history(variant: Variant) -> bool:
    return variant == "ltc_residual_action"


def tensorboard_run_name(variant: Variant) -> str:
    if variant == "ltc_residual":
        return "ltc_res"
    if variant == "ltc_residual_action":
        return "ltc_act"
    if variant == "rbf_actor_mlp_critic_64":
        return "rbf_act_mlp64"
    if variant == "rbf_actor_mlp_critic_192":
        return "rbf_act_mlp192"
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


def rbf_num_centers(config, variant: Variant) -> int:
    if variant in {"rbf_64", "rbf_actor_mlp_critic_64"}:
        return config.rbf["small_num_centers"]
    if variant in {"rbf_192", "rbf_actor_mlp_critic_192"}:
        return config.rbf["large_num_centers"]
    raise ValueError(f"{variant!r} is not an RBF variant.")


def rbf_policy_kwargs(config, variant: Variant) -> dict[str, Any]:
    rbf = config.rbf
    return {
        "rbf_num_centers": rbf_num_centers(config, variant),
        "rbf_center_init_range": rbf["center_init_range"],
        "rbf_initial_bandwidth": rbf["initial_bandwidth"],
        "rbf_min_bandwidth": rbf["min_bandwidth"],
    }


def variant_policy(config, variant: Variant):
    """Return the SB3 policy class/string selected by one configured variant."""

    if variant in {"rbf_64", "rbf_192"}:
        return RBFActorCriticPolicy
    if variant in {"rbf_actor_mlp_critic_64", "rbf_actor_mlp_critic_192"}:
        return RBFActorMLPCriticPolicy
    return config.policy


def variant_policy_kwargs(
    config,
    variant: Variant,
    raw_obs_dim: int | None = None,
) -> dict[str, Any]:
    if variant == "mlp":
        return base_policy_kwargs(config)
    if variant in {"rbf_64", "rbf_192"}:
        # A strict RBF actor and critic only use the final SB3 linear heads.
        return {"net_arch": [], **rbf_policy_kwargs(config, variant)}
    if variant in {"rbf_actor_mlp_critic_64", "rbf_actor_mlp_critic_192"}:
        return {**base_policy_kwargs(config), **rbf_policy_kwargs(config, variant)}
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
    if is_rbf_variant(variant):
        return "FlattenExtractor"
    if variant == "ltc_simple":
        return "LTCTemporalFeaturesExtractor"
    if variant == "ltc":
        return "CircuitLTCTemporalFeaturesExtractor"
    if variant in {"ltc_residual", "ltc_residual_action"}:
        return "ResidualCircuitLTCFeaturesExtractor"
    raise ValueError(f"Unknown variant: {variant}")


def variant_network_summary(config, variant: Variant) -> dict[str, Any]:
    """Describe the actor/critic parameterization stored in experiment JSON."""

    if variant in {"rbf_64", "rbf_192"}:
        return {
            "actor": "GaussianRBF -> linear Gaussian-policy head",
            "critic": "GaussianRBF -> linear value head",
            "hidden_mlp": False,
        }
    if variant in {"rbf_actor_mlp_critic_64", "rbf_actor_mlp_critic_192"}:
        return {
            "actor": "GaussianRBF -> linear Gaussian-policy head",
            "critic": "MLP -> linear value head",
            "critic_net_arch": list(config.policy_net_arch),
            "hidden_mlp": True,
        }
    return {
        "actor": "SB3 MLP policy head",
        "critic": "SB3 MLP value head",
        "hidden_mlp": True,
    }
