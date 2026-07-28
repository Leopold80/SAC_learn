"""Explicit registry for every supported experiment variant.

The registry is the one place to learn which variants exist and which behavior
they select. Heavy policy modules are imported only when a model is built, so
loading a YAML config remains lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal


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
    "sac_rbf_matched",
    "sac_rbf_actor_mlp_critic_matched",
]
VariantFamily = Literal[
    "mlp",
    "ltc_simple",
    "ltc_circuit",
    "ltc_residual",
    "rbf_strict",
    "rbf_actor_mlp_critic",
]


@dataclass(frozen=True)
class VariantSpec:
    """Stable metadata needed by config, model construction, and reporting."""

    name: Variant
    family: VariantFamily
    feature_extractor_name: str
    tensorboard_name: str | None = None
    uses_action_history: bool = False
    sac_only: bool = False
    rbf_center_key: str | None = None


_SPECS = (
    VariantSpec("mlp", "mlp", "FlattenExtractor"),
    VariantSpec("ltc", "ltc_circuit", "CircuitLTCTemporalFeaturesExtractor"),
    VariantSpec(
        "ltc_residual",
        "ltc_residual",
        "ResidualCircuitLTCFeaturesExtractor",
        tensorboard_name="ltc_res",
    ),
    VariantSpec(
        "ltc_residual_action",
        "ltc_residual",
        "ResidualCircuitLTCFeaturesExtractor",
        tensorboard_name="ltc_act",
        uses_action_history=True,
    ),
    VariantSpec("ltc_simple", "ltc_simple", "LTCTemporalFeaturesExtractor"),
    VariantSpec(
        "rbf_64",
        "rbf_strict",
        "FlattenExtractor",
        rbf_center_key="small_num_centers",
    ),
    VariantSpec(
        "rbf_192",
        "rbf_strict",
        "FlattenExtractor",
        rbf_center_key="large_num_centers",
    ),
    VariantSpec(
        "sac_rbf_matched",
        "rbf_strict",
        "FlattenExtractor",
        tensorboard_name="rbf_match",
        sac_only=True,
        rbf_center_key="sac_strict_matched_num_centers",
    ),
    VariantSpec(
        "rbf_actor_mlp_critic_64",
        "rbf_actor_mlp_critic",
        "FlattenExtractor",
        tensorboard_name="rbf_act_mlp64",
        rbf_center_key="small_num_centers",
    ),
    VariantSpec(
        "rbf_actor_mlp_critic_192",
        "rbf_actor_mlp_critic",
        "FlattenExtractor",
        tensorboard_name="rbf_act_mlp192",
        rbf_center_key="large_num_centers",
    ),
    VariantSpec(
        "sac_rbf_actor_mlp_critic_matched",
        "rbf_actor_mlp_critic",
        "FlattenExtractor",
        tensorboard_name="rbf_act_match",
        sac_only=True,
        rbf_center_key="sac_actor_matched_num_centers",
    ),
)
VARIANT_SPECS = MappingProxyType({spec.name: spec for spec in _SPECS})

DEFAULT_VARIANTS: tuple[Variant, ...] = (
    "mlp",
    "ltc",
    "ltc_residual",
    "ltc_residual_action",
)
LEGACY_VARIANTS: tuple[Variant, ...] = ("ltc_simple",)
RBF_STRICT_VARIANTS: tuple[Variant, ...] = tuple(
    spec.name for spec in _SPECS if spec.family == "rbf_strict"
)
RBF_ACTOR_MLP_CRITIC_VARIANTS: tuple[Variant, ...] = tuple(
    spec.name for spec in _SPECS if spec.family == "rbf_actor_mlp_critic"
)
SAC_ONLY_RBF_VARIANTS: tuple[Variant, ...] = tuple(
    spec.name for spec in _SPECS if spec.sac_only
)
RBF_VARIANTS: tuple[Variant, ...] = (
    *RBF_STRICT_VARIANTS,
    *RBF_ACTOR_MLP_CRITIC_VARIANTS,
)
ALL_VARIANTS: tuple[Variant, ...] = tuple(VARIANT_SPECS)
LEGACY_VARIANT_ALIASES = {
    "stacked_mlp": "mlp",
    "stacked_ltc_circuit": "ltc",
    "stacked_ltc": "ltc_simple",
    "stacked_ltc_simple": "ltc_simple",
}


def canonical_variant(variant: str) -> Variant:
    canonical = LEGACY_VARIANT_ALIASES.get(variant, variant)
    if canonical not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant: {variant}")
    return canonical  # type: ignore[return-value]


def variant_spec(variant: Variant) -> VariantSpec:
    return VARIANT_SPECS[variant]


def base_policy_kwargs(config: Any) -> dict[str, Any]:
    return {"net_arch": list(config.policy_net_arch)}


def is_ltc_variant(variant: Variant) -> bool:
    return variant_spec(variant).family.startswith("ltc_")


def is_rbf_variant(variant: Variant) -> bool:
    return variant_spec(variant).family.startswith("rbf_")


def is_sac_only_rbf_variant(variant: Variant) -> bool:
    return variant_spec(variant).sac_only


def uses_action_history(variant: Variant) -> bool:
    return variant_spec(variant).uses_action_history


def tensorboard_run_name(variant: Variant) -> str:
    spec = variant_spec(variant)
    return spec.tensorboard_name or spec.name


def circuit_ltc_kwargs(config: Any) -> dict[str, Any]:
    ltc = config.ltc
    return {
        "liquid_hidden_dim": ltc["liquid_hidden_dim"],
        "features_dim": ltc["features_dim"],
        "dt": ltc["dt"],
        "tau_min": ltc["tau_min"],
        "ode_unfolds": ltc["ode_unfolds"],
        "reversal_init_scale": ltc["reversal_init_scale"],
    }


def rbf_num_centers(config: Any, variant: Variant) -> int:
    key = variant_spec(variant).rbf_center_key
    if key is None:
        raise ValueError(f"{variant!r} is not an RBF variant.")
    return int(config.rbf[key])


def rbf_policy_kwargs(config: Any, variant: Variant) -> dict[str, Any]:
    rbf = config.rbf
    return {
        "rbf_num_centers": rbf_num_centers(config, variant),
        "rbf_center_init_range": rbf["center_init_range"],
        "rbf_initial_bandwidth": rbf["initial_bandwidth"],
        "rbf_min_bandwidth": rbf["min_bandwidth"],
    }


def variant_policy(config: Any, variant: Variant) -> Any:
    """Return the SB3 policy class/string selected by one configured variant."""

    family = variant_spec(variant).family
    if family == "rbf_strict":
        if config.algorithm == "SAC":
            from sac_experiments.rbf_sac_policies import RBFSACPolicy

            return RBFSACPolicy
        from sac_experiments.rbf_policies import RBFActorCriticPolicy

        return RBFActorCriticPolicy
    if family == "rbf_actor_mlp_critic":
        if config.algorithm == "SAC":
            from sac_experiments.rbf_sac_policies import (
                RBFSACActorMLPCriticPolicy,
            )

            return RBFSACActorMLPCriticPolicy
        from sac_experiments.rbf_policies import RBFActorMLPCriticPolicy

        return RBFActorMLPCriticPolicy
    return config.policy


def variant_policy_kwargs(
    config: Any,
    variant: Variant,
    raw_obs_dim: int | None = None,
) -> dict[str, Any]:
    family = variant_spec(variant).family
    if family == "mlp":
        return base_policy_kwargs(config)
    if family == "rbf_strict":
        return {"net_arch": [], **rbf_policy_kwargs(config, variant)}
    if family == "rbf_actor_mlp_critic":
        return {**base_policy_kwargs(config), **rbf_policy_kwargs(config, variant)}
    if family == "ltc_simple":
        from sac_experiments.ltc_features import LTCTemporalFeaturesExtractor

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
    if family == "ltc_circuit":
        from sac_experiments.ltc_features import (
            CircuitLTCTemporalFeaturesExtractor,
        )

        return {
            **base_policy_kwargs(config),
            "features_extractor_class": CircuitLTCTemporalFeaturesExtractor,
            "features_extractor_kwargs": circuit_ltc_kwargs(config),
        }

    from sac_experiments.ltc_features import ResidualCircuitLTCFeaturesExtractor

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


def feature_extractor_name(variant: Variant) -> str:
    return variant_spec(variant).feature_extractor_name


def variant_ltc_summary(
    config: Any,
    variant: Variant,
) -> dict[str, Any] | None:
    """Return LTC-specific JSON metadata, if the variant uses LTC."""

    family = variant_spec(variant).family
    if not family.startswith("ltc_"):
        return None
    ltc = config.ltc
    summary = {
        "liquid_hidden_dim": ltc["liquid_hidden_dim"],
        "features_dim": ltc["features_dim"],
        "dt": ltc["dt"],
    }
    if family in {"ltc_circuit", "ltc_residual"}:
        summary |= {
            "tau_min": ltc["tau_min"],
            "ode_unfolds": ltc["ode_unfolds"],
            "reversal_init_scale": ltc["reversal_init_scale"],
            "ode": "dx_i = -x_i/tau_i + sum_j f_ij(x_j,u;theta) * (A_ij - x_i)",
        }
    if family == "ltc_residual":
        summary |= {
            "raw_features_dim": ltc["raw_features_dim"],
            "fusion_hidden_dim": ltc["fusion_hidden_dim"],
            "residual": "raw stacked observation projection + LTC feature concat",
        }
    return summary


def variant_rbf_summary(
    config: Any,
    variant: Variant,
) -> dict[str, Any] | None:
    """Return RBF-specific JSON metadata, if the variant uses RBF."""

    if not is_rbf_variant(variant):
        return None
    rbf = config.rbf
    summary = {
        "num_centers": rbf_num_centers(config, variant),
        "center_init_range": rbf["center_init_range"],
        "initial_bandwidth": rbf["initial_bandwidth"],
        "min_bandwidth": rbf["min_bandwidth"],
        "centers": "learnable",
        "bandwidths": "learnable diagonal per-center widths",
        "basis": "exp(-0.5 * sum(((x-c)/sigma)^2))",
    }
    if variant == "sac_rbf_matched":
        summary["capacity_match"] = "SAC actor + online twin-Q optimizer parameters"
    if variant == "sac_rbf_actor_mlp_critic_matched":
        summary["capacity_match"] = "SAC actor optimizer parameters"
    return summary


def variant_network_summary(config: Any, variant: Variant) -> dict[str, Any]:
    """Describe the actor/critic parameterization stored in experiment JSON."""

    family = variant_spec(variant).family
    if family == "rbf_strict":
        if config.algorithm == "SAC":
            return {
                "actor": "GaussianRBF -> squashed Gaussian-policy heads",
                "critic": "two independent GaussianRBF([state, action]) -> linear Q heads",
                "target_critic": "Polyak-updated copies of the twin RBF Q critics",
                "hidden_mlp": False,
            }
        return {
            "actor": "GaussianRBF -> linear Gaussian-policy head",
            "critic": "GaussianRBF -> linear value head",
            "hidden_mlp": False,
        }
    if family == "rbf_actor_mlp_critic":
        if config.algorithm == "SAC":
            return {
                "actor": "GaussianRBF -> squashed Gaussian-policy heads",
                "critic": "two independent SB3 MLP Q(state, action) heads",
                "target_critic": "Polyak-updated copies of the MLP twin Q critics",
                "critic_net_arch": list(config.policy_net_arch),
                "hidden_mlp": True,
            }
        return {
            "actor": "GaussianRBF -> linear Gaussian-policy head",
            "critic": "MLP -> linear value head",
            "critic_net_arch": list(config.policy_net_arch),
            "hidden_mlp": True,
        }
    if config.algorithm == "SAC":
        return {
            "actor": "SB3 MLP -> squashed Gaussian-policy heads",
            "critic": "two independent SB3 MLP Q(state, action) heads",
            "target_critic": "Polyak-updated copies of the MLP twin Q critics",
            "hidden_mlp": True,
        }
    return {
        "actor": "SB3 MLP policy head",
        "critic": "SB3 MLP value head",
        "hidden_mlp": True,
    }
