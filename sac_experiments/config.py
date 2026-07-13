"""YAML configuration contract for LunarLander SAC experiments."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from sac_experiments.lunarlander_common import (
    DEFAULT_DEVICE,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_FREQ,
    DEFAULT_FRAME_STACK,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLICY_NET_ARCH,
    DEFAULT_SEED,
    DEFAULT_TENSORBOARD_LOG,
    DEFAULT_TIMESTEPS,
    ENV_ID,
    SAC_CONFIG,
)
from sac_experiments.variants import DEFAULT_VARIANTS, Variant, canonical_variant


DEFAULT_CONFIG_PATH = Path("configs/lunarlander.yaml")
SUPPORTED_ALGORITHM = "SAC"
SUPPORTED_POLICY = "MlpPolicy"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "environment": ENV_ID,
        "algorithm": SUPPORTED_ALGORITHM,
        "policy": SUPPORTED_POLICY,
        "variants": list(DEFAULT_VARIANTS),
    },
    "environment": {
        "frame_stack": DEFAULT_FRAME_STACK,
        "n_envs": 1,
    },
    "training": {
        "timesteps": DEFAULT_TIMESTEPS,
        "seed": DEFAULT_SEED,
        "device": DEFAULT_DEVICE,
        "allow_cpu": False,
        "progress_bar": True,
    },
    "evaluation": {
        "episodes": DEFAULT_EVAL_EPISODES,
        "frequency": DEFAULT_EVAL_FREQ,
    },
    "output": {
        "directory": str(DEFAULT_OUTPUT_DIR),
        "tensorboard_log": str(DEFAULT_TENSORBOARD_LOG),
        "run_tag": None,
    },
    "sac": {
        "learning_rate": DEFAULT_LEARNING_RATE,
        "learning_rate_schedule": "linear",
        "policy_net_arch": list(DEFAULT_POLICY_NET_ARCH),
        **SAC_CONFIG,
    },
    "ltc": {
        "liquid_hidden_dim": 128,
        "features_dim": 256,
        "raw_features_dim": 128,
        "fusion_hidden_dim": 256,
        "dt": 1.0,
        "tau_min": 0.1,
        "ode_unfolds": 4,
        "reversal_init_scale": 1.0,
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated, runtime-ready experiment settings loaded from YAML."""

    config_path: Path
    env_id: str
    algorithm: str
    policy: str
    variants: tuple[Variant, ...]
    frame_stack: int
    n_envs: int
    timesteps: int
    seed: int
    device: str
    allow_cpu: bool
    progress_bar: bool
    eval_episodes: int
    eval_freq: int
    output_dir: Path
    tensorboard_log: Path
    run_tag: str | None
    learning_rate: float
    learning_rate_schedule: str
    policy_net_arch: tuple[int, ...]
    sac: Mapping[str, Any]
    ltc: Mapping[str, Any]


def load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required for YAML configs. Install dependencies from "
            "requirements-sac-demo.txt."
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got {type(data).__name__}.")
    return data


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def reject_unknown_keys(
    override: Mapping[str, Any],
    template: Mapping[str, Any],
    prefix: str = "",
) -> None:
    for key, value in override.items():
        dotted_key = f"{prefix}.{key}" if prefix else key
        if key not in template:
            raise ValueError(f"Unknown config key: {dotted_key}")
        template_value = template[key]
        if isinstance(value, Mapping) and isinstance(template_value, Mapping):
            reject_unknown_keys(value, template_value, dotted_key)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section '{name}' must be a mapping.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _entropy_coefficient(value: Any) -> float | str:
    if isinstance(value, str):
        if value == "auto":
            return value
        if value.startswith("auto_"):
            try:
                _positive_float(float(value.removeprefix("auto_")), "sac.ent_coef")
            except ValueError as exc:
                raise ValueError(
                    "sac.ent_coef must be positive, 'auto', or 'auto_<positive>'."
                ) from exc
            return value
        raise ValueError("sac.ent_coef must be positive, 'auto', or 'auto_<positive>'.")
    return _positive_float(value, "sac.ent_coef")


def load_config(path: Path) -> ExperimentConfig:
    override = load_yaml_file(path)
    reject_unknown_keys(override, DEFAULT_CONFIG)
    config = deep_merge(DEFAULT_CONFIG, override)

    experiment = _section(config, "experiment")
    environment = _section(config, "environment")
    training = _section(config, "training")
    evaluation = _section(config, "evaluation")
    output = _section(config, "output")
    sac = _section(config, "sac")
    ltc = _section(config, "ltc")

    env_id = str(experiment["environment"])
    algorithm = str(experiment["algorithm"])
    policy = str(experiment["policy"])
    if env_id != ENV_ID:
        raise ValueError(f"Only {ENV_ID} is supported, got {env_id!r}.")
    if algorithm != SUPPORTED_ALGORITHM:
        raise ValueError(f"Only {SUPPORTED_ALGORITHM} is supported, got {algorithm!r}.")
    if policy != SUPPORTED_POLICY:
        raise ValueError(f"Only {SUPPORTED_POLICY} is supported, got {policy!r}.")

    variants_value = experiment["variants"]
    if not isinstance(variants_value, list) or not variants_value:
        raise ValueError("experiment.variants must be a non-empty list.")
    variants = tuple(canonical_variant(str(variant)) for variant in variants_value)
    if len(set(variants)) != len(variants):
        raise ValueError(f"experiment.variants contains duplicates: {variants}")

    frame_stack = _positive_int(environment["frame_stack"], "environment.frame_stack")
    n_envs = _positive_int(environment["n_envs"], "environment.n_envs")
    if frame_stack == 1 and any(variant != "mlp" for variant in variants):
        raise ValueError("LTC variants require environment.frame_stack to be at least 2.")
    timesteps = _positive_int(training["timesteps"], "training.timesteps")
    if n_envs > timesteps:
        raise ValueError("environment.n_envs must not exceed training.timesteps.")
    if timesteps % n_envs != 0:
        raise ValueError(
            "training.timesteps must be divisible by environment.n_envs so the "
            "requested total transition count is exact."
        )
    seed = training["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"training.seed must be a non-negative integer, got {seed!r}.")
    device = training["device"]
    if not isinstance(device, str) or not device.strip():
        raise ValueError("training.device must be a non-empty string.")
    allow_cpu = training["allow_cpu"]
    progress_bar = training["progress_bar"]
    if not isinstance(allow_cpu, bool) or not isinstance(progress_bar, bool):
        raise ValueError("training.allow_cpu and training.progress_bar must be booleans.")

    eval_episodes = _positive_int(evaluation["episodes"], "evaluation.episodes")
    eval_freq = _positive_int(evaluation["frequency"], "evaluation.frequency")
    if eval_freq > timesteps:
        raise ValueError(
            "evaluation.frequency must not exceed training.timesteps; otherwise no "
            "best-model evaluation would run."
        )
    if eval_freq % n_envs != 0:
        raise ValueError(
            "evaluation.frequency must be divisible by environment.n_envs because "
            "vectorized callbacks run once per VecEnv step."
        )

    output_directory_value = output["directory"]
    tensorboard_log_value = output["tensorboard_log"]
    if not isinstance(output_directory_value, str) or not output_directory_value.strip():
        raise ValueError("output.directory must be a non-empty path string.")
    if not isinstance(tensorboard_log_value, str) or not tensorboard_log_value.strip():
        raise ValueError("output.tensorboard_log must be a non-empty path string.")
    output_dir = Path(output_directory_value)
    tensorboard_log = Path(tensorboard_log_value)
    run_tag_value = output["run_tag"]
    if run_tag_value is not None and (
        not isinstance(run_tag_value, str) or not run_tag_value.strip()
    ):
        raise ValueError("output.run_tag must be null or a non-empty string.")
    run_tag = run_tag_value.strip() if isinstance(run_tag_value, str) else None
    if run_tag and (
        run_tag in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_tag) is None
    ):
        raise ValueError(
            "output.run_tag must be one safe path segment containing only letters, "
            "numbers, dots, underscores, or hyphens."
        )
    if run_tag:
        output_dir /= run_tag
        tensorboard_log /= run_tag

    learning_rate = _positive_float(sac["learning_rate"], "sac.learning_rate")
    learning_rate_schedule = str(sac["learning_rate_schedule"])
    if learning_rate_schedule not in {"linear", "constant"}:
        raise ValueError(
            "sac.learning_rate_schedule must be 'linear' or 'constant', got "
            f"{learning_rate_schedule!r}."
        )
    policy_net_arch_value = sac["policy_net_arch"]
    if not isinstance(policy_net_arch_value, list) or not policy_net_arch_value:
        raise ValueError("sac.policy_net_arch must be a non-empty list.")
    policy_net_arch = tuple(
        _positive_int(width, f"sac.policy_net_arch[{index}]")
        for index, width in enumerate(policy_net_arch_value)
    )

    sac_kwargs = {
        key: value
        for key, value in sac.items()
        if key not in {"learning_rate", "learning_rate_schedule", "policy_net_arch"}
    }
    sac_kwargs["ent_coef"] = _entropy_coefficient(sac_kwargs["ent_coef"])
    for key in ("buffer_size", "batch_size", "train_freq", "gradient_steps"):
        _positive_int(sac_kwargs[key], f"sac.{key}")
    learning_starts = sac_kwargs["learning_starts"]
    if isinstance(learning_starts, bool) or not isinstance(learning_starts, int) or learning_starts < 0:
        raise ValueError("sac.learning_starts must be a non-negative integer.")
    for key in ("gamma", "tau"):
        value = _positive_float(sac_kwargs[key], f"sac.{key}")
        if value > 1:
            raise ValueError(f"sac.{key} must be at most 1, got {value}.")

    for key in ("liquid_hidden_dim", "features_dim", "raw_features_dim", "fusion_hidden_dim", "ode_unfolds"):
        _positive_int(ltc[key], f"ltc.{key}")
    for key in ("dt", "tau_min", "reversal_init_scale"):
        _positive_float(ltc[key], f"ltc.{key}")

    return ExperimentConfig(
        config_path=path,
        env_id=env_id,
        algorithm=algorithm,
        policy=policy,
        variants=variants,
        frame_stack=frame_stack,
        n_envs=n_envs,
        timesteps=timesteps,
        seed=seed,
        device=device,
        allow_cpu=allow_cpu,
        progress_bar=progress_bar,
        eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        output_dir=output_dir,
        tensorboard_log=tensorboard_log,
        run_tag=run_tag,
        learning_rate=learning_rate,
        learning_rate_schedule=learning_rate_schedule,
        policy_net_arch=policy_net_arch,
        sac=MappingProxyType(sac_kwargs),
        ltc=MappingProxyType(dict(ltc)),
    )
