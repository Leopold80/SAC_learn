"""YAML configuration for Go2 locomotion SAC and PPO experiments."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from sac_experiments.go2_env import ACT_DIM, OBS_DIM

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path("configs/go2/sac_baseline.yaml")
SUPPORTED_ALGORITHMS = ("SAC", "PPO")
SUPPORTED_POLICIES = ("MlpPolicy", "TanhMlpPolicy")
ENV_ID = "Go2Locomotion-v0"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "environment": ENV_ID,
        "algorithm": "SAC",
        "policy": "MlpPolicy",
    },
    "environment": {
        "frame_stack": 1,
        "n_envs": 8,
        "action_scale": 0.25,
        "command_x_range": [0.0, 1.0],
        "command_y_range": [-0.3, 0.3],
        "command_yaw_range": [-0.5, 0.5],
        "domain_rand_mass": 0.15,
        "domain_rand_friction": 0.30,
        "domain_rand_kp": 0.15,
    },
    "reward": {
        "tracking_mode": "smooth_baseline",
        "tracking_linear_weight": 1.5,
        "tracking_yaw_weight": 0.75,
        "tracking_sigma": 0.25,
        "alive_weight": 0.0,
    },
    "training": {
        "timesteps": 5_000_000,
        "seed": 42,
        "device": "cuda",
        "allow_cpu": False,
        "progress_bar": True,
    },
    "evaluation": {
        "episodes": 10,
        "frequency": 50_000,
        "seed": 10_000,
        "holdout_seed": 20_000,
        "holdout_episodes": 20,
        "command": [0.5, 0.0, 0.0],
        "domain_randomization": False,
    },
    "output": {
        "directory": "outputs/go2_baseline",
        "tensorboard_log": "runs/go2_baseline",
        "run_tag": None,
    },
    "sac": {
        "learning_rate": 0.0003,
        "learning_rate_schedule": "constant",
        "policy_net_arch": [256, 256],
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "ent_coef": "auto",
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 8,
        "learning_starts": 10_000,
    },
    "ppo": {
        "learning_rate": 0.0003,
        "learning_rate_schedule": "constant",
        "policy_net_arch": [256, 256],
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
        "log_std_init": 0.0,
        "log_std_min": -5.0,
        "log_std_max": 1.0,
    },
}


# ── ExperimentConfig ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    env_id: str
    algorithm: str
    policy: str
    frame_stack: int
    n_envs: int
    timesteps: int
    seed: int
    device: str
    allow_cpu: bool
    progress_bar: bool
    eval_episodes: int
    eval_freq: int
    eval_seed: int
    holdout_seed: int
    holdout_episodes: int
    eval_command: tuple[float, float, float] | None
    eval_domain_randomization: bool
    output_dir: Path
    tensorboard_log: Path
    run_tag: str | None
    learning_rate: float
    learning_rate_schedule: str
    policy_net_arch: tuple[int, ...]
    sac: Mapping[str, Any]
    ppo: Mapping[str, Any]
    ppo_policy_kwargs: Mapping[str, float]
    environment: Mapping[str, Any]
    reward: Mapping[str, Any]
    raw_obs_dim: int
    action_dim: int

    @property
    def env_kwargs(self) -> dict:
        """Return the keyword arguments for the Go2 environment."""
        return {
            **dict(self.environment),
            **dict(self.reward),
        }

    @property
    def eval_env_kwargs(self) -> dict:
        """Return a deterministic, matched evaluation environment."""
        kwargs = self.env_kwargs
        if self.eval_command is not None:
            command_x, command_y, command_yaw = self.eval_command
            kwargs.update({
                "command_x_range": (command_x, command_x),
                "command_y_range": (command_y, command_y),
                "command_yaw_range": (command_yaw, command_yaw),
            })
        if not self.eval_domain_randomization:
            kwargs.update({
                "domain_rand_mass": 0.0,
                "domain_rand_friction": 0.0,
                "domain_rand_kp": 0.0,
            })
        return kwargs


# ── YAML helpers ──────────────────────────────────────────────────────────────

def load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required for YAML configs."
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


# ── Validators ────────────────────────────────────────────────────────────────


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


def _non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _float_range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers.")
    low = _finite_float(value[0], f"{name}[0]")
    high = _finite_float(value[1], f"{name}[1]")
    if low > high:
        raise ValueError(f"{name} must be ordered from low to high.")
    return low, high


def _float_triplet(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers.")
    return tuple(
        _finite_float(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


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


def _validate_algorithm_sections(
    sac: Mapping[str, Any],
    ppo: Mapping[str, Any],
    algorithm: str,
    n_envs: int,
    timesteps: int,
) -> tuple[
    float,
    str,
    tuple[int, ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, float],
]:
    algo = algorithm.lower()
    section = sac if algorithm == "SAC" else ppo
    lr = _positive_float(section["learning_rate"], f"{algo}.learning_rate")
    lr_schedule = str(section["learning_rate_schedule"])
    if lr_schedule not in {"linear", "constant"}:
        raise ValueError(f"{algo}.learning_rate_schedule must be 'linear' or 'constant'.")
    arch_value = section["policy_net_arch"]
    if not isinstance(arch_value, list) or not arch_value:
        raise ValueError(f"{algo}.policy_net_arch must be a non-empty list.")
    arch = tuple(_positive_int(w, f"{algo}.policy_net_arch[{i}]") for i, w in enumerate(arch_value))

    sac_kwargs = {
        k: v for k, v in sac.items()
        if k not in {"learning_rate", "learning_rate_schedule", "policy_net_arch"}
    }
    sac_kwargs["ent_coef"] = _entropy_coefficient(sac_kwargs["ent_coef"])
    for key in ("buffer_size", "batch_size", "train_freq", "gradient_steps"):
        _positive_int(sac_kwargs[key], f"sac.{key}")
    ls = sac_kwargs["learning_starts"]
    if isinstance(ls, bool) or not isinstance(ls, int) or ls < 0:
        raise ValueError("sac.learning_starts must be a non-negative integer.")
    for key in ("gamma", "tau"):
        v = _positive_float(sac_kwargs[key], f"sac.{key}")
        if v > 1:
            raise ValueError(f"sac.{key} must be at most 1, got {v}.")

    ppo_policy_keys = {"log_std_init", "log_std_min", "log_std_max"}
    ppo_policy_kwargs = {
        key: _finite_float(ppo[key], f"ppo.{key}")
        for key in ppo_policy_keys
    }
    if ppo_policy_kwargs["log_std_min"] >= ppo_policy_kwargs["log_std_max"]:
        raise ValueError("ppo.log_std_min must be smaller than ppo.log_std_max.")
    if not (
        ppo_policy_kwargs["log_std_min"]
        <= ppo_policy_kwargs["log_std_init"]
        <= ppo_policy_kwargs["log_std_max"]
    ):
        raise ValueError("ppo.log_std_init must lie within the configured bounds.")

    ppo_kwargs = {
        k: v for k, v in ppo.items()
        if k not in {
            "learning_rate",
            "learning_rate_schedule",
            "policy_net_arch",
            *ppo_policy_keys,
        }
    }
    for key in ("n_steps", "batch_size", "n_epochs"):
        ppo_kwargs[key] = _positive_int(ppo_kwargs[key], f"ppo.{key}")
    rollout_size = n_envs * ppo_kwargs["n_steps"]
    if rollout_size <= 1:
        raise ValueError("PPO requires n_envs * ppo.n_steps > 1.")
    if ppo_kwargs["batch_size"] > rollout_size:
        raise ValueError("ppo.batch_size must not exceed n_envs * ppo.n_steps.")
    if rollout_size % ppo_kwargs["batch_size"] != 0:
        raise ValueError("n_envs * ppo.n_steps must be divisible by ppo.batch_size.")
    if algorithm == "PPO" and timesteps % rollout_size != 0:
        raise ValueError("PPO timesteps must be divisible by n_envs * ppo.n_steps.")
    for key in ("gamma", "gae_lambda", "clip_range"):
        v = _positive_float(ppo_kwargs[key], f"ppo.{key}")
        if v > 1:
            raise ValueError(f"ppo.{key} must be at most 1, got {v}.")
        ppo_kwargs[key] = v
    cvf = ppo_kwargs["clip_range_vf"]
    if cvf is not None:
        ppo_kwargs["clip_range_vf"] = _positive_float(cvf, "ppo.clip_range_vf")
    for key in ("ent_coef", "vf_coef", "max_grad_norm"):
        ppo_kwargs[key] = _non_negative_float(ppo_kwargs[key], f"ppo.{key}")
    for key in ("normalize_advantage", "use_sde"):
        if not isinstance(ppo_kwargs[key], bool):
            raise ValueError(f"ppo.{key} must be a boolean.")
    sde = ppo_kwargs["sde_sample_freq"]
    if isinstance(sde, bool) or not isinstance(sde, int):
        raise ValueError("ppo.sde_sample_freq must be an integer.")
    if sde < -1:
        raise ValueError("ppo.sde_sample_freq must be -1 or non-negative.")
    target_kl = ppo_kwargs["target_kl"]
    if target_kl is not None:
        ppo_kwargs["target_kl"] = _positive_float(target_kl, "ppo.target_kl")

    return lr, lr_schedule, arch, sac_kwargs, ppo_kwargs, ppo_policy_kwargs


# ── Main loader ───────────────────────────────────────────────────────────────


def load_config(path: Path) -> ExperimentConfig:
    override = load_yaml_file(path)
    reject_unknown_keys(override, DEFAULT_CONFIG)
    cfg = deep_merge(DEFAULT_CONFIG, override)

    exp = _section(cfg, "experiment")
    env = _section(cfg, "environment")
    reward = _section(cfg, "reward")
    train = _section(cfg, "training")
    ev = _section(cfg, "evaluation")
    out = _section(cfg, "output")
    sac = _section(cfg, "sac")
    ppo = _section(cfg, "ppo")

    env_id = str(exp["environment"])
    algorithm = str(exp["algorithm"])
    policy = str(exp["policy"])
    if env_id != ENV_ID:
        raise ValueError(f"Only {ENV_ID} is supported, got {env_id!r}.")
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Only {', '.join(SUPPORTED_ALGORITHMS)} are supported.")
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Only {', '.join(SUPPORTED_POLICIES)} are supported.")
    if algorithm == "SAC" and policy != "MlpPolicy":
        raise ValueError("SAC currently supports only MlpPolicy.")

    frame_stack = _positive_int(env["frame_stack"], "environment.frame_stack")
    if frame_stack != 1:
        raise ValueError(
            "environment.frame_stack must be 1; frame stacking is not implemented."
        )
    n_envs = _positive_int(env["n_envs"], "environment.n_envs")
    environment_kwargs = {
        "action_scale": _positive_float(
            env["action_scale"], "environment.action_scale"
        ),
        "command_x_range": _float_range(
            env["command_x_range"], "environment.command_x_range"
        ),
        "command_y_range": _float_range(
            env["command_y_range"], "environment.command_y_range"
        ),
        "command_yaw_range": _float_range(
            env["command_yaw_range"], "environment.command_yaw_range"
        ),
        "domain_rand_mass": _non_negative_float(
            env["domain_rand_mass"], "environment.domain_rand_mass"
        ),
        "domain_rand_friction": _non_negative_float(
            env["domain_rand_friction"], "environment.domain_rand_friction"
        ),
        "domain_rand_kp": _non_negative_float(
            env["domain_rand_kp"], "environment.domain_rand_kp"
        ),
    }
    for name in ("domain_rand_mass", "domain_rand_friction", "domain_rand_kp"):
        if environment_kwargs[name] >= 1:
            raise ValueError(f"environment.{name} must be smaller than 1.")

    tracking_mode = str(reward["tracking_mode"])
    if tracking_mode not in {"smooth_baseline", "relative_error"}:
        raise ValueError(
            "reward.tracking_mode must be 'smooth_baseline' or 'relative_error'."
        )
    reward_kwargs = {
        "tracking_mode": tracking_mode,
        "tracking_linear_weight": _non_negative_float(
            reward["tracking_linear_weight"], "reward.tracking_linear_weight"
        ),
        "tracking_yaw_weight": _non_negative_float(
            reward["tracking_yaw_weight"], "reward.tracking_yaw_weight"
        ),
        "tracking_sigma": _positive_float(
            reward["tracking_sigma"], "reward.tracking_sigma"
        ),
        "alive_weight": _non_negative_float(
            reward["alive_weight"], "reward.alive_weight"
        ),
    }
    timesteps = _positive_int(train["timesteps"], "training.timesteps")
    if n_envs > timesteps:
        raise ValueError("n_envs must not exceed timesteps.")
    if timesteps % n_envs != 0:
        raise ValueError("timesteps must be divisible by n_envs.")

    seed = train["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be non-negative integer, got {seed!r}.")
    device = train["device"]
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")
    allow_cpu = train["allow_cpu"]
    progress_bar = train["progress_bar"]
    if not isinstance(allow_cpu, bool) or not isinstance(progress_bar, bool):
        raise ValueError("allow_cpu and progress_bar must be booleans.")

    eval_episodes = _positive_int(ev["episodes"], "evaluation.episodes")
    eval_freq = _positive_int(ev["frequency"], "evaluation.frequency")
    eval_seed = ev["seed"]
    if isinstance(eval_seed, bool) or not isinstance(eval_seed, int) or eval_seed < 0:
        raise ValueError("evaluation.seed must be a non-negative integer.")
    holdout_seed = ev["holdout_seed"]
    if (
        isinstance(holdout_seed, bool)
        or not isinstance(holdout_seed, int)
        or holdout_seed < 0
    ):
        raise ValueError("evaluation.holdout_seed must be a non-negative integer.")
    holdout_episodes = _positive_int(
        ev["holdout_episodes"], "evaluation.holdout_episodes"
    )
    eval_command = (
        None
        if ev["command"] is None
        else _float_triplet(ev["command"], "evaluation.command")
    )
    eval_domain_randomization = ev["domain_randomization"]
    if not isinstance(eval_domain_randomization, bool):
        raise ValueError("evaluation.domain_randomization must be a boolean.")
    if eval_freq > timesteps:
        raise ValueError("eval frequency must not exceed timesteps.")
    if eval_freq % n_envs != 0:
        raise ValueError("eval frequency must be divisible by n_envs.")

    od = out["directory"]
    tb = out["tensorboard_log"]
    if not isinstance(od, str) or not od.strip():
        raise ValueError("output.directory must be a non-empty path string.")
    if not isinstance(tb, str) or not tb.strip():
        raise ValueError("output.tensorboard_log must be a non-empty path string.")
    output_dir = Path(od)
    tensorboard_log = Path(tb)
    run_tag_value = out["run_tag"]
    if run_tag_value is not None and (not isinstance(run_tag_value, str) or not run_tag_value.strip()):
        raise ValueError("output.run_tag must be null or a non-empty string.")
    run_tag = run_tag_value.strip() if isinstance(run_tag_value, str) else None
    if run_tag and (run_tag in {".", ".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_tag) is None):
        raise ValueError("run_tag must be a safe path segment.")
    if run_tag:
        output_dir /= run_tag
        tensorboard_log /= run_tag

    (
        lr,
        lr_schedule,
        arch,
        sac_kwargs,
        ppo_kwargs,
        ppo_policy_kwargs,
    ) = _validate_algorithm_sections(
        sac, ppo, algorithm, n_envs, timesteps
    )

    return ExperimentConfig(
        config_path=path,
        env_id=env_id,
        algorithm=algorithm,
        policy=policy,
        frame_stack=frame_stack,
        n_envs=n_envs,
        timesteps=timesteps,
        seed=seed,
        device=device,
        allow_cpu=allow_cpu,
        progress_bar=progress_bar,
        eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        eval_seed=eval_seed,
        holdout_seed=holdout_seed,
        holdout_episodes=holdout_episodes,
        eval_command=eval_command,
        eval_domain_randomization=eval_domain_randomization,
        output_dir=output_dir,
        tensorboard_log=tensorboard_log,
        run_tag=run_tag,
        learning_rate=lr,
        learning_rate_schedule=lr_schedule,
        policy_net_arch=arch,
        sac=MappingProxyType(sac_kwargs),
        ppo=MappingProxyType(ppo_kwargs),
        ppo_policy_kwargs=MappingProxyType(ppo_policy_kwargs),
        environment=MappingProxyType(environment_kwargs),
        reward=MappingProxyType(reward_kwargs),
        raw_obs_dim=OBS_DIM,
        action_dim=ACT_DIM,
    )
