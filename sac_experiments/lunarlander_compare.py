"""Training workflow for LunarLander SAC comparison experiments.

The root script is intentionally thin. Most experiment behavior lives here so
the training flow can be read top-to-bottom and reused by future experiments.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from sac_experiments.lunarlander_common import (
    DEFAULT_DEVICE,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_FREQ,
    DEFAULT_FRAME_STACK,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LEARNING_RATE_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLICY_NET_ARCH,
    DEFAULT_SEED,
    DEFAULT_TENSORBOARD_LOG,
    DEFAULT_TIMESTEPS,
    ENV_ID,
    SAC_CONFIG,
    best_eval_reward,
    configure_torch,
    evaluate,
    linear_schedule,
    lunarlander_dimensions,
    make_lunarlander_env,
)
from sac_experiments.variants import (
    ALL_VARIANTS,
    DEFAULT_VARIANTS,
    Variant,
    canonical_variant,
    feature_extractor_name,
    tensorboard_run_name,
    uses_action_history,
    variant_policy_kwargs,
)


DEFAULT_CONFIG_PATH = Path("configs/lunarlander.yaml")

DEFAULT_CONFIG: dict[str, Any] = {
    "variants": list(DEFAULT_VARIANTS),
    "timesteps": DEFAULT_TIMESTEPS,
    "eval_episodes": DEFAULT_EVAL_EPISODES,
    "eval_freq": DEFAULT_EVAL_FREQ,
    "seed": DEFAULT_SEED,
    "frame_stack": DEFAULT_FRAME_STACK,
    "device": DEFAULT_DEVICE,
    "allow_cpu": False,
    "output_dir": str(DEFAULT_OUTPUT_DIR),
    "tensorboard_log": str(DEFAULT_TENSORBOARD_LOG),
    "run_tag": None,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LunarLanderContinuous-v3 SAC comparison from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required for YAML configs. Install with "
            "`conda run -n sac_sb3_demo python -m pip install -r requirements-sac-demo.txt`."
        ) from exc

    if not path.exists():
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


def load_config(path: Path) -> SimpleNamespace:
    config = deep_merge(DEFAULT_CONFIG, load_yaml_file(path))
    config["variants"] = [canonical_variant(variant) for variant in config["variants"]]
    unknown_variants = sorted(set(config["variants"]) - set(ALL_VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown variants after canonicalization: {unknown_variants}")

    run_tag = config.get("run_tag")
    output_dir = Path(config["output_dir"])
    tensorboard_log = Path(config["tensorboard_log"])
    if run_tag:
        output_dir = output_dir / str(run_tag)
        tensorboard_log = tensorboard_log / str(run_tag)

    config["output_dir"] = output_dir
    config["tensorboard_log"] = tensorboard_log
    return SimpleNamespace(**config)


def ltc_config_for_summary(config: SimpleNamespace, variant: Variant) -> dict[str, Any] | None:
    if variant not in {"ltc_simple", "ltc", "ltc_residual", "ltc_residual_action"}:
        return None

    ltc = config.ltc
    summary = {
        "liquid_hidden_dim": ltc["liquid_hidden_dim"],
        "features_dim": ltc["features_dim"],
        "dt": ltc["dt"],
    }
    if variant == "ltc":
        summary |= {
            "tau_min": ltc["tau_min"],
            "ode_unfolds": ltc["ode_unfolds"],
            "reversal_init_scale": ltc["reversal_init_scale"],
            "ode": "dx_i = -x_i/tau_i + sum_j f_ij(x_j,u;theta) * (A_ij - x_i)",
        }
    if variant in {"ltc_residual", "ltc_residual_action"}:
        summary |= {
            "raw_features_dim": ltc["raw_features_dim"],
            "fusion_hidden_dim": ltc["fusion_hidden_dim"],
            "tau_min": ltc["tau_min"],
            "ode_unfolds": ltc["ode_unfolds"],
            "reversal_init_scale": ltc["reversal_init_scale"],
            "ode": "dx_i = -x_i/tau_i + sum_j f_ij(x_j,u;theta) * (A_ij - x_i)",
            "residual": "raw stacked observation projection + LTC feature concat",
        }
    return summary


def build_variant_summary(
    config: SimpleNamespace,
    variant: Variant,
    before_eval: tuple[float, float],
    after_eval: tuple[float, float],
    final_model_path: Path,
    best_model_path: Path,
    eval_log_path: Path,
    tensorboard_log: Path,
    raw_obs_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "env_id": ENV_ID,
        "algorithm": "SAC",
        "algorithm_source": "stable-baselines3==2.7.0",
        "policy": "MlpPolicy",
        "seed": config.seed,
        "timesteps": config.timesteps,
        "frame_stack": config.frame_stack,
        "uses_action_history": uses_action_history(variant),
        "raw_obs_dim": raw_obs_dim,
        "action_dim": action_dim,
        "learning_rate": DEFAULT_LEARNING_RATE_NAME,
        **SAC_CONFIG,
        "policy_kwargs": {
            "net_arch": list(DEFAULT_POLICY_NET_ARCH),
            "features_extractor": feature_extractor_name(variant),
        },
        "ltc": ltc_config_for_summary(config, variant),
        "eval_episodes": config.eval_episodes,
        "eval_freq": config.eval_freq,
        "before_training": {
            "mean_reward": before_eval[0],
            "std_reward": before_eval[1],
        },
        "after_training": {
            "mean_reward": after_eval[0],
            "std_reward": after_eval[1],
        },
        "best_eval_mean_reward": best_eval_reward(eval_log_path),
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path),
        "tensorboard_log": str(tensorboard_log),
    }


def train_variant(config: SimpleNamespace, variant: Variant, device: str) -> dict[str, Any]:
    print(f"\n=== Training variant: {variant} ===")

    variant_output_dir = config.output_dir / variant
    variant_tensorboard_log = config.tensorboard_log
    tb_run_name = tensorboard_run_name(variant)
    monitor_dir = variant_output_dir / "monitor"
    best_model_dir = variant_output_dir / "best_model"
    checkpoint_dir = variant_output_dir / "checkpoints"
    final_model_path = variant_output_dir / "final_model"
    best_model_path = best_model_dir / "best_model.zip"
    summary_path = variant_output_dir / "eval_summary.json"
    eval_log_path = variant_output_dir / "eval_logs" / "evaluations.npz"

    variant_output_dir.mkdir(parents=True, exist_ok=True)
    variant_tensorboard_log.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_action_history = uses_action_history(variant)
    train_env = make_lunarlander_env(
        config.seed,
        config.frame_stack,
        monitor_dir / "train",
        use_action_history=use_action_history,
    )
    eval_env = make_lunarlander_env(
        config.seed + 1,
        config.frame_stack,
        monitor_dir / "eval",
        use_action_history=use_action_history,
    )
    try:
        raw_obs_dim, action_dim = lunarlander_dimensions(train_env)
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(DEFAULT_LEARNING_RATE),
            **SAC_CONFIG,
            policy_kwargs=variant_policy_kwargs(config, variant, raw_obs_dim=raw_obs_dim),
            tensorboard_log=str(variant_tensorboard_log),
            seed=config.seed,
            device=device,
            verbose=1,
        )

        before_eval = evaluate(model, eval_env, config.eval_episodes, f"{variant} before training")

        callbacks = CallbackList(
            [
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(best_model_dir),
                    log_path=str(variant_output_dir / "eval_logs"),
                    eval_freq=config.eval_freq,
                    n_eval_episodes=config.eval_episodes,
                    deterministic=True,
                    render=False,
                ),
                CheckpointCallback(
                    save_freq=config.eval_freq,
                    save_path=str(checkpoint_dir),
                    name_prefix=f"sac_lunarlander_{variant}",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                ),
            ]
        )

        model.learn(
            total_timesteps=config.timesteps,
            callback=callbacks,
            log_interval=4,
            tb_log_name=tb_run_name,
            progress_bar=True,
        )

        model.save(final_model_path)
        loaded_model = SAC.load(final_model_path, env=eval_env, device=device)
        after_eval = evaluate(loaded_model, eval_env, config.eval_episodes, f"{variant} after training")

        summary = build_variant_summary(
            config=config,
            variant=variant,
            before_eval=before_eval,
            after_eval=after_eval,
            final_model_path=final_model_path.with_suffix(".zip"),
            best_model_path=best_model_path,
            eval_log_path=eval_log_path,
            tensorboard_log=variant_tensorboard_log / f"{tb_run_name}_1",
            raw_obs_dim=raw_obs_dim,
            action_dim=action_dim,
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        train_env.close()
        eval_env.close()


def run_experiment(config: SimpleNamespace) -> Path:
    device = configure_torch(config.device, config.allow_cpu)
    set_random_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.tensorboard_log.mkdir(parents=True, exist_ok=True)

    summaries = [train_variant(config, variant, device) for variant in config.variants]
    compare_summary = {
        "env_id": ENV_ID,
        "algorithm": "SAC",
        "algorithm_source": "stable-baselines3==2.7.0",
        "seed": config.seed,
        "timesteps": config.timesteps,
        "frame_stack": config.frame_stack,
        "config": str(config.config_path),
        "variants": summaries,
    }
    if len(config.variants) == 1:
        compare_summary_path = config.output_dir / f"compare_summary_{config.variants[0]}.json"
    else:
        compare_summary_path = config.output_dir / "compare_summary.json"
    compare_summary_path.write_text(json.dumps(compare_summary, indent=2), encoding="utf-8")

    print("\n=== Comparison complete ===")
    print(f"Summary: {compare_summary_path}")
    for summary in summaries:
        print(
            f"{summary['variant']}: final={summary['after_training']['mean_reward']:.2f}, "
            f"best_eval={summary['best_eval_mean_reward']}"
        )
    return compare_summary_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.config_path = args.config
    run_experiment(config)


if __name__ == "__main__":
    main()
