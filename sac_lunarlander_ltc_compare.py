"""Compare stacked MLP, simple LTC, and circuit LTC SAC on LunarLanderContinuous-v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from sac_experiments.ltc_features import (
    CircuitLTCTemporalFeaturesExtractor,
    LTCTemporalFeaturesExtractor,
)
from sac_experiments.lunarlander_common import (
    DEFAULT_DEVICE,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_FREQ,
    DEFAULT_FRAME_STACK,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_TENSORBOARD_LOG,
    DEFAULT_TIMESTEPS,
    ENV_ID,
    SAC_CONFIG,
    best_eval_reward,
    configure_torch,
    evaluate,
    linear_schedule,
    make_lunarlander_env,
)
from sac_experiments.variants import (
    ALL_VARIANTS,
    DEFAULT_VARIANTS,
    Variant,
    canonical_variant,
    feature_extractor_name,
    variant_policy_kwargs,
)


LEGACY_VARIANTS = ("stacked_ltc",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stacked MLP SAC, simple LTC SAC, and circuit LTC SAC."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=ALL_VARIANTS + LEGACY_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback for debugging only. Normal comparison should use CUDA.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tensorboard-log", type=Path, default=DEFAULT_TENSORBOARD_LOG)
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional subdirectory under output and TensorBoard roots, e.g. threeway_ltc_20260622.",
    )
    parser.add_argument("--liquid-hidden-dim", type=int, default=128)
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--ltc-dt", type=float, default=1.0)
    parser.add_argument("--ltc-tau-min", type=float, default=0.1)
    parser.add_argument("--ltc-ode-unfolds", type=int, default=4)
    parser.add_argument("--ltc-reversal-init-scale", type=float, default=1.0)
    return parser.parse_args()


def build_variant_summary(
    args: argparse.Namespace,
    variant: Variant,
    before_eval: tuple[float, float],
    after_eval: tuple[float, float],
    final_model_path: Path,
    best_model_path: Path,
    eval_log_path: Path,
    tensorboard_log: Path,
) -> dict[str, Any]:
    ltc_config = None
    if variant in {"stacked_ltc_simple", "stacked_ltc_circuit"}:
        ltc_config = {
            "liquid_hidden_dim": args.liquid_hidden_dim,
            "features_dim": args.features_dim,
        }
        ltc_config["dt"] = args.ltc_dt
    if variant == "stacked_ltc_circuit":
        ltc_config |= {
            "tau_min": args.ltc_tau_min,
            "ode_unfolds": args.ltc_ode_unfolds,
            "reversal_init_scale": args.ltc_reversal_init_scale,
            "ode": "dx_i = -x_i/tau_i + sum_j f_ij(x_j,u;theta) * (A_ij - x_i)",
        }
    return {
        "variant": variant,
        "env_id": ENV_ID,
        "algorithm": "SAC",
        "policy": "MlpPolicy",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "frame_stack": args.frame_stack,
        "learning_rate": "linear_7.3e-4",
        **SAC_CONFIG,
        "policy_kwargs": {
            "net_arch": [400, 300],
            "features_extractor": feature_extractor_name(variant),
        },
        "ltc": ltc_config,
        "eval_episodes": args.eval_episodes,
        "eval_freq": args.eval_freq,
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


def train_variant(args: argparse.Namespace, variant: Variant, device: str) -> dict[str, Any]:
    print(f"\n=== Training variant: {variant} ===")

    variant_output_dir = args.output_dir / variant
    variant_tensorboard_log = args.tensorboard_log / variant
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

    train_env = make_lunarlander_env(args.seed, args.frame_stack, monitor_dir / "train")
    eval_env = make_lunarlander_env(args.seed + 1, args.frame_stack, monitor_dir / "eval")

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(7.3e-4),
        **SAC_CONFIG,
        policy_kwargs=variant_policy_kwargs(args, variant),
        tensorboard_log=str(variant_tensorboard_log),
        seed=args.seed,
        device=device,
        verbose=1,
    )

    before_eval = evaluate(model, eval_env, args.eval_episodes, f"{variant} before training")

    callbacks = CallbackList(
        [
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(variant_output_dir / "eval_logs"),
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            ),
            CheckpointCallback(
                save_freq=args.eval_freq,
                save_path=str(checkpoint_dir),
                name_prefix=f"sac_lunarlander_{variant}",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
        ]
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        log_interval=4,
        tb_log_name=variant,
        progress_bar=True,
    )

    model.save(final_model_path)
    loaded_model = SAC.load(final_model_path, env=eval_env, device=device)
    after_eval = evaluate(loaded_model, eval_env, args.eval_episodes, f"{variant} after training")

    summary = build_variant_summary(
        args=args,
        variant=variant,
        before_eval=before_eval,
        after_eval=after_eval,
        final_model_path=final_model_path.with_suffix(".zip"),
        best_model_path=best_model_path,
        eval_log_path=eval_log_path,
        tensorboard_log=variant_tensorboard_log,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    train_env.close()
    eval_env.close()
    return summary


def main() -> None:
    args = parse_args()
    args.variants = [canonical_variant(variant) for variant in args.variants]
    if args.run_tag:
        args.output_dir = args.output_dir / args.run_tag
        args.tensorboard_log = args.tensorboard_log / args.run_tag
    device = configure_torch(args.device, args.allow_cpu)
    set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)

    summaries = [train_variant(args, variant, device) for variant in args.variants]
    compare_summary = {
        "env_id": ENV_ID,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "frame_stack": args.frame_stack,
        "variants": summaries,
    }
    if len(args.variants) == 1:
        compare_summary_path = args.output_dir / f"compare_summary_{args.variants[0]}.json"
    else:
        compare_summary_path = args.output_dir / "compare_summary.json"
    compare_summary_path.write_text(json.dumps(compare_summary, indent=2), encoding="utf-8")

    print("\n=== Comparison complete ===")
    print(f"Summary: {compare_summary_path}")
    for summary in summaries:
        print(
            f"{summary['variant']}: final={summary['after_training']['mean_reward']:.2f}, "
            f"best_eval={summary['best_eval_mean_reward']}"
        )


if __name__ == "__main__":
    main()
