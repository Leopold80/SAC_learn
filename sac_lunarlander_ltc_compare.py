"""Compare stacked MLP SAC with stacked LTC-feature SAC on LunarLanderContinuous-v3.

This experiment keeps the SAC hyperparameters and frame stacking identical
between variants. The only intended difference is the feature extractor:

* stacked_mlp: SB3 default flattened MLP features over 4 stacked observations.
* stacked_ltc: a custom Liquid Time-Constant temporal feature extractor.

Full GPU comparison:

    MPLCONFIGDIR=/tmp/matplotlib-sac-demo \
    conda run -n sac_sb3_demo python sac_lunarlander_ltc_compare.py

Quick CPU smoke test:

    MPLCONFIGDIR=/tmp/matplotlib-sac-demo \
    conda run -n sac_sb3_demo python sac_lunarlander_ltc_compare.py \
        --variants stacked_mlp stacked_ltc \
        --timesteps 2000 \
        --eval-episodes 2 \
        --eval-freq 1000 \
        --allow-cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Literal

import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from torch import nn


ENV_ID = "LunarLanderContinuous-v3"
DEFAULT_TIMESTEPS = 500_000
DEFAULT_EVAL_EPISODES = 10
DEFAULT_EVAL_FREQ = 10_000
DEFAULT_SEED = 42
DEFAULT_DEVICE = "cuda"
DEFAULT_FRAME_STACK = 4
DEFAULT_OUTPUT_DIR = Path("outputs/sac_lunarlander_compare")
DEFAULT_TENSORBOARD_LOG = Path("runs/sac_lunarlander_compare")

Variant = Literal["stacked_mlp", "stacked_ltc"]
ALL_VARIANTS: tuple[Variant, ...] = ("stacked_mlp", "stacked_ltc")


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


class LTCTemporalFeaturesExtractor(BaseFeaturesExtractor):
    """A compact Liquid Time-Constant feature extractor for stacked observations."""

    def __init__(
        self,
        observation_space: spaces.Box,
        liquid_hidden_dim: int = 128,
        features_dim: int = 256,
        dt: float = 1.0,
    ) -> None:
        if len(observation_space.shape) != 2:
            raise ValueError(
                "LTCTemporalFeaturesExtractor expects stacked observations with "
                f"shape (time, obs_dim), got {observation_space.shape}."
            )

        super().__init__(observation_space, features_dim)
        self.frame_stack = int(observation_space.shape[0])
        self.obs_dim = int(observation_space.shape[1])
        self.liquid_hidden_dim = liquid_hidden_dim
        self.dt = dt

        self.input_layer = nn.Linear(self.obs_dim, liquid_hidden_dim)
        self.recurrent_layer = nn.Linear(liquid_hidden_dim, liquid_hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(liquid_hidden_dim))
        self.log_tau = nn.Parameter(torch.zeros(liquid_hidden_dim))
        self.output_layer = nn.Sequential(
            nn.LayerNorm(liquid_hidden_dim),
            nn.Linear(liquid_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 2:
            observations = observations.reshape(-1, self.frame_stack, self.obs_dim)
        elif observations.ndim != 3:
            raise ValueError(f"Expected observations with 2 or 3 dims, got {observations.shape}.")

        batch_size = observations.shape[0]
        h = torch.zeros(
            batch_size,
            self.liquid_hidden_dim,
            dtype=observations.dtype,
            device=observations.device,
        )
        tau = torch.nn.functional.softplus(self.log_tau) + 1e-3

        for t in range(self.frame_stack):
            target = torch.tanh(
                self.input_layer(observations[:, t, :]) + self.recurrent_layer(h) + self.bias
            )
            dh = (-h + target) / tau
            h = h + self.dt * dh

        return self.output_layer(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stacked MLP SAC and stacked LTC SAC on LunarLanderContinuous-v3."
    )
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=list(ALL_VARIANTS))
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
    parser.add_argument("--liquid-hidden-dim", type=int, default=128)
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--ltc-dt", type=float, default=1.0)
    return parser.parse_args()


def configure_torch(args: argparse.Namespace) -> str:
    cuda_available = torch.cuda.is_available()
    requested_cuda = str(args.device).startswith("cuda")

    if requested_cuda and not cuda_available:
        if args.allow_cpu:
            print("CUDA is not available; falling back to CPU because --allow-cpu was set.")
            return "cpu"
        raise RuntimeError(
            "CUDA was requested for LunarLander SAC comparison, but PyTorch cannot see "
            "a GPU in this session. Run outside the Codex sandbox or pass --allow-cpu "
            "for a short debug run only."
        )

    if args.device == "auto" and not cuda_available and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA device is available and CPU fallback is disabled. Use a GPU-enabled "
            "session, or pass --allow-cpu for a debug run."
        )

    if cuda_available:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")

    return args.device


def make_monitored_env(seed: int, frame_stack: int, monitor_dir: Path | None = None) -> Monitor:
    env = gym.make(ENV_ID)
    env = FrameStackObservation(env, stack_size=frame_stack)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(env, filename=str(monitor_dir / "monitor.csv") if monitor_dir else None)


def evaluate(model: SAC, eval_env: Monitor, episodes: int, label: str) -> tuple[float, float]:
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=True,
        warn=False,
    )
    print(f"{label}: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return float(mean_reward), float(std_reward)


def base_policy_kwargs() -> dict[str, Any]:
    return {"net_arch": [400, 300]}


def variant_policy_kwargs(args: argparse.Namespace, variant: Variant) -> dict[str, Any]:
    if variant == "stacked_mlp":
        return base_policy_kwargs()
    if variant == "stacked_ltc":
        return {
            **base_policy_kwargs(),
            "features_extractor_class": LTCTemporalFeaturesExtractor,
            "features_extractor_kwargs": {
                "liquid_hidden_dim": args.liquid_hidden_dim,
                "features_dim": args.features_dim,
                "dt": args.ltc_dt,
            },
        }
    raise ValueError(f"Unknown variant: {variant}")


def best_eval_reward(eval_log_path: Path) -> float | None:
    if not eval_log_path.exists():
        return None

    import numpy as np

    data = np.load(eval_log_path)
    if "results" not in data.files or len(data["results"]) == 0:
        return None
    return float(data["results"].mean(axis=1).max())


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
    return {
        "variant": variant,
        "env_id": ENV_ID,
        "algorithm": "SAC",
        "policy": "MlpPolicy",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "frame_stack": args.frame_stack,
        "learning_rate": "linear_7.3e-4",
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "ent_coef": "auto",
        "gamma": 0.99,
        "tau": 0.01,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 10_000,
        "policy_kwargs": {
            "net_arch": [400, 300],
            "features_extractor": "LTCTemporalFeaturesExtractor"
            if variant == "stacked_ltc"
            else "FlattenExtractor",
        },
        "ltc": {
            "liquid_hidden_dim": args.liquid_hidden_dim,
            "features_dim": args.features_dim,
            "dt": args.ltc_dt,
        }
        if variant == "stacked_ltc"
        else None,
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

    train_env = make_monitored_env(args.seed, args.frame_stack, monitor_dir / "train")
    eval_env = make_monitored_env(args.seed + 1, args.frame_stack, monitor_dir / "eval")

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(7.3e-4),
        buffer_size=1_000_000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.01,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
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
    device = configure_torch(args)
    set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)

    summaries = []
    for variant in args.variants:
        summaries.append(train_variant(args, variant, device))

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
