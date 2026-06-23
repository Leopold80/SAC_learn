"""Render a trained SAC LunarLanderContinuous-v3 policy to a GIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from stable_baselines3 import SAC

# Import custom feature extractors so SB3 can deserialize LTC models.
from sac_experiments.ltc_features import (
    CircuitLTCTemporalFeaturesExtractor,
    LTCTemporalFeaturesExtractor,
    ResidualCircuitLTCFeaturesExtractor,
)
from sac_experiments.lunarlander_common import DEFAULT_FRAME_STACK, ENV_ID, make_lunarlander_env


DEFAULT_MODEL_PATH = Path("outputs/sac_lunarlander/best_model/best_model.zip")
DEFAULT_OUTPUT_PATH = Path("outputs/sac_lunarlander/lunarlander_best.gif")
DEFAULT_STEPS = 1_000
DEFAULT_SEED = 123
DEFAULT_FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a trained Stable-Baselines3 SAC LunarLanderContinuous-v3 model to GIF."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument(
        "--action-history",
        action="store_true",
        help="Use previous-action observations for ltc_residual_action models.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def render_policy(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_lunarlander_env(
        seed=args.seed,
        frame_stack=args.frame_stack,
        render_mode="rgb_array",
        use_action_history=args.action_history,
    )
    model = SAC.load(args.model_path, device=args.device)

    obs, _info = env.reset(seed=args.seed)
    frames = []
    total_reward = 0.0

    for _step in range(args.steps):
        frames.append(env.render())
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            frames.append(env.render())
            break

    env.close()

    imageio.mimsave(args.output_path, frames, fps=args.fps)
    result = {
        "env_id": ENV_ID,
        "model_path": str(args.model_path),
        "output_path": str(args.output_path),
        "frame_stack": args.frame_stack,
        "uses_action_history": args.action_history,
        "seed": args.seed,
        "fps": args.fps,
        "frames": len(frames),
        "episode_reward": total_reward,
    }

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def main() -> None:
    args = parse_args()
    result = render_policy(args)
    print(f"Saved GIF to {result['output_path']}")
    print(f"Frames: {result['frames']}")
    print(f"Episode reward: {result['episode_reward']:.2f}")


if __name__ == "__main__":
    main()
