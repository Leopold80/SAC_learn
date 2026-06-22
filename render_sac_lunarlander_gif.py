"""Render a trained SAC LunarLanderContinuous-v3 policy to a GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import SAC


ENV_ID = "LunarLanderContinuous-v3"
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
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model_path}. Run sac_sb3_lunarlander_demo.py first."
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID, render_mode="rgb_array")
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
    print(f"Saved GIF to {args.output_path}")
    print(f"Frames: {len(frames)}")
    print(f"Episode reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
