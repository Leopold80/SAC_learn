"""Visualize a trained SAC/PPO policy on Go2 — live viewer or recorded video."""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO, SAC

import sac_experiments.go2_env  # registers Go2Locomotion-v0


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained Go2 policy")
    parser.add_argument("model", type=Path, help="Path to trained model .zip")
    parser.add_argument("--mode", choices=["viewer", "video"], default="viewer",
                        help="viewer (live window) or video (save MP4)")
    parser.add_argument("--output", type=Path, default=Path("go2_policy_demo.mp4"),
                        help="Output video path (mode=video)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def run_viewer(model_path: Path):
    """Interactive MuJoCo viewer — close window to exit."""
    model_class = SAC if "sac" in str(model_path).lower() else PPO
    env = gym.make("Go2Locomotion-v0", render_mode=None)
    trained = model_class.load(model_path, env=env, device="cpu")

    # Use raw MuJoCo model for rendering
    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    obs, _ = env.reset()
    done = False

    print(f"Loaded: {model_path}")
    print("Launching viewer — close window to exit")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            if done:
                obs, _ = env.reset()
                done = False

            action, _ = trained.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            viewer.sync()

    env.close()


def run_video(model_path: Path, output: Path, episodes: int, fps: int):
    """Record policy rollout to MP4 video."""
    import imageio

    model_class = SAC if "sac" in str(model_path).lower() else PPO
    env = gym.make("Go2Locomotion-v0", render_mode="rgb_array")
    trained = model_class.load(model_path, env=env, device="cpu")

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data
    renderer = mujoco.Renderer(mj_model, 480, 640)

    frames = []
    obs, _ = env.reset()
    done = False
    episode_count = 0

    print(f"Loaded: {model_path}")
    print(f"Recording {episodes} episodes to {output}...")

    while episode_count < episodes:
        if done:
            obs, _ = env.reset()
            done = False
            episode_count += 1
            print(f"  Episode {episode_count}/{episodes}")

        action, _ = trained.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        renderer.update_scene(mj_data, camera=-1)
        frames.append(renderer.render())

    renderer.close()
    env.close()

    imageio.mimsave(str(output), frames, fps=fps)
    print(f"Saved: {output}  ({len(frames)} frames, {fps} fps)")


if __name__ == "__main__":
    args = main()
    if args.mode == "viewer":
        run_viewer(args.model)
    else:
        run_video(args.model, args.output, args.episodes, args.fps)
