"""Render trained Go2 policy walking + turning. Camera tracks the dog body."""

from __future__ import annotations

import gymnasium as gym
import imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

import sac_experiments.go2_env

MODEL_PATH = "outputs/go2_ppo_32env/best_model/best_model.zip"
OUTPUT = "go2_walk_turn.mp4"

SCHEDULE = [
    (100,  0.0, 0.0, 0.0, "stand"),
    (250,  1.0, 0.0, 0.0, "forward fast"),
    (250,  0.5, 0.0, 0.4, "turn left"),
    (200,  0.0, 0.0, 0.0, "stop"),
    (250,  0.5, 0.0, -0.4, "turn right"),
    (200,  0.5, 0.3, 0.0, "side right"),
    (200,  0.5, -0.3, 0.0, "side left"),
    (200,  0.0, 0.0, 0.5, "spin left"),
    (200,  0.0, 0.0, -0.5, "spin right"),
    (250,  0.0, 0.0, 0.0, "stop"),
]


def main():
    env = gym.make("Go2Locomotion-v0", render_mode="rgb_array")
    model = PPO.load(MODEL_PATH)
    print(f"Loaded: {MODEL_PATH}")

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance = 3.0
    cam.elevation = -25
    cam.azimuth = 130

    renderer = mujoco.Renderer(mj_model, 360, 480)

    frames = []
    total_steps = sum(s[0] for s in SCHEDULE)

    obs, _ = env.reset()
    env.unwrapped._commands[:] = [0.0, 0.0, 0.0]

    print(f"Recording {total_steps} steps to {OUTPUT}...")

    for dur, cx, cy, cyaw, label in SCHEDULE:
        env.unwrapped._commands[:] = [cx, cy, cyaw]
        print(f"  {label:20s} ({cx:.1f}, {cy:.1f}, {cyaw:.1f})")

        for _ in range(dur):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            renderer.update_scene(mj_data, camera=cam)
            frames.append(renderer.render())

            if terminated or truncated:
                obs, _ = env.reset()
                env.unwrapped._commands[:] = [cx, cy, cyaw]

    renderer.close()
    env.close()

    fps = 50
    imageio.mimsave(OUTPUT, frames, fps=fps)
    duration = len(frames) / fps
    print(f"Saved: {OUTPUT}  ({len(frames)} frames, {duration:.1f}s)")
    print("Play at 50fps or 1x speed for real-time playback.")


if __name__ == "__main__":
    main()
