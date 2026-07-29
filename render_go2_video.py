"""Render Go2 model + periodic gait motion to MP4 video."""
from __future__ import annotations

import math
from pathlib import Path

import imageio
import mujoco
import numpy as np

GO2_SCENE = Path(__file__).resolve().parent / "assets" / "unitree_go2" / "scene.xml"
OUTPUT = Path(__file__).resolve().parent / "go2_demo.mp4"

# Load model
model = mujoco.MjModel.from_xml_path(str(GO2_SCENE))
data = mujoco.MjData(model)

# Set standing pose
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Custom renderer
width, height = 480, 640  # MuJoCo Renderer uses (height, width) ordering!
renderer = mujoco.Renderer(model, width, height)

# Get joint qpos indices (skip the first 7 freejoint entries)
joint_qpos_start = 7

# Default standing pose
default_pose = np.array([
    -0.1, 0.8, -1.5,   # FL
     0.1, 0.8, -1.5,   # FR
    -0.1, 1.0, -1.5,   # RL
     0.1, 1.0, -1.5,   # RR
])

# Generate sinusoidal motion with varying phase offsets
frames = []
fps = 30
total_seconds = 5
total_frames = fps * total_seconds

for frame_idx in range(total_frames):
    t = frame_idx / fps
    phase = 2 * math.pi * t * 0.5  # 0.5 Hz oscillation

    # Gentle sinusoidal offset in knee and hip joints
    offset = np.zeros(12)
    # Front legs
    offset[2]  = 0.15 * math.sin(phase)           # FL knee
    offset[5]  = 0.15 * math.sin(phase + math.pi) # FR knee (opposite)
    # Back legs
    offset[8]  = 0.15 * math.sin(phase + math.pi) # RL knee
    offset[11] = 0.15 * math.sin(phase)           # RR knee

    qpos = np.array(data.qpos)
    qpos[joint_qpos_start:joint_qpos_start+12] = default_pose + offset
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=-1)
    frame = renderer.render()
    frames.append(frame)

renderer.close()

# Save video
imageio.mimsave(str(OUTPUT), frames, fps=fps)
print(f"Video saved to: {OUTPUT}  ({len(frames)} frames, {fps} fps, {width}x{height})")
