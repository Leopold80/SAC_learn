"""Unitree Go2 locomotion environment using MuJoCo physics."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.envs.registration import register


_GO2_ASSETS = Path(__file__).resolve().parent.parent / "assets" / "unitree_go2"

# ── Joint order (12 DoF) ─────────────────────────────────────────────────────
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

ACTUATOR_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

_NUM_JOINTS = len(JOINT_NAMES)

# ── Default standing pose (rad) ───────────────────────────────────────────────
DEFAULT_STAND = np.array([
    -0.1, 0.8, -1.5,   # FL
     0.1, 0.8, -1.5,   # FR
    -0.1, 1.0, -1.5,   # RL
     0.1, 1.0, -1.5,   # RR
], dtype=np.float64)

# ── PD gains ──────────────────────────────────────────────────────────────────
DEFAULT_KP = 40.0
DEFAULT_KD = 0.5

# ── Observation dim (proprioception) ──────────────────────────────────────────
# base lin vel (3) + base ang vel (3) + projected gravity (3) + commands (3)
# + joint pos (12) + joint vel (12) + prev action (12) = 48
OBS_DIM = 48

# ── Action dim ────────────────────────────────────────────────────────────────
ACT_DIM = _NUM_JOINTS  # 12


class Go2LocomotionEnv(MujocoEnv):
    """Go2 quadruped locomotion with PD control and velocity tracking reward.

    Registered as ``Go2Locomotion-v0`` via Gymnasium.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        xml_path: str | None = None,
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        default_pose: np.ndarray | None = None,
        max_episode_steps: int = 1000,
        command_x_range: tuple[float, float] = (0.0, 1.0),
        command_y_range: tuple[float, float] = (-0.3, 0.3),
        command_yaw_range: tuple[float, float] = (-0.5, 0.5),
        domain_rand_mass: float = 0.15,
        domain_rand_friction: float = 0.30,
        domain_rand_kp: float = 0.15,
        **_kwargs,
    ):
        self._xml_path = xml_path or str(_GO2_ASSETS / "scene.xml")
        self._kp = kp
        self._kd = kd
        self._default_pose = (
            np.array(default_pose, dtype=np.float64)
            if default_pose is not None
            else DEFAULT_STAND.copy()
        )
        self._command_x_range = command_x_range
        self._command_y_range = command_y_range
        self._command_yaw_range = command_yaw_range
        self._domain_rand_mass = domain_rand_mass
        self._domain_rand_friction = domain_rand_friction
        self._domain_rand_kp = domain_rand_kp

        # Pre-scan model for joint actuator IDs — MujocoEnv.__init__
        # already loads mjModel, so we can query it after super().__init__.
        MujocoEnv.__init__(
            self,
            model_path=self._xml_path,
            frame_skip=10,
            observation_space=None,  # we override below
            render_mode=None,
            default_camera_config={"trackbodyid": 1, "distance": 3.0, "elevation": -25, "azimuth": 130},
        )

        # Resolve joint & actuator IDs
        joint_names_in_model = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self._joint_ids = np.array(
            [joint_names_in_model.index(name) for name in JOINT_NAMES], dtype=np.int32
        )
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        self._actuator_ids = np.array(
            [actuator_names.index(name) for name in ACTUATOR_NAMES], dtype=np.int32
        )

        # Override observation & action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float64
        )

        # Internal state
        self._commands = np.zeros(3, dtype=np.float64)
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)

    # ── Reset ────────────────────────────────────────────────────────────

    def reset_model(self):
        self._domain_randomize()
        self._resample_commands()

        # Start near default standing pose with a tiny random offset
        init_q = self._default_pose.copy()
        init_q += self.np_random.uniform(-0.05, 0.05, size=_NUM_JOINTS)
        # Set all qpos entries (first 7 are freejoint: x,y,z,qw,qx,qy,qz)
        qpos = self.init_qpos.copy()
        qpos[2] = 0.445  # nominal base height from go2.xml
        qpos[7 : 7 + _NUM_JOINTS] = init_q
        self.set_state(qpos, self.init_qvel)
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)

        return self._get_obs()

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self._default_pose + action * 0.5  # scale action to ±0.5 rad offset

        # PD control
        q = self.data.qpos[7 : 7 + _NUM_JOINTS]
        qd = self.data.qvel[6 : 6 + _NUM_JOINTS]
        tau = self._current_kp * (target - q) - self._current_kd * qd

        # Apply torques and step
        self.data.ctrl[self._actuator_ids] = tau
        mujoco.mj_step(self.model, self.data, nstep=1)

        obs = self._get_obs()
        self._prev_action = action.copy()

        reward, reward_info = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self.data.time >= 15.0  # max 15 s episodes

        info = {
            "reward_info": reward_info,
            "commands": self._commands.copy(),
            "x_velocity": self.data.qvel[0],
            "y_velocity": self.data.qvel[1],
            "base_height": self.data.qpos[2],
        }
        return obs, reward, terminated, truncated, info

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build 48-dim proprioceptive observation."""
        # Base linear velocity (world frame → base frame via quaternion rotation)
        quat = self.data.qpos[3:7].copy()  # w, x, y, z
        base_lin_vel_world = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]
        # Rotate linear velocity to base frame
        base_lin_vel = self._rotate_by_quat(base_lin_vel_world, self._quat_conjugate(quat))

        # Projected gravity (last column of body rotation matrix in base frame)
        projected_gravity = self._rotate_by_quat(np.array([0, 0, -1]), quat)

        # Joint positions & velocities
        joint_pos = self.data.qpos[7 : 7 + _NUM_JOINTS].copy()
        joint_vel = self.data.qvel[6 : 6 + _NUM_JOINTS].copy()

        return np.concatenate([
            base_lin_vel,           # 3
            base_ang_vel,           # 3
            projected_gravity,      # 3
            self._commands,         # 3
            joint_pos,              # 12
            joint_vel,              # 12
            self._prev_action,      # 12
        ], dtype=np.float64)

    # ── Reward ────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        cmd_x, cmd_y, cmd_yaw = self._commands
        vel_x = self.data.qvel[0]
        vel_y = self.data.qvel[1]
        vel_yaw = self.data.qvel[5]

        # Velocity tracking
        vel_error = np.sqrt(
            (cmd_x - vel_x) ** 2 + (cmd_y - vel_y) ** 2 + 0.5 * (cmd_yaw - vel_yaw) ** 2
        )
        vel_reward = np.exp(-2.0 * vel_error)

        # Orientation penalty — keep base upright
        projected_up = self._rotate_by_quat(
            np.array([0, 0, 1]), self.data.qpos[3:7]
        )
        orientation_reward = float(projected_up[2])  # dot with world Z

        # Action smoothness
        action_mag = float(np.sum(np.square(action)))

        # Energy penalty
        torque = self.data.actuator_force[self._actuator_ids]
        energy = float(np.sum(np.square(torque)))

        # Alive bonus
        alive_bonus = 1.0

        reward = (
            2.0 * vel_reward
            + 0.5 * orientation_reward
            + 0.5 * alive_bonus
            - 0.005 * action_mag
            - 0.0001 * energy
        )

        info = {
            "vel_reward": float(vel_reward),
            "orientation_reward": float(orientation_reward),
            "action_mag": float(action_mag),
            "energy": float(energy),
        }
        return reward, info

    # ── Termination ───────────────────────────────────────────────────────

    def _is_terminated(self) -> bool:
        # Fall: body pitch or roll exceeds ~60 degrees
        quat = self.data.qpos[3:7]
        _, pitch, roll = self._quat_to_euler(quat)
        if abs(pitch) > math.radians(60) or abs(roll) > math.radians(60):
            return True
        # Body too low (collapsed)
        if self.data.qpos[2] < 0.25:
            return True
        return False

    # ── Commands ──────────────────────────────────────────────────────────

    def _resample_commands(self):
        self._commands[0] = self.np_random.uniform(*self._command_x_range)
        self._commands[1] = self.np_random.uniform(*self._command_y_range)
        self._commands[2] = self.np_random.uniform(*self._command_yaw_range)

    # ── Domain Randomization ──────────────────────────────────────────────

    def _domain_randomize(self):
        # Vary body mass
        scale = 1.0 + self.np_random.uniform(
            -self._domain_rand_mass, self._domain_rand_mass
        )
        self.model.body_mass[1:] *= scale  # skip world body

        # Vary friction
        friction_scale = 1.0 + self.np_random.uniform(
            -self._domain_rand_friction, self._domain_rand_friction
        )
        for i in range(self.model.ngeom):
            self.model.geom_friction[i, 0] *= friction_scale

        # Vary PD gains
        self._current_kp = self._kp * (
            1.0 + self.np_random.uniform(-self._domain_rand_kp, self._domain_rand_kp)
        )
        self._current_kd = self._kd * (
            1.0 + self.np_random.uniform(-self._domain_rand_kp, self._domain_rand_kp)
        )

    # ── Quaternion helpers ─────────────────────────────────────────────────

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        """Conjugate of quaternion [w, x, y, z]."""
        out = q.copy()
        out[1:] *= -1
        return out

    @staticmethod
    def _rotate_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate 3-vector `v` by quaternion `q` [w, x, y, z]."""
        # v' = q * [0, v] * q_conj
        w, x, y, z = q
        # Quaternion multiplication optimized
        ix = w * v[0] + y * v[2] - z * v[1]
        iy = w * v[1] + z * v[0] - x * v[2]
        iz = w * v[2] + x * v[1] - y * v[0]
        iw = -x * v[0] - y * v[1] - z * v[2]
        return np.array([
            ix * w + iw * -x + iy * -z - iz * -y,
            iy * w + iw * -y + iz * -x - ix * -z,
            iz * w + iw * -z + ix * -y - iy * -x,
        ])

    @staticmethod
    def _quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
        """Quaternion [w, x, y, z] → (yaw, pitch, roll)."""
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(np.clip(sinp, -1, 1))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw, pitch, roll


# ── Gymnasium registration ───────────────────────────────────────────────────

register(
    id="Go2Locomotion-v0",
    entry_point="sac_experiments.go2_env:Go2LocomotionEnv",
    max_episode_steps=1000,
)
