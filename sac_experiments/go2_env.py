"""Unitree Go2 locomotion environment using MuJoCo physics."""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
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

# ── PD control ────────────────────────────────────────────────────────────────
DEFAULT_KP = 25.0
DEFAULT_KD = 0.6
DEFAULT_ACTION_SCALE = 0.25

# ── Reward shaping ────────────────────────────────────────────────────────────
TRACKING_ERROR_EPS = 1.0e-4
TRACKING_REWARD_MIN = -1.0
EPISODE_MAX_STEPS = 1000

TERMINATION_CODES = {
    "timeout": 0,
    "height": 1,
    "orientation": 2,
    "nonfinite": 3,
    "base_contact": 4,
}

EPISODE_INFO_KEYS = (
    "episode_vx_rmse",
    "episode_vy_rmse",
    "episode_yaw_rmse",
    "episode_mean_vx",
    "episode_mean_vy",
    "episode_mean_yaw_rate",
    "episode_action_saturation",
    "episode_torque_sq",
    "termination_code",
)

# ── Observation/action dimensions ─────────────────────────────────────────────
# local base lin vel (3) + local base ang vel (3) + projected gravity (3)
# + commands (3) + relative joint pos (12) + scaled joint vel (12)
# + previous action (12) = 48
OBS_DIM = 48
ACT_DIM = _NUM_JOINTS


class Go2LocomotionEnv(MujocoEnv):
    """Go2 quadruped locomotion with PD control and velocity tracking reward.

    The policy runs at 50 Hz: MuJoCo integrates ten 2 ms physics steps for each
    policy action. The default task retains the full planar command space and
    applies mass, friction, and PD-gain domain randomization at every reset.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        xml_path: str | None = None,
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        action_scale: float = DEFAULT_ACTION_SCALE,
        default_pose: np.ndarray | None = None,
        command_x_range: tuple[float, float] = (0.0, 1.0),
        command_y_range: tuple[float, float] = (-0.3, 0.3),
        command_yaw_range: tuple[float, float] = (-0.5, 0.5),
        domain_rand_mass: float = 0.15,
        domain_rand_friction: float = 0.30,
        domain_rand_kp: float = 0.15,
        tracking_mode: str = "smooth_baseline",
        tracking_linear_weight: float = 1.5,
        tracking_yaw_weight: float = 0.75,
        tracking_sigma: float = 0.25,
        alive_weight: float = 0.0,
        render_mode: str | None = None,
    ):
        self._xml_path = xml_path or str(_GO2_ASSETS / "scene.xml")
        self._kp = float(kp)
        self._kd = float(kd)
        self._action_scale = float(action_scale)
        self._default_pose = (
            np.asarray(default_pose, dtype=np.float64).copy()
            if default_pose is not None
            else DEFAULT_STAND.copy()
        )
        if self._default_pose.shape != (_NUM_JOINTS,):
            raise ValueError(f"default_pose must have shape ({_NUM_JOINTS},).")
        if self._action_scale <= 0.0:
            raise ValueError("action_scale must be positive.")

        self._command_x_range = self._validate_range(command_x_range, "command_x_range")
        self._command_y_range = self._validate_range(command_y_range, "command_y_range")
        self._command_yaw_range = self._validate_range(command_yaw_range, "command_yaw_range")
        self._domain_rand_mass = self._validate_fraction(domain_rand_mass, "domain_rand_mass")
        self._domain_rand_friction = self._validate_fraction(
            domain_rand_friction, "domain_rand_friction"
        )
        self._domain_rand_kp = self._validate_fraction(domain_rand_kp, "domain_rand_kp")
        if tracking_mode not in {"smooth_baseline", "relative_error"}:
            raise ValueError(
                "tracking_mode must be 'smooth_baseline' or 'relative_error'."
            )
        self._tracking_mode = tracking_mode
        self._tracking_linear_weight = self._validate_non_negative(
            tracking_linear_weight, "tracking_linear_weight"
        )
        self._tracking_yaw_weight = self._validate_non_negative(
            tracking_yaw_weight, "tracking_yaw_weight"
        )
        self._tracking_sigma = self._validate_positive(
            tracking_sigma, "tracking_sigma"
        )
        self._alive_weight = self._validate_non_negative(
            alive_weight, "alive_weight"
        )

        MujocoEnv.__init__(
            self,
            model_path=self._xml_path,
            frame_skip=10,
            observation_space=None,
            render_mode=render_mode,
            default_camera_config={
                "trackbodyid": 1,
                "distance": 3.0,
                "elevation": -25,
                "azimuth": 130,
            },
        )

        joint_names_in_model = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self._joint_ids = np.array(
            [joint_names_in_model.index(name) for name in JOINT_NAMES],
            dtype=np.int32,
        )
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        self._actuator_ids = np.array(
            [actuator_names.index(name) for name in ACTUATOR_NAMES],
            dtype=np.int32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACT_DIM,),
            dtype=np.float32,
        )

        self._commands = np.zeros(3, dtype=np.float64)
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)
        self._current_kp = self._kp
        self._current_kd = self._kd
        self._control_dt = float(self.model.opt.timestep * self.frame_skip)

        # Domain randomization must always be sampled from these immutable
        # nominal values. Multiplying the current model values at each reset
        # causes an unbounded random walk in mass and friction.
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_body_inertia = self.model.body_inertia.copy()
        self._nominal_geom_friction = self.model.geom_friction.copy()
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        if self._floor_geom_id < 0 or self._base_body_id < 0:
            raise ValueError("Go2 model must contain named floor geom and base body.")

        self._reset_episode_metrics()

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset_model(self) -> np.ndarray:
        self._domain_randomize()
        self._resample_commands()

        init_q = self._default_pose.copy()
        init_q += self.np_random.uniform(-0.05, 0.05, size=_NUM_JOINTS)

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[2] = 0.445
        qpos[7 : 7 + _NUM_JOINTS] = init_q
        qvel[:] = 0.0
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

        self._prev_action.fill(0.0)
        self._reset_episode_metrics()
        return self._get_obs()

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ACT_DIM,):
            raise ValueError(f"action must have shape ({ACT_DIM},), got {action.shape}.")
        if not np.isfinite(action).all():
            finite_action = np.clip(
                np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0),
                -1.0,
                1.0,
            )
            return self._nonfinite_transition(finite_action)
        action = np.clip(action, -1.0, 1.0)
        if self._termination_reason() == "nonfinite":
            return self._nonfinite_transition(action)

        previous_action = self._prev_action.copy()
        target = self._default_pose + action * self._action_scale

        joint_pos = self.data.qpos[7 : 7 + _NUM_JOINTS]
        joint_vel = self.data.qvel[6 : 6 + _NUM_JOINTS]
        joint_torque = self._current_kp * (target - joint_pos) - self._current_kd * joint_vel

        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[self._actuator_ids] = joint_torque
        ctrl = np.clip(
            ctrl,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

        # Advance all frame_skip physics substeps. With the supplied XML this
        # is 10 × 0.002 s = 0.02 s per policy action (50 Hz).
        self.do_simulation(ctrl, self.frame_skip)

        termination_reason = self._termination_reason()
        if termination_reason == "nonfinite":
            return self._nonfinite_transition(action)
        terminated = termination_reason is not None
        reward, reward_info = self._compute_reward(
            action=action,
            previous_action=previous_action,
            terminated=terminated,
        )
        self._prev_action = action.copy()
        obs = self._get_obs()

        local_lin_vel, local_ang_vel, _ = self._base_kinematics()
        self._update_episode_metrics(
            local_lin_vel=local_lin_vel,
            local_ang_vel=local_ang_vel,
            action=action,
            reward_info=reward_info,
        )
        episode_finished = terminated or self._episode_steps >= EPISODE_MAX_STEPS
        final_reason = termination_reason or "timeout"
        info = {
            "reward_info": reward_info,
            "commands": self._commands.astype(np.float32, copy=True),
            "local_linear_velocity": local_lin_vel.astype(np.float32, copy=True),
            "local_angular_velocity": local_ang_vel.astype(np.float32, copy=True),
            "base_height": float(self.data.qpos[2]),
            "control_dt": self._control_dt,
            "command_x": float(self._commands[0]),
            "command_y": float(self._commands[1]),
            "command_yaw": float(self._commands[2]),
            "velocity_x": float(local_lin_vel[0]),
            "velocity_y": float(local_lin_vel[1]),
            "yaw_rate": float(local_ang_vel[2]),
            "action_saturation": float(np.mean(np.abs(action) > 0.95)),
            "torque_sq": reward_info["torque_sq"],
            "termination_reason": final_reason if episode_finished else "running",
        }
        info.update({
            f"reward_{key}": value
            for key, value in reward_info.items()
            if key in {
                "alive",
                "tracking_lin_vel",
                "tracking_ang_vel",
                "lin_vel_z",
                "ang_vel_xy",
                "orientation",
                "action_rate",
                "torque",
                "termination",
                "total",
            }
        })
        if episode_finished:
            info.update(self._episode_summary(final_reason))

        # Gymnasium's TimeLimit wrapper supplies truncation at 1000 policy
        # steps. Environment termination is reserved for actual falls.
        truncated = False
        return obs, reward, terminated, truncated, info

    def _nonfinite_transition(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """将数值异常转换为有限的终止样本，避免污染 PPO 更新。"""
        local_lin_vel = np.zeros(3, dtype=np.float64)
        local_ang_vel = np.zeros(3, dtype=np.float64)
        reward_info = {
            "alive": 0.0,
            "tracking_lin_vel": 0.0,
            "tracking_ang_vel": 0.0,
            "lin_vel_z": 0.0,
            "ang_vel_xy": 0.0,
            "orientation": 0.0,
            "action_rate": 0.0,
            "torque": 0.0,
            "termination": -5.0,
            "lin_vel_error_sq": float(np.sum(np.square(self._commands[:2]))),
            "yaw_vel_error_sq": float(np.square(self._commands[2])),
            "lin_vel_stand_error_sq": float(np.sum(np.square(self._commands[:2]))),
            "yaw_vel_stand_error_sq": float(np.square(self._commands[2])),
            "action_rate_cost": 0.0,
            "torque_sq": 0.0,
            "total": -5.0,
        }
        self._prev_action = action.copy()
        self._update_episode_metrics(
            local_lin_vel=local_lin_vel,
            local_ang_vel=local_ang_vel,
            action=action,
            reward_info=reward_info,
        )
        info = {
            "reward_info": reward_info,
            "commands": self._commands.astype(np.float32, copy=True),
            "local_linear_velocity": local_lin_vel.astype(np.float32),
            "local_angular_velocity": local_ang_vel.astype(np.float32),
            "base_height": 0.0,
            "control_dt": self._control_dt,
            "command_x": float(self._commands[0]),
            "command_y": float(self._commands[1]),
            "command_yaw": float(self._commands[2]),
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "yaw_rate": 0.0,
            "action_saturation": float(np.mean(np.abs(action) > 0.95)),
            "torque_sq": 0.0,
            "termination_reason": "nonfinite",
        }
        info.update({
            f"reward_{key}": value
            for key, value in reward_info.items()
            if key in {
                "alive",
                "tracking_lin_vel",
                "tracking_ang_vel",
                "lin_vel_z",
                "ang_vel_xy",
                "orientation",
                "action_rate",
                "torque",
                "termination",
                "total",
            }
        })
        info.update(self._episode_summary("nonfinite"))
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation, -5.0, True, False, info

    # ── Observation ──────────────────────────────────────────────────────────

    def _base_kinematics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return local linear velocity, local angular velocity and gravity."""
        quat = self.data.qpos[3:7].copy()  # MuJoCo freejoint: w, x, y, z
        world_to_base = self._quat_conjugate(quat)
        local_lin_vel = self._rotate_by_quat(self.data.qvel[0:3], world_to_base)
        # MuJoCo stores free-joint angular velocity in the local body frame.
        local_ang_vel = self.data.qvel[3:6].copy()
        projected_gravity = self._rotate_by_quat(
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            world_to_base,
        )
        return local_lin_vel, local_ang_vel, projected_gravity

    def _get_obs(self) -> np.ndarray:
        """Build the normalized 48-dimensional proprioceptive observation."""
        local_lin_vel, local_ang_vel, projected_gravity = self._base_kinematics()
        joint_pos_error = (
            self.data.qpos[7 : 7 + _NUM_JOINTS] - self._default_pose
        )
        joint_vel = self.data.qvel[6 : 6 + _NUM_JOINTS]

        obs = np.concatenate([
            2.0 * local_lin_vel,
            0.25 * local_ang_vel,
            projected_gravity,
            self._commands,
            joint_pos_error,
            0.05 * joint_vel,
            self._prev_action,
        ])
        return obs.astype(np.float32, copy=False)

    # ── Reward ───────────────────────────────────────────────────────────────

    @staticmethod
    def _relative_error_improvement(stand_error: float, current_error: float) -> float:
        """Return the fraction of standstill error removed by the current motion.

        A value of 0 means the motion is no better than standing still, 1 means
        perfect tracking, and negative values mean the motion is worse than
        standing.  The lower clip prevents extreme wrong-way velocities from
        dominating the other reward terms.
        """
        denominator = max(float(stand_error), TRACKING_ERROR_EPS)
        improvement = (float(stand_error) - float(current_error)) / denominator
        return float(np.clip(improvement, TRACKING_REWARD_MIN, 1.0))

    @staticmethod
    def _smooth_baseline_improvement(
        stand_error: float,
        current_error: float,
        sigma: float,
    ) -> float:
        """Smooth tracking score relative to the standstill baseline."""
        return float(
            math.exp(-float(current_error) / sigma)
            - math.exp(-float(stand_error) / sigma)
        )

    def _compute_reward(
        self,
        *,
        action: np.ndarray,
        previous_action: np.ndarray,
        terminated: bool,
    ) -> tuple[float, dict[str, float]]:
        local_lin_vel, local_ang_vel, projected_gravity = self._base_kinematics()

        lin_vel_error = float(np.sum(np.square(self._commands[:2] - local_lin_vel[:2])))
        yaw_vel_error = float(np.square(self._commands[2] - local_ang_vel[2]))
        lin_vel_stand_error = float(np.sum(np.square(self._commands[:2])))
        yaw_vel_stand_error = float(np.square(self._commands[2]))

        if self._tracking_mode == "smooth_baseline":
            tracking_lin_vel = self._smooth_baseline_improvement(
                lin_vel_stand_error,
                lin_vel_error,
                self._tracking_sigma,
            )
            tracking_ang_vel = self._smooth_baseline_improvement(
                yaw_vel_stand_error,
                yaw_vel_error,
                self._tracking_sigma,
            )
        else:
            tracking_lin_vel = self._relative_error_improvement(
                lin_vel_stand_error,
                lin_vel_error,
            )
            tracking_ang_vel = self._relative_error_improvement(
                yaw_vel_stand_error,
                yaw_vel_error,
            )

        lin_vel_z = float(np.square(self.data.qvel[2]))
        ang_vel_xy = float(np.sum(np.square(local_ang_vel[:2])))
        orientation = float(np.sum(np.square(projected_gravity[:2])))
        action_rate = float(np.sum(np.square(action - previous_action)))

        torque = self.data.actuator_force[self._actuator_ids]
        torque_sq = float(np.sum(np.square(torque)))
        termination = float(terminated)

        components = {
            "alive": self._alive_weight * (1.0 - termination),
            "tracking_lin_vel": self._tracking_linear_weight * tracking_lin_vel,
            "tracking_ang_vel": self._tracking_yaw_weight * tracking_ang_vel,
            "lin_vel_z": -2.0 * lin_vel_z,
            "ang_vel_xy": -0.05 * ang_vel_xy,
            "orientation": -1.0 * orientation,
            "action_rate": -0.01 * action_rate,
            "torque": -0.00002 * torque_sq,
            "termination": -5.0 * termination,
        }
        reward = float(sum(components.values()))

        info = {
            **{key: float(value) for key, value in components.items()},
            "lin_vel_error_sq": lin_vel_error,
            "yaw_vel_error_sq": yaw_vel_error,
            "lin_vel_stand_error_sq": lin_vel_stand_error,
            "yaw_vel_stand_error_sq": yaw_vel_stand_error,
            "action_rate_cost": action_rate,
            "torque_sq": torque_sq,
            "total": reward,
        }
        return reward, info

    # ── Termination ──────────────────────────────────────────────────────────

    def _termination_reason(self) -> str | None:
        if not (
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
        ):
            return "nonfinite"
        quat = self.data.qpos[3:7]
        _, pitch, roll = self._quat_to_euler(quat)
        if abs(pitch) > math.radians(60) or abs(roll) > math.radians(60):
            return "orientation"
        if self.data.qpos[2] < 0.25:
            return "height"
        if self._has_base_contact():
            return "base_contact"
        return None

    def _is_terminated(self) -> bool:
        """Compatibility helper used by focused environment tests."""
        return self._termination_reason() is not None

    def _has_base_contact(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_a = int(contact.geom1)
            geom_b = int(contact.geom2)
            if geom_a == self._floor_geom_id:
                other = geom_b
            elif geom_b == self._floor_geom_id:
                other = geom_a
            else:
                continue
            if int(self.model.geom_bodyid[other]) == self._base_body_id:
                return True
        return False

    # ── Episode diagnostics ──────────────────────────────────────────────────

    def _reset_episode_metrics(self) -> None:
        self._episode_steps = 0
        self._episode_vx_error_sq = 0.0
        self._episode_vy_error_sq = 0.0
        self._episode_yaw_error_sq = 0.0
        self._episode_vx = 0.0
        self._episode_vy = 0.0
        self._episode_yaw_rate = 0.0
        self._episode_action_saturation = 0.0
        self._episode_torque_sq = 0.0
        self._episode_reward_components: dict[str, float] = {}

    def _update_episode_metrics(
        self,
        *,
        local_lin_vel: np.ndarray,
        local_ang_vel: np.ndarray,
        action: np.ndarray,
        reward_info: dict[str, float],
    ) -> None:
        self._episode_steps += 1
        self._episode_vx_error_sq += float(
            np.square(self._commands[0] - local_lin_vel[0])
        )
        self._episode_vy_error_sq += float(
            np.square(self._commands[1] - local_lin_vel[1])
        )
        self._episode_yaw_error_sq += float(
            np.square(self._commands[2] - local_ang_vel[2])
        )
        self._episode_vx += float(local_lin_vel[0])
        self._episode_vy += float(local_lin_vel[1])
        self._episode_yaw_rate += float(local_ang_vel[2])
        self._episode_action_saturation += float(np.mean(np.abs(action) > 0.95))
        self._episode_torque_sq += reward_info["torque_sq"]
        for key, value in reward_info.items():
            if key in {
                "alive",
                "tracking_lin_vel",
                "tracking_ang_vel",
                "lin_vel_z",
                "ang_vel_xy",
                "orientation",
                "action_rate",
                "torque",
                "termination",
                "total",
            }:
                self._episode_reward_components[key] = (
                    self._episode_reward_components.get(key, 0.0) + float(value)
                )

    def _episode_summary(self, termination_reason: str) -> dict[str, float]:
        count = max(self._episode_steps, 1)
        summary = {
            "episode_vx_rmse": math.sqrt(self._episode_vx_error_sq / count),
            "episode_vy_rmse": math.sqrt(self._episode_vy_error_sq / count),
            "episode_yaw_rmse": math.sqrt(self._episode_yaw_error_sq / count),
            "episode_mean_vx": self._episode_vx / count,
            "episode_mean_vy": self._episode_vy / count,
            "episode_mean_yaw_rate": self._episode_yaw_rate / count,
            "episode_action_saturation": self._episode_action_saturation / count,
            "episode_torque_sq": self._episode_torque_sq / count,
            "termination_code": float(TERMINATION_CODES[termination_reason]),
        }
        summary.update({
            f"episode_reward_{key}": total / count
            for key, total in self._episode_reward_components.items()
        })
        return summary

    # ── Commands ─────────────────────────────────────────────────────────────

    def _resample_commands(self) -> None:
        self._commands[0] = self.np_random.uniform(*self._command_x_range)
        self._commands[1] = self.np_random.uniform(*self._command_y_range)
        self._commands[2] = self.np_random.uniform(*self._command_yaw_range)

    # ── Domain randomization ─────────────────────────────────────────────────

    def _domain_randomize(self) -> None:
        mass_scale = 1.0 + self.np_random.uniform(
            -self._domain_rand_mass,
            self._domain_rand_mass,
        )
        friction_scale = 1.0 + self.np_random.uniform(
            -self._domain_rand_friction,
            self._domain_rand_friction,
        )

        self.model.body_mass[:] = self._nominal_body_mass
        self.model.body_inertia[:] = self._nominal_body_inertia
        self.model.geom_friction[:] = self._nominal_geom_friction

        self.model.body_mass[1:] = self._nominal_body_mass[1:] * mass_scale
        self.model.body_inertia[1:] = self._nominal_body_inertia[1:] * mass_scale
        self.model.geom_friction[:, 0] = (
            self._nominal_geom_friction[:, 0] * friction_scale
        )
        mujoco.mj_setConst(self.model, self.data)

        self._current_kp = self._kp * (
            1.0 + self.np_random.uniform(-self._domain_rand_kp, self._domain_rand_kp)
        )
        self._current_kd = self._kd * (
            1.0 + self.np_random.uniform(-self._domain_rand_kp, self._domain_rand_kp)
        )

    # ── Validation helpers ───────────────────────────────────────────────────

    @staticmethod
    def _validate_range(value: tuple[float, float], name: str) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        low, high = float(value[0]), float(value[1])
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise ValueError(f"{name} must be a finite ordered pair, got {value!r}.")
        return low, high

    @staticmethod
    def _validate_fraction(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result < 1.0:
            raise ValueError(f"{name} must satisfy 0 <= value < 1, got {value!r}.")
        return result

    @staticmethod
    def _validate_non_negative(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return result

    @staticmethod
    def _validate_positive(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
        return result

    # ── Quaternion helpers ───────────────────────────────────────────────────

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        """Conjugate of quaternion [w, x, y, z]."""
        out = q.copy()
        out[1:] *= -1
        return out

    @staticmethod
    def _rotate_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate 3-vector ``v`` by quaternion ``q`` in [w, x, y, z]."""
        w, x, y, z = q
        ix = w * v[0] + y * v[2] - z * v[1]
        iy = w * v[1] + z * v[0] - x * v[2]
        iz = w * v[2] + x * v[1] - y * v[0]
        iw = -x * v[0] - y * v[1] - z * v[2]
        return np.array([
            ix * w + iw * -x + iy * -z - iz * -y,
            iy * w + iw * -y + iz * -x - ix * -z,
            iz * w + iw * -z + ix * -y - iy * -x,
        ], dtype=np.float64)

    @staticmethod
    def _quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
        """Quaternion [w, x, y, z] to (yaw, pitch, roll)."""
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


register(
    id="Go2Locomotion-v0",
    entry_point="sac_experiments.go2_env:Go2LocomotionEnv",
    max_episode_steps=1000,
)
