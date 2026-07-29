"""Fail-fast diagnostics for the Go2 MuJoCo locomotion environment.

Run before a long PPO/SAC experiment:

    PYTHONPATH=. python3 scripts/diagnose_go2_env.py

The checks are deliberately lightweight and require no training.
"""

from __future__ import annotations

import numpy as np

from sac_experiments.go2_env import ACT_DIM, OBS_DIM, Go2LocomotionEnv


def _assert_close(actual: float, expected: float, *, atol: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def check_control_timing() -> None:
    env = Go2LocomotionEnv(
        command_x_range=(0.0, 0.0),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
    )
    try:
        obs, _ = env.reset(seed=1)
        if obs.shape != (OBS_DIM,) or obs.dtype != np.float32:
            raise AssertionError(
                f"observation must be float32 shape ({OBS_DIM},), got {obs.dtype} {obs.shape}"
            )

        start_time = float(env.data.time)
        zero_action = np.zeros(ACT_DIM, dtype=np.float32)
        final_info: dict = {}
        for step in range(100):
            _, _, terminated, truncated, final_info = env.step(zero_action)
            if terminated or truncated:
                raise AssertionError(f"zero-action standing terminated at diagnostic step {step}")

        elapsed = float(env.data.time) - start_time
        _assert_close(elapsed, 2.0, atol=1e-10, label="100-step simulated time")
        _assert_close(
            float(final_info["control_dt"]),
            0.02,
            atol=1e-12,
            label="policy control period",
        )
        print("[ok] control timing: 100 policy steps = 2.0 s (50 Hz)")
    finally:
        env.close()


def check_domain_randomization_bounds() -> None:
    mass_fraction = 0.15
    friction_fraction = 0.30
    env = Go2LocomotionEnv(
        command_x_range=(0.4, 0.4),
        domain_rand_mass=mass_fraction,
        domain_rand_friction=friction_fraction,
        domain_rand_kp=0.15,
    )
    try:
        nominal_mass = env._nominal_body_mass.copy()
        nominal_friction = env._nominal_geom_friction.copy()

        env.reset(seed=2)
        for _ in range(250):
            env.reset()
            mass = env.model.body_mass
            friction = env.model.geom_friction[:, 0]

            mass_low = nominal_mass[1:] * (1.0 - mass_fraction)
            mass_high = nominal_mass[1:] * (1.0 + mass_fraction)
            if np.any(mass[1:] < mass_low - 1e-12) or np.any(mass[1:] > mass_high + 1e-12):
                raise AssertionError("body mass escaped the configured nominal randomization range")

            friction_low = nominal_friction[:, 0] * (1.0 - friction_fraction)
            friction_high = nominal_friction[:, 0] * (1.0 + friction_fraction)
            if np.any(friction < friction_low - 1e-12) or np.any(friction > friction_high + 1e-12):
                raise AssertionError("friction escaped the configured nominal randomization range")

        print("[ok] domain randomization remains bounded across 250 resets")
    finally:
        env.close()


def check_action_rate_reward() -> None:
    env = Go2LocomotionEnv(command_x_range=(0.4, 0.4))
    try:
        env.reset(seed=3)
        zeros = np.zeros(ACT_DIM, dtype=np.float64)
        ones = np.ones(ACT_DIM, dtype=np.float64)

        _, unchanged_info = env._compute_reward(
            action=zeros,
            previous_action=zeros,
            terminated=False,
        )
        _, changed_info = env._compute_reward(
            action=ones,
            previous_action=zeros,
            terminated=False,
        )
        _assert_close(
            unchanged_info["action_rate_cost"],
            0.0,
            atol=0.0,
            label="unchanged action-rate cost",
        )
        _assert_close(
            changed_info["action_rate_cost"],
            float(ACT_DIM),
            atol=1e-12,
            label="unit action-rate cost",
        )
        if changed_info["action_rate"] >= unchanged_info["action_rate"]:
            raise AssertionError("changing the action must reduce reward through action-rate penalty")

        print("[ok] action-rate penalty uses ||a_t - a_(t-1)||^2")
    finally:
        env.close()


def print_reward_snapshot() -> None:
    env = Go2LocomotionEnv(command_x_range=(0.4, 0.4))
    try:
        env.reset(seed=4)
        _, reward, terminated, _, info = env.step(np.zeros(ACT_DIM, dtype=np.float32))
        components = info["reward_info"]
        print(f"[info] one-step zero-action reward={reward:.6f}, terminated={terminated}")
        for key in (
            "tracking_lin_vel",
            "tracking_ang_vel",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "action_rate",
            "torque",
            "termination",
        ):
            print(f"       {key:>21s}: {components[key]: .6f}")
    finally:
        env.close()


def main() -> None:
    check_control_timing()
    check_domain_randomization_bounds()
    check_action_rate_reward()
    print_reward_snapshot()
    print("All Go2 environment diagnostics passed.")


if __name__ == "__main__":
    main()
