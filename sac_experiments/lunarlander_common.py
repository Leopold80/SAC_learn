"""Common LunarLander SAC experiment utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


ENV_ID = "LunarLanderContinuous-v3"
DEFAULT_TIMESTEPS = 500_000
DEFAULT_EVAL_EPISODES = 10
DEFAULT_EVAL_FREQ = 10_000
DEFAULT_SEED = 42
DEFAULT_DEVICE = "cuda"
DEFAULT_FRAME_STACK = 4
DEFAULT_LEARNING_RATE = 7.3e-4
DEFAULT_LEARNING_RATE_NAME = "linear_7.3e-4"
DEFAULT_POLICY_NET_ARCH = (400, 300)
DEFAULT_OUTPUT_DIR = Path("outputs/lunarlander")
DEFAULT_TENSORBOARD_LOG = Path("runs/lunarlander")
LUNARLANDER_OBSERVATION_DIM = 8
LUNARLANDER_ACTION_DIM = 2

SAC_CONFIG = {
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "ent_coef": "auto",
    "gamma": 0.99,
    "tau": 0.01,
    "train_freq": 1,
    "gradient_steps": 1,
    "learning_starts": 10_000,
}


class PreviousActionObservation(gym.Wrapper):
    """Append the previous action to each observation.

    The reset observation uses a zero previous action. After each step, the
    observation is paired with the action that produced it.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("PreviousActionObservation requires a Box observation space.")
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("PreviousActionObservation requires a Box action space.")
        if len(env.observation_space.shape) != 1:
            raise ValueError(
                "PreviousActionObservation expects a flat observation, "
                f"got shape {env.observation_space.shape}."
            )

        obs_low = env.observation_space.low.astype(np.float32)
        obs_high = env.observation_space.high.astype(np.float32)
        action_low = env.action_space.low.astype(np.float32)
        action_high = env.action_space.high.astype(np.float32)
        self.raw_obs_dim = int(obs_low.shape[0])
        self.action_dim = int(action_low.shape[0])
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([obs_low, action_low]),
            high=np.concatenate([obs_high, action_high]),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.previous_action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float32), self.previous_action]).astype(np.float32)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def configure_torch(device: str, allow_cpu: bool) -> str:
    cuda_available = torch.cuda.is_available()
    requested_cuda = str(device).startswith("cuda")

    if requested_cuda and not cuda_available:
        if allow_cpu:
            print(
                "CUDA is not available; falling back to CPU because "
                "training.allow_cpu is enabled in the YAML config."
            )
            return "cpu"
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see a GPU in this session. "
            "Run outside the Codex sandbox or set training.allow_cpu: true in a debug config."
        )

    if device == "auto" and not cuda_available and not allow_cpu:
        raise RuntimeError(
            "No CUDA device is available and CPU fallback is disabled. Use a GPU-enabled "
            "session, or set training.allow_cpu: true in a debug config."
        )

    if cuda_available:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")

    return device


def make_lunarlander_env(
    seed: int,
    frame_stack: int = DEFAULT_FRAME_STACK,
    monitor_dir: Path | None = None,
    render_mode: str | None = None,
    use_action_history: bool = False,
) -> Monitor:
    env = gym.make(ENV_ID, render_mode=render_mode)
    if use_action_history:
        env = PreviousActionObservation(env)
    if frame_stack > 1:
        env = FrameStackObservation(env, stack_size=frame_stack)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(env, filename=str(monitor_dir / "monitor.csv") if monitor_dir else None)


def lunarlander_dimensions(env: gym.Env) -> tuple[int, int]:
    """Return the raw LunarLander observation and action dimensions below wrappers."""

    base_env = env.unwrapped
    observation_space = base_env.observation_space
    action_space = base_env.action_space
    if not isinstance(observation_space, spaces.Box) or not isinstance(action_space, spaces.Box):
        raise TypeError("LunarLander must expose Box observation and action spaces.")
    if len(observation_space.shape) != 1 or len(action_space.shape) != 1:
        raise ValueError(
            "Expected flat LunarLander observation and action spaces, got "
            f"{observation_space.shape} and {action_space.shape}."
        )
    return int(observation_space.shape[0]), int(action_space.shape[0])


def infer_lunarlander_observation_setup(observation_space: spaces.Space) -> tuple[int, bool]:
    """Infer frame stacking and previous-action use from a saved model's input space."""

    if not isinstance(observation_space, spaces.Box):
        raise TypeError(
            "LunarLander rendering requires a Box observation space, got "
            f"{type(observation_space).__name__}."
        )

    shape = observation_space.shape
    if shape == (LUNARLANDER_OBSERVATION_DIM,):
        return 1, False
    if len(shape) != 2:
        raise ValueError(
            "Cannot infer LunarLander observation setup from saved model shape "
            f"{shape}. Expected ({LUNARLANDER_OBSERVATION_DIM},), "
            f"(frames, {LUNARLANDER_OBSERVATION_DIM}), or "
            f"(frames, {LUNARLANDER_OBSERVATION_DIM + LUNARLANDER_ACTION_DIM})."
        )

    frame_stack, observation_dim = (int(dimension) for dimension in shape)
    if frame_stack < 1:
        raise ValueError(f"frame_stack must be positive, got {frame_stack}.")
    if observation_dim == LUNARLANDER_OBSERVATION_DIM:
        return frame_stack, False
    if observation_dim == LUNARLANDER_OBSERVATION_DIM + LUNARLANDER_ACTION_DIM:
        return frame_stack, True
    raise ValueError(
        "Cannot infer whether the saved model uses action history from observation shape "
        f"{shape}. Expected per-frame dimension {LUNARLANDER_OBSERVATION_DIM} or "
        f"{LUNARLANDER_OBSERVATION_DIM + LUNARLANDER_ACTION_DIM}."
    )


def evaluate(model, eval_env: Monitor, episodes: int, label: str) -> tuple[float, float]:
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=True,
        warn=False,
    )
    print(f"{label}: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return float(mean_reward), float(std_reward)


def best_eval_reward(eval_log_path: Path) -> float | None:
    if not eval_log_path.exists():
        return None

    import numpy as np

    data = np.load(eval_log_path)
    if "results" not in data.files or len(data["results"]) == 0:
        return None
    return float(data["results"].mean(axis=1).max())
