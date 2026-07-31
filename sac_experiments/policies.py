"""Go2 训练使用的自定义 Stable-Baselines3 策略。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    Distribution,
    SquashedDiagGaussianDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import Schedule


class TanhActorCriticPolicy(ActorCriticPolicy):
    """使用 Tanh-Gaussian 的 PPO 策略。

    标准 SB3 PPO 使用无界高斯采样，再由算法或环境裁剪动作。Go2 的历史
    训练中，这导致策略均值和标准差持续增大，而环境几乎只收到 ``±1``。
    本策略让 rollout、log-prob 与环境实际使用的有界动作保持一致。
    """

    def __init__(
        self,
        *args: Any,
        log_std_min: float = -5.0,
        log_std_max: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("use_sde", False):
            raise ValueError("TanhActorCriticPolicy does not support use_sde=True.")
        if kwargs.pop("squash_output", False):
            raise ValueError(
                "Do not pass squash_output=True; TanhActorCriticPolicy enables it internally."
            )
        if not np.isfinite(log_std_min) or not np.isfinite(log_std_max):
            raise ValueError("log_std bounds must be finite.")
        if log_std_min >= log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max.")
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        super().__init__(*args, squash_output=False, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("TanhActorCriticPolicy requires a Box action space.")
        if not (
            np.isfinite(self.action_space.low).all()
            and np.isfinite(self.action_space.high).all()
        ):
            raise ValueError("TanhActorCriticPolicy requires finite action bounds.")

        self.action_dist = SquashedDiagGaussianDistribution(
            get_action_dim(self.action_space)
        )
        # 告诉 SB3：policy 输出的是内部 [-1, 1] 动作，送入环境前应按
        # action_space 做反缩放。Go2 的动作空间本身就是 [-1, 1]。
        self._squash_output = True
        super()._build(lr_schedule)

    def effective_log_std(self) -> torch.Tensor:
        """返回实际参与分布计算的有界 log standard deviation。"""
        return torch.clamp(self.log_std, self.log_std_min, self.log_std_max)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(
            mean_actions,
            self.effective_log_std(),
        )
