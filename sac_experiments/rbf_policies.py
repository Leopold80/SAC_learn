"""PPO policies whose actor and/or critic use learnable Gaussian RBF features."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn
from torch.nn import functional as functional


class GaussianRBF(nn.Module):
    """A diagonal Gaussian RBF layer with trainable centers and bandwidths."""

    def __init__(
        self,
        input_dim: int,
        num_centers: int,
        center_init_range: float,
        initial_bandwidth: float,
        min_bandwidth: float,
    ) -> None:
        super().__init__()
        if input_dim < 1 or num_centers < 1:
            raise ValueError("input_dim and num_centers must both be positive.")
        if center_init_range <= 0:
            raise ValueError("center_init_range must be positive.")
        if min_bandwidth <= 0 or initial_bandwidth <= min_bandwidth:
            raise ValueError("initial_bandwidth must be greater than min_bandwidth > 0.")

        self.input_dim = input_dim
        self.num_centers = num_centers
        self.min_bandwidth = min_bandwidth
        self.centers = nn.Parameter(torch.empty(num_centers, input_dim))
        nn.init.uniform_(self.centers, -center_init_range, center_init_range)

        # sigma = min_bandwidth + softplus(raw_bandwidth).  Initialize the
        # unconstrained parameter so every coordinate starts at the requested
        # positive bandwidth without using a non-differentiable clamp.
        shifted_bandwidth = initial_bandwidth - min_bandwidth
        raw_bandwidth = math.log(math.expm1(shifted_bandwidth))
        self.raw_bandwidths = nn.Parameter(
            torch.full((num_centers, input_dim), raw_bandwidth)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "GaussianRBF expects features with shape "
                f"(batch, {self.input_dim}), got {tuple(features.shape)}."
            )
        bandwidths = functional.softplus(self.raw_bandwidths) + self.min_bandwidth
        normalized_delta = (
            features.unsqueeze(1) - self.centers.unsqueeze(0)
        ) / bandwidths.unsqueeze(0)
        squared_distance = normalized_delta.square().sum(dim=-1)
        return torch.exp(-0.5 * squared_distance)


class RBFActorCriticExtractor(nn.Module):
    """Separate RBF actor and critic branches followed by SB3 linear readouts."""

    def __init__(
        self,
        feature_dim: int,
        num_centers: int,
        center_init_range: float,
        initial_bandwidth: float,
        min_bandwidth: float,
    ) -> None:
        super().__init__()
        self.actor_rbf = GaussianRBF(
            feature_dim,
            num_centers,
            center_init_range,
            initial_bandwidth,
            min_bandwidth,
        )
        self.critic_rbf = GaussianRBF(
            feature_dim,
            num_centers,
            center_init_range,
            initial_bandwidth,
            min_bandwidth,
        )
        self.latent_dim_pi = num_centers
        self.latent_dim_vf = num_centers

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_rbf(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_rbf(features)


class RBFActorMLPCriticExtractor(nn.Module):
    """RBF policy branch with an ordinary MLP value branch for PPO ablation."""

    def __init__(
        self,
        feature_dim: int,
        critic_net_arch: Sequence[int],
        activation_fn: type[nn.Module],
        device: torch.device | str,
        num_centers: int,
        center_init_range: float,
        initial_bandwidth: float,
        min_bandwidth: float,
    ) -> None:
        super().__init__()
        self.actor_rbf = GaussianRBF(
            feature_dim,
            num_centers,
            center_init_range,
            initial_bandwidth,
            min_bandwidth,
        )
        self.critic_extractor = MlpExtractor(
            feature_dim=feature_dim,
            net_arch={"pi": [], "vf": list(critic_net_arch)},
            activation_fn=activation_fn,
            device=device,
        )
        self.latent_dim_pi = num_centers
        self.latent_dim_vf = self.critic_extractor.latent_dim_vf

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_rbf(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_extractor.forward_critic(features)


class _BaseRBFActorCriticPolicy(ActorCriticPolicy):
    """Capture RBF-specific ``policy_kwargs`` before SB3 builds the policy."""

    def __init__(
        self,
        *args: Any,
        rbf_num_centers: int,
        rbf_center_init_range: float,
        rbf_initial_bandwidth: float,
        rbf_min_bandwidth: float,
        **kwargs: Any,
    ) -> None:
        self.rbf_num_centers = rbf_num_centers
        self.rbf_center_init_range = rbf_center_init_range
        self.rbf_initial_bandwidth = rbf_initial_bandwidth
        self.rbf_min_bandwidth = rbf_min_bandwidth
        super().__init__(*args, **kwargs)

    def _rbf_kwargs(self) -> dict[str, float | int]:
        return {
            "num_centers": self.rbf_num_centers,
            "center_init_range": self.rbf_center_init_range,
            "initial_bandwidth": self.rbf_initial_bandwidth,
            "min_bandwidth": self.rbf_min_bandwidth,
        }


class RBFActorCriticPolicy(_BaseRBFActorCriticPolicy):
    """Strict PPO RBF policy: RBF actor and value branches with linear heads."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = RBFActorCriticExtractor(
            feature_dim=self.features_dim,
            **self._rbf_kwargs(),
        )


class RBFActorMLPCriticPolicy(_BaseRBFActorCriticPolicy):
    """PPO ablation: RBF actor paired with the configured ordinary MLP critic."""

    def _build_mlp_extractor(self) -> None:
        critic_net_arch = self.net_arch
        if isinstance(critic_net_arch, dict):
            critic_net_arch = critic_net_arch.get("vf", [])
        self.mlp_extractor = RBFActorMLPCriticExtractor(
            feature_dim=self.features_dim,
            critic_net_arch=critic_net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            **self._rbf_kwargs(),
        )
