"""Feature extractors for stacked-observation LunarLander SAC experiments."""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class LTCTemporalFeaturesExtractor(BaseFeaturesExtractor):
    """Simple LTC-style feature extractor kept for backward-compatible comparison."""

    def __init__(
        self,
        observation_space: spaces.Box,
        liquid_hidden_dim: int = 128,
        features_dim: int = 256,
        dt: float = 1.0,
    ) -> None:
        if len(observation_space.shape) != 2:
            raise ValueError(f"Expected stacked observation shape (time, obs_dim), got {observation_space.shape}.")

        super().__init__(observation_space, features_dim)
        self.frame_stack = int(observation_space.shape[0])
        self.obs_dim = int(observation_space.shape[1])
        self.liquid_hidden_dim = liquid_hidden_dim
        self.dt = dt

        self.input_layer = nn.Linear(self.obs_dim, liquid_hidden_dim)
        self.recurrent_layer = nn.Linear(liquid_hidden_dim, liquid_hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(liquid_hidden_dim))
        self.log_tau = nn.Parameter(torch.zeros(liquid_hidden_dim))
        self.output_layer = nn.Sequential(
            nn.LayerNorm(liquid_hidden_dim),
            nn.Linear(liquid_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self._reshape_observations(observations)
        h = observations.new_zeros((observations.shape[0], self.liquid_hidden_dim))
        tau = torch.nn.functional.softplus(self.log_tau) + 1e-3

        for t in range(self.frame_stack):
            target = torch.tanh(
                self.input_layer(observations[:, t, :]) + self.recurrent_layer(h) + self.bias
            )
            h = h + self.dt * ((-h + target) / tau)

        return self.output_layer(h)

    def _reshape_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 2:
            return observations.reshape(-1, self.frame_stack, self.obs_dim)
        if observations.ndim == 3:
            return observations
        raise ValueError(f"Expected observations with 2 or 3 dims, got {observations.shape}.")


class CircuitLTCTemporalFeaturesExtractor(BaseFeaturesExtractor):
    """LTC circuit feature extractor matching dx_i=-x_i/tau_i+sum_j f_ij(A_ij-x_i)."""

    def __init__(
        self,
        observation_space: spaces.Box,
        liquid_hidden_dim: int = 128,
        features_dim: int = 256,
        dt: float = 1.0,
        tau_min: float = 0.1,
        ode_unfolds: int = 4,
        reversal_init_scale: float = 1.0,
    ) -> None:
        if len(observation_space.shape) != 2:
            raise ValueError(f"Expected stacked observation shape (time, obs_dim), got {observation_space.shape}.")
        if ode_unfolds < 1:
            raise ValueError("ode_unfolds must be >= 1.")

        super().__init__(observation_space, features_dim)
        self.frame_stack = int(observation_space.shape[0])
        self.obs_dim = int(observation_space.shape[1])
        self.liquid_hidden_dim = liquid_hidden_dim
        self.dt = dt
        self.tau_min = tau_min
        self.ode_unfolds = ode_unfolds

        self.input_projection = nn.Linear(self.obs_dim, liquid_hidden_dim)
        self.gate_layer = nn.Linear(self.obs_dim + 1, liquid_hidden_dim)
        nn.init.constant_(self.gate_layer.bias, -5.0)
        self.raw_tau = nn.Parameter(torch.zeros(liquid_hidden_dim))
        self.reversal_potential = nn.Parameter(
            torch.empty(liquid_hidden_dim, liquid_hidden_dim).uniform_(
                -reversal_init_scale, reversal_init_scale
            )
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(liquid_hidden_dim),
            nn.Linear(liquid_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self._reshape_observations(observations)
        x = torch.tanh(self.input_projection(observations[:, 0, :]))
        sub_dt = self.dt / float(self.ode_unfolds)
        tau = torch.nn.functional.softplus(self.raw_tau) + self.tau_min

        for t in range(self.frame_stack):
            u_t = observations[:, t, :]
            for _ in range(self.ode_unfolds):
                pair_inputs = torch.cat(
                    [
                        x.unsqueeze(-1),
                        u_t.unsqueeze(1).expand(-1, self.liquid_hidden_dim, -1),
                    ],
                    dim=-1,
                )
                # gates[b, i, j] is f_ij(x_j, u; theta)
                gates = torch.sigmoid(self.gate_layer(pair_inputs)).transpose(1, 2)
                conductance = gates.sum(dim=2)
                reversal_drive = (gates * self.reversal_potential.unsqueeze(0)).sum(dim=2)
                # Semi-implicit Euler update of:
                # dx_i/dt = -x_i/tau_i + sum_j f_ij(x_j,u;theta) * (A_ij - x_i)
                numerator = x + sub_dt * reversal_drive
                denominator = 1.0 + sub_dt * ((1.0 / tau) + conductance)
                x = numerator / denominator

        return self.output_layer(x)

    def _reshape_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 2:
            return observations.reshape(-1, self.frame_stack, self.obs_dim)
        if observations.ndim == 3:
            return observations
        raise ValueError(f"Expected observations with 2 or 3 dims, got {observations.shape}.")


class ResidualCircuitLTCFeaturesExtractor(BaseFeaturesExtractor):
    """Circuit LTC encoder fused with a raw stacked-observation residual branch."""

    def __init__(
        self,
        observation_space: spaces.Box,
        liquid_hidden_dim: int = 128,
        features_dim: int = 256,
        raw_features_dim: int = 128,
        fusion_hidden_dim: int = 256,
        dt: float = 1.0,
        tau_min: float = 0.1,
        ode_unfolds: int = 4,
        reversal_init_scale: float = 1.0,
        raw_obs_dim: int | None = None,
    ) -> None:
        if len(observation_space.shape) != 2:
            raise ValueError(f"Expected stacked observation shape (time, obs_dim), got {observation_space.shape}.")

        super().__init__(observation_space, features_dim)
        self.frame_stack = int(observation_space.shape[0])
        self.obs_dim = int(observation_space.shape[1])
        self.raw_obs_dim = raw_obs_dim if raw_obs_dim is not None else self.obs_dim
        if not 1 <= self.raw_obs_dim <= self.obs_dim:
            raise ValueError(f"raw_obs_dim must be in [1, {self.obs_dim}], got {self.raw_obs_dim}.")

        self.ltc_encoder = CircuitLTCTemporalFeaturesExtractor(
            observation_space=observation_space,
            liquid_hidden_dim=liquid_hidden_dim,
            features_dim=features_dim,
            dt=dt,
            tau_min=tau_min,
            ode_unfolds=ode_unfolds,
            reversal_init_scale=reversal_init_scale,
        )
        self.raw_projection = nn.Sequential(
            nn.Linear(self.frame_stack * self.raw_obs_dim, raw_features_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(features_dim + raw_features_dim),
            nn.Linear(features_dim + raw_features_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self._reshape_observations(observations)
        raw_observations = observations[:, :, : self.raw_obs_dim].reshape(observations.shape[0], -1)
        raw_features = self.raw_projection(raw_observations)
        ltc_features = self.ltc_encoder(observations)
        return self.fusion(torch.cat([raw_features, ltc_features], dim=-1))

    def _reshape_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 2:
            return observations.reshape(-1, self.frame_stack, self.obs_dim)
        if observations.ndim == 3:
            return observations
        raise ValueError(f"Expected observations with 2 or 3 dims, got {observations.shape}.")
