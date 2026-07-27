"""SAC policies whose actor and/or twin Q critics use Gaussian RBF features."""

from __future__ import annotations

from typing import Any

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, ContinuousCritic, SACPolicy
from torch import nn

from sac_experiments.rbf_policies import GaussianRBF


class RBFActor(Actor):
    """SAC actor with an RBF latent policy feature map and Gaussian action heads."""

    def __init__(
        self,
        *args: Any,
        rbf_num_centers: int,
        rbf_center_init_range: float,
        rbf_initial_bandwidth: float,
        rbf_min_bandwidth: float,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("use_sde", False):
            raise ValueError("RBF SAC policies currently require use_sde=False.")
        super().__init__(*args, **kwargs)
        self.rbf_num_centers = rbf_num_centers
        self.rbf_center_init_range = rbf_center_init_range
        self.rbf_initial_bandwidth = rbf_initial_bandwidth
        self.rbf_min_bandwidth = rbf_min_bandwidth

        action_dim = get_action_dim(self.action_space)
        self.net_arch = []
        self.latent_pi = GaussianRBF(
            input_dim=self.features_dim,
            num_centers=rbf_num_centers,
            center_init_range=rbf_center_init_range,
            initial_bandwidth=rbf_initial_bandwidth,
            min_bandwidth=rbf_min_bandwidth,
        )
        self.mu = nn.Linear(rbf_num_centers, action_dim)
        self.log_std = nn.Linear(rbf_num_centers, action_dim)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "rbf_num_centers": self.rbf_num_centers,
                "rbf_center_init_range": self.rbf_center_init_range,
                "rbf_initial_bandwidth": self.rbf_initial_bandwidth,
                "rbf_min_bandwidth": self.rbf_min_bandwidth,
            }
        )
        return data


class RBFContinuousCritic(ContinuousCritic):
    """Twin-Q critic whose independent Q heads approximate ``Q(s, a)`` with RBFs."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        *,
        rbf_num_centers: int,
        rbf_center_init_range: float,
        rbf_initial_bandwidth: float,
        rbf_min_bandwidth: float,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )
        self.rbf_num_centers = rbf_num_centers
        self.rbf_center_init_range = rbf_center_init_range
        self.rbf_initial_bandwidth = rbf_initial_bandwidth
        self.rbf_min_bandwidth = rbf_min_bandwidth

        q_input_dim = features_dim + get_action_dim(action_space)
        self.q_networks = []
        for index in range(n_critics):
            q_network = nn.Sequential(
                GaussianRBF(
                    input_dim=q_input_dim,
                    num_centers=rbf_num_centers,
                    center_init_range=rbf_center_init_range,
                    initial_bandwidth=rbf_initial_bandwidth,
                    min_bandwidth=rbf_min_bandwidth,
                ),
                nn.Linear(rbf_num_centers, 1),
            )
            setattr(self, f"qf{index}", q_network)
            self.q_networks.append(q_network)


class _BaseRBFSACPolicy(SACPolicy):
    """Stores RBF settings before SB3 constructs actor and critic modules."""

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
            "rbf_num_centers": self.rbf_num_centers,
            "rbf_center_init_range": self.rbf_center_init_range,
            "rbf_initial_bandwidth": self.rbf_initial_bandwidth,
            "rbf_min_bandwidth": self.rbf_min_bandwidth,
        }

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(self._rbf_kwargs())
        return data

    def make_actor(
        self,
        features_extractor: BaseFeaturesExtractor | None = None,
    ) -> RBFActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs,
            features_extractor,
        )
        actor_kwargs["net_arch"] = []
        return RBFActor(**actor_kwargs, **self._rbf_kwargs()).to(self.device)


class RBFSACPolicy(_BaseRBFSACPolicy):
    """Strict SAC RBF policy: RBF actor and independent RBF twin-Q critics."""

    def make_critic(
        self,
        features_extractor: BaseFeaturesExtractor | None = None,
    ) -> RBFContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs,
            features_extractor,
        )
        critic_kwargs["net_arch"] = []
        return RBFContinuousCritic(**critic_kwargs, **self._rbf_kwargs()).to(self.device)


class RBFSACActorMLPCriticPolicy(_BaseRBFSACPolicy):
    """SAC ablation with an RBF actor and ordinary MLP online/target twin-Q critics."""

    # Inherit SACPolicy.make_critic() so both online and target critics remain
    # standard SB3 MLP twin-Q modules. Only make_actor() is replaced above.
