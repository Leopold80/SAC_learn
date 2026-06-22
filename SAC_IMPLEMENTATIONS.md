# SAC Algorithm Implementations on GitHub

This document summarizes authoritative and reusable Soft Actor-Critic (SAC)
implementations found on GitHub, with a focus on practical reuse, maintenance,
documentation, and ease of integration.

## Recommendation Summary

| Rank | Project | Best For | Notes |
| --- | --- | --- | --- |
| 1 | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | Fastest practical reuse | Mature PyTorch RL library with a stable SAC implementation and strong documentation. |
| 2 | [CleanRL](https://github.com/vwxyzjn/cleanrl) | Reading and modifying SAC source code | Single-file implementations, easy to copy into custom experiments. |
| 3 | [Tianshou](https://github.com/thu-ml/tianshou) | Research-oriented RL framework | Modular PyTorch framework supporting SAC, discrete SAC, offline RL, multi-agent RL, and vectorized environments. |
| 4 | [TorchRL](https://github.com/pytorch/rl) | PyTorch-native modular RL development | Official PyTorch ecosystem library with SAC-related components and TensorDict integration. |
| 5 | [Ray RLlib](https://github.com/ray-project/ray) | Distributed or production-scale RL | More heavyweight, but suitable for multi-worker, multi-GPU, or large-scale training. |

## 1. Stable-Baselines3

Repository: <https://github.com/DLR-RM/stable-baselines3>

Stable-Baselines3 is the most practical first choice if the goal is to reuse SAC
quickly. It provides reliable PyTorch implementations of common reinforcement
learning algorithms, including SAC for continuous action spaces.

Strengths:

- Mature and widely used.
- Simple API, similar to scikit-learn style.
- Good documentation and examples.
- Supports custom Gymnasium environments.
- Includes TensorBoard support, callbacks, model saving/loading, and evaluation helpers.
- Better for application work than reimplementing SAC from scratch.

Typical install:

```bash
pip install stable-baselines3
```

Or with extras:

```bash
pip install "stable-baselines3[extra]"
```

Minimal SAC example:

```python
import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("Pendulum-v1")
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("sac_pendulum")
```

Related project: [RL-Baselines3-Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

RL-Baselines3-Zoo is useful when you want training scripts, evaluation scripts,
hyperparameter tuning, pretrained agents, and tuned hyperparameters for common
environments.

## 2. CleanRL

Repository: <https://github.com/vwxyzjn/cleanrl>

CleanRL is a strong choice if the goal is to understand, modify, or directly
copy a compact SAC implementation into a custom project. Its main design is to
keep each algorithm implementation in a single file.

Strengths:

- Very readable compared with larger RL frameworks.
- Good for algorithm study and custom modifications.
- Easy to fork into project-specific experiments.
- Provides SAC implementations such as continuous-action SAC.

Best fit:

- You want to modify the actor or critic architecture.
- You want to change the entropy loss, replay logic, logging, or update schedule.
- You want a minimal but serious implementation instead of a large framework.

## 3. Tianshou

Repository: <https://github.com/thu-ml/tianshou>

Tianshou is a PyTorch RL framework with a broad algorithm set and a more
research-oriented design. It supports SAC and discrete SAC, along with many
other online, offline, and multi-agent RL algorithms.

Strengths:

- Modular architecture.
- Supports online RL, offline RL, multi-agent RL, and model-based experiments.
- Supports vectorized environments and logging.
- Good fit for longer-term RL research infrastructure.

Tradeoff:

- More framework concepts to learn than Stable-Baselines3 or CleanRL.

## 4. TorchRL

Repository: <https://github.com/pytorch/rl>

TorchRL is the official PyTorch ecosystem library for reinforcement learning.
It is more modular and lower-level than Stable-Baselines3, and integrates with
PyTorch concepts such as TensorDict.

Strengths:

- Strong PyTorch ecosystem alignment.
- Good for building custom RL pipelines from reusable primitives.
- Suitable if the project already relies heavily on modern PyTorch tooling.

Tradeoff:

- Higher learning curve for quick experiments.
- Less convenient than Stable-Baselines3 for simply training a SAC baseline.

## 5. Ray RLlib

Repository: <https://github.com/ray-project/ray>

Documentation: <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html>

RLlib is best when SAC needs to run at larger scale, such as with many parallel
environment workers, distributed learners, or production-style training
infrastructure.

Strengths:

- Designed for distributed RL.
- Supports scalable sampling and learning.
- Integrates with the broader Ray ecosystem.

Tradeoff:

- Too heavy if the goal is only to run a local SAC baseline.

## Historical or Reference-Only Repositories

### haarnoja/sac

Repository: <https://github.com/haarnoja/sac>

This is the original SAC author's repository and is historically important.
However, it is no longer maintained and uses an older TensorFlow-based stack.
It is useful as a reference, but not recommended as the primary dependency for
new projects.

### pranz24/pytorch-soft-actor-critic

Repository: <https://github.com/pranz24/pytorch-soft-actor-critic>

This repository is a popular PyTorch SAC implementation, but it has been
archived and is now read-only. It can still be useful for comparison or study,
but should not be used as the main dependency for a new project.

## Practical Choice

For most projects:

1. Use **Stable-Baselines3** if you want to run SAC quickly and reliably.
2. Use **RL-Baselines3-Zoo** if you also want ready-made training scripts,
   tuned hyperparameters, and evaluation workflows.
3. Use **CleanRL** if you want a compact SAC implementation that is easy to read
   and modify.
4. Use **Tianshou** if you are building a broader RL research framework.
5. Use **RLlib** only when distributed training or production-scale RL matters.

Default recommendation:

```text
Stable-Baselines3 + RL-Baselines3-Zoo for fast reuse.
CleanRL for source-level customization.
```
