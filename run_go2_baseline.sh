#!/usr/bin/env bash
# 3-seed SAC vs PPO baseline comparison for Unitree Go2 locomotion
set -euo pipefail

CONDA_ENV="cybernetic_env"
SAC_CONFIG="configs/go2/sac_baseline.yaml"
PPO_CONFIG="configs/go2/ppo_baseline.yaml"
SEEDS=(42 123 456)

echo "=== Go2 SAC vs PPO Baseline — 3 seeds ==="
echo "Conda env: $CONDA_ENV"
echo ""

for seed in "${SEEDS[@]}"; do
  echo "--- SAC seed=$seed ---"
  conda run -n "$CONDA_ENV" python main.py --config "$SAC_CONFIG" --seed "$seed"
  echo "--- PPO seed=$seed ---"
  conda run -n "$CONDA_ENV" python main.py --config "$PPO_CONFIG" --seed "$seed"
done

echo "=== Done ==="
