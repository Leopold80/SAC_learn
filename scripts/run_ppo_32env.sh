#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/go2/ppo_32env.yaml"
OUTPUT_DIR="outputs/go2_ppo_32env_fixed"
RUN_DIR="runs/go2_ppo_32env_fixed"
LOG_FILE="train_ppo_32env.log"

cd "$(dirname "$0")/.."

echo "=== Cleaning old outputs ==="
rm -rf "$OUTPUT_DIR" "$RUN_DIR"

echo "=== Starting PPO 32env training ==="
source ~/anaconda3/bin/activate sac_sb3_demo
export PYTHONPATH=.

nohup python3 main.py --config "$CONFIG" > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID=$PID"
echo "Log: $LOG_FILE"
echo "Config: $CONFIG"
