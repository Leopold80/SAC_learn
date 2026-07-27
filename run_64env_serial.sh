#!/bin/bash
set -euo pipefail

cd ~/sac_test
PYTHON=/home/leopold/anaconda3/envs/sac_sb3_demo/bin/python
CONFIG=configs/parallel_64env.yaml
SEEDS=(42 123 456)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/experiment_64env_serial_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

SUMMARY="${LOG_DIR}/00_SUMMARY.txt"
{
    echo "Experiment: 64env serial 3 seeds started at $(date)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
    echo ""
} > "$SUMMARY"

TOTAL=${#SEEDS[@]}
CURRENT=0
for SEED in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    LOGFILE="${LOG_DIR}/64env_seed${SEED}.log"
    TMP_CFG="/tmp/sac_exp_64env_seed${SEED}.yaml"

    cp "$CONFIG" "$TMP_CFG"
    sed -i "s/seed: [0-9]*/seed: ${SEED}/" "$TMP_CFG"
    sed -i "s/seed42/64env_seed${SEED}/" "$TMP_CFG"

    echo "[$CURRENT/$TOTAL] seed=${SEED} started: $(date)" | tee -a "$SUMMARY"

    START_TS=$(date +%s)
    $PYTHON main.py --config "$TMP_CFG" > "$LOGFILE" 2>&1
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    echo "[$CURRENT/$TOTAL] seed=${SEED} wall=${ELAPSED}s exit=$? finished: $(date)" | tee -a "$SUMMARY"
    rm -f "$TMP_CFG"
    sleep 3
done

echo "" | tee -a "$SUMMARY"
echo "=== ALL COMPLETE at $(date) ===" | tee -a "$SUMMARY"
cat "$SUMMARY" >> "$SUMMARY"
