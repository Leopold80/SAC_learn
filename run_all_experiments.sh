#!/bin/bash
set -euo pipefail

cd /home/leopold/sac_test
PYTHON=/home/leopold/anaconda3/envs/sac_sb3_demo/bin/python

# ── experiment matrix ──────────────────────────────────────────────
declare -A CONFIGS=(
    ["baseline"]="configs/baseline.yaml"
    ["parallel_2env"]="configs/parallel_2env.yaml"
    ["parallel_8env"]="configs/parallel_baseline.yaml"
)
SEEDS=(42 123 456)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/experiment_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

SUMMARY="${LOG_DIR}/00_SUMMARY.txt"
{
    echo "Experiment matrix started at $(date)"
    echo "Configs: ${!CONFIGS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
    echo ""
} > "$SUMMARY"

TOTAL=$((${#CONFIGS[@]} * ${#SEEDS[@]}))
CURRENT=0

for NAME in baseline parallel_2env parallel_8env; do
    CFG="${CONFIGS[$NAME]}"
    for SEED in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        LOGFILE="${LOG_DIR}/${NAME}_seed${SEED}.log"
        TMP_CFG="/tmp/sac_exp_${NAME}_seed${SEED}.yaml"

        # generate per-run config
        cp "$CFG" "$TMP_CFG"
        sed -i "s/seed: [0-9]*/seed: ${SEED}/"  "$TMP_CFG"
        sed -i "s/run_tag: null/run_tag: seed${SEED}/" "$TMP_CFG"
        sed -i "s/progress_bar: true/progress_bar: false/" "$TMP_CFG"

        {
            echo "=============================================="
            echo "  [$CURRENT/$TOTAL] $NAME  seed=$SEED"
            echo "  started: $(date)"
            echo "=============================================="
        } | tee -a "$LOGFILE"

        START_TS=$(date +%s)
        set +e
        $PYTHON main.py --config "$TMP_CFG" >> "$LOGFILE" 2>&1
        EXIT_CODE=$?
        set -e
        END_TS=$(date +%s)
        ELAPSED=$((END_TS - START_TS))

        {
            echo ""
            echo "=============================================="
            echo "  finished: $(date)"
            echo "  wall time: ${ELAPSED}s ($((ELAPSED/60))m $((ELAPSED%60))s)"
            echo "  exit code: ${EXIT_CODE}"
            echo "=============================================="
            echo ""
        } | tee -a "$LOGFILE"

        # append to summary
        {
            echo "[$CURRENT/$TOTAL] $NAME seed=$SEED  wall=${ELAPSED}s  exit=$EXIT_CODE  $(date)"
        } >> "$SUMMARY"

        rm -f "$TMP_CFG"

        # brief pause between runs
        sleep 3
    done
done

{
    echo ""
    echo "=== ALL EXPERIMENTS COMPLETE ==="
    echo "Finished at $(date)"
} >> "$SUMMARY"

cat "$SUMMARY"
