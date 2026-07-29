#!/usr/bin/env bash
# SAC 8-env scaling experiment — 3 seeds, serial.
set -euo pipefail
source /home/leopold/anaconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/sac/parallel/parallel_8env.yaml"
SEED_LIST="${SEED_LIST:-42 123 456}"
SLEEP_SECONDS="${SLEEP_SECONDS:-3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/sac_8env_scale_${TIMESTAMP}"
MPL_CONFIG_DIR="$LOG_DIR/matplotlib"

if [[ ! -f "$CONFIG" ]]; then echo "Config not found: $CONFIG" >&2; exit 1; fi

read -r -a SEEDS <<< "$SEED_LIST"
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "SEED_LIST must contain at least one integer seed." >&2; exit 1
fi

HOST_OS="$(uname -s)"
GPU_INFO="none"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)"
fi

CONDA_ENV="${CONDA_ENV:-sac_sb3_demo}"
RUNNER=(conda run -n "$CONDA_ENV" python)

mkdir -p "$LOG_DIR" "$MPL_CONFIG_DIR"
SUMMARY="$LOG_DIR/00_SUMMARY.txt"

{
    echo "SAC 8-env scaling experiment — $(date)"
    echo "host=$HOST_OS  gpu=$GPU_INFO"
    echo "seeds=${SEEDS[*]}  config=$CONFIG"
    echo "runner=${RUNNER[*]}"
    echo
} > "$SUMMARY"

TOTAL=${#SEEDS[@]}
CURRENT=0
FINAL_EXIT=0

for SEED in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    LOGFILE="$LOG_DIR/seed${SEED}.log"
    TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/sac8env_seed${SEED}.XXXXXX")"
    RUN_TAG="sac8env-${TIMESTAMP}-seed${SEED}"

    cp "$CONFIG" "$TMP_CONFIG"
    sed -i.bak -E \
        -e "s/^(  seed: ).*/\\1${SEED}/" \
        -e "s/^(  progress_bar: ).*/\\1false/" \
        -e "s/^(  run_tag: ).*/\\1${RUN_TAG}/" \
        "$TMP_CONFIG"
    rm -f "${TMP_CONFIG}.bak"

    {
        echo "══════════════════════════════════════════"
        echo "[${CURRENT}/${TOTAL}] SAC 8env  seed=${SEED}"
        echo "started: $(date)"
        echo "run_tag: ${RUN_TAG}"
        echo "══════════════════════════════════════════"
    } | tee -a "$LOGFILE"

    START_TS=$(date +%s)
    set +e
    MPLCONFIGDIR="$MPL_CONFIG_DIR" "${RUNNER[@]}" main.py --config "$TMP_CONFIG" >> "$LOGFILE" 2>&1
    EXIT_CODE=$?
    set -e
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    printf "[%s/%s] seed=%-3s  wall=%ss  exit=%s  %s\n" \
        "$CURRENT" "$TOTAL" "$SEED" "$ELAPSED" "$EXIT_CODE" "$(date)" | tee -a "$SUMMARY"
    rm -f "$TMP_CONFIG"

    if [[ $EXIT_CODE -ne 0 ]]; then
        FINAL_EXIT=1
        echo "seed=$SEED failed; continuing." | tee -a "$SUMMARY"
    fi
    if [[ $CURRENT -lt $TOTAL ]]; then
        sleep "$SLEEP_SECONDS"
    fi
done

echo "=== DONE at $(date) ===" | tee -a "$SUMMARY"
cat "$SUMMARY"
exit "$FINAL_EXIT"
