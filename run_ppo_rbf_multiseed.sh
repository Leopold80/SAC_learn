#!/usr/bin/env bash
# Run the PPO RBF comparison serially across seeds on macOS or Ubuntu.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/ppo/rbf_comparison.yaml}"
SEED_LIST="${SEED_LIST:-42 123 456}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-ppo-rbf}"
SLEEP_SECONDS="${SLEEP_SECONDS:-3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/ppo_rbf_multiseed_${TIMESTAMP}"
MPL_CONFIG_DIR="$LOG_DIR/matplotlib"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    exit 1
fi

read -r -a SEEDS <<< "$SEED_LIST"
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "SEED_LIST must contain at least one integer seed." >&2
    exit 1
fi

HOST_OS="$(uname -s)"
if [[ -n "${DEVICE:-}" ]]; then
    SELECTED_DEVICE="$DEVICE"
elif [[ "$HOST_OS" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    SELECTED_DEVICE="cuda"
else
    SELECTED_DEVICE="cpu"
fi

if [[ "$SELECTED_DEVICE" == cuda* ]]; then
    ALLOW_CPU=false
else
    ALLOW_CPU=true
fi

if [[ -n "${PYTHON:-}" ]]; then
    RUNNER=("$PYTHON")
elif command -v conda >/dev/null 2>&1; then
    CONDA_ENV="${CONDA_ENV:-sac_sb3_demo}"
    RUNNER=(conda run -n "$CONDA_ENV" python)
else
    RUNNER=(python3)
fi

GPU_INFO="none"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)"
fi

mkdir -p "$LOG_DIR" "$MPL_CONFIG_DIR"
SUMMARY="$LOG_DIR/00_SUMMARY.txt"
{
    echo "PPO RBF multi-seed comparison started at $(date)"
    echo "host_os=$HOST_OS"
    echo "device=$SELECTED_DEVICE allow_cpu=$ALLOW_CPU"
    echo "gpu=$GPU_INFO"
    echo "config=$CONFIG"
    echo "seeds=${SEEDS[*]}"
    echo "runner=${RUNNER[*]}"
    echo
} > "$SUMMARY"

TOTAL=${#SEEDS[@]}
CURRENT=0
for SEED in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    LOGFILE="$LOG_DIR/seed${SEED}.log"
    TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/ppo_rbf_seed${SEED}.XXXXXX")"
    RUN_TAG="${RUN_TAG_PREFIX}-${TIMESTAMP}-seed${SEED}"

    cp "$CONFIG" "$TMP_CONFIG"
    sed -i.bak -E \
        -e "s/^(  seed: ).*/\\1${SEED}/" \
        -e "s/^(  device: ).*/\\1${SELECTED_DEVICE}/" \
        -e "s/^(  allow_cpu: ).*/\\1${ALLOW_CPU}/" \
        -e "s/^(  progress_bar: ).*/\\1false/" \
        -e "s/^(  run_tag: ).*/\\1${RUN_TAG}/" \
        "$TMP_CONFIG"
    rm -f "${TMP_CONFIG}.bak"

    {
        echo "=============================================="
        echo "[$CURRENT/$TOTAL] seed=$SEED device=$SELECTED_DEVICE"
        echo "started: $(date)"
        echo "run_tag: $RUN_TAG"
        echo "=============================================="
    } | tee -a "$LOGFILE"

    START_TS=$(date +%s)
    set +e
    MPLCONFIGDIR="$MPL_CONFIG_DIR" "${RUNNER[@]}" main.py --config "$TMP_CONFIG" >> "$LOGFILE" 2>&1
    EXIT_CODE=$?
    set -e
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    {
        echo "[$CURRENT/$TOTAL] seed=$SEED wall=${ELAPSED}s exit=$EXIT_CODE $(date)"
    } | tee -a "$SUMMARY"
    rm -f "$TMP_CONFIG"

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "Seed $SEED failed; later seeds will still run. See $LOGFILE" | tee -a "$SUMMARY"
    fi
    if [[ $CURRENT -lt $TOTAL ]]; then
        sleep "$SLEEP_SECONDS"
    fi
done

echo "=== COMPLETE at $(date) ===" | tee -a "$SUMMARY"
cat "$SUMMARY"
