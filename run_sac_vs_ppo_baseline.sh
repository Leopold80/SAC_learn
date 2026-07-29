#!/usr/bin/env bash
# SAC vs PPO baseline comparison — 3 seeds each, serial.
# PPO CPU and GPU use the SAME config; only device differs.
set -euo pipefail
source /home/leopold/anaconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SAC_CONFIG="configs/sac/baseline.yaml"
PPO_CONFIG="configs/ppo/parallel.yaml"
SEED_LIST="${SEED_LIST:-42 123 456}"
SLEEP_SECONDS="${SLEEP_SECONDS:-3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/sac_vs_ppo_baseline_${TIMESTAMP}"
MPL_CONFIG_DIR="$LOG_DIR/matplotlib"

for CFG in "$SAC_CONFIG" "$PPO_CONFIG"; do
    if [[ ! -f "$CFG" ]]; then echo "Config not found: $CFG" >&2; exit 1; fi
done

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
    echo "SAC vs PPO (CPU+GPU, same config) baseline — $(date)"
    echo "host=$HOST_OS  gpu=$GPU_INFO"
    echo "seeds=${SEEDS[*]}"
    echo "runner=${RUNNER[*]}"
    echo "---"
    echo "SAC:      $SAC_CONFIG"
    echo "PPO_CPU:  $PPO_CONFIG  (device=cpu)"
    echo "PPO_GPU:  $PPO_CONFIG  (device=cuda)"
    echo
} > "$SUMMARY"

run_one() {
    local label="$1" config="$2" seed="$3" current="$4" total="$5"
    local device="$6"  # "cuda" or "cpu"
    local logfile="$LOG_DIR/${label}_seed${seed}.log"
    local tmp_config run_tag start_ts end_ts elapsed exit_code

    tmp_config="$(mktemp "${TMPDIR:-/tmp}/baseline_${label}_seed${seed}.XXXXXX")"
    run_tag="${label}-${TIMESTAMP}-seed${seed}"

    cp "$config" "$tmp_config"
    sed -i.bak -E \
        -e "s/^(  seed: ).*/\\1${seed}/" \
        -e "s/^(  device: ).*/\\1${device}/" \
        -e "s/^(  allow_cpu: ).*/\\1$([ "$device" = "cpu" ] && echo "true" || echo "false")/" \
        -e "s/^(  progress_bar: ).*/\\1false/" \
        -e "s/^(  run_tag: ).*/\\1${run_tag}/" \
        "$tmp_config"
    rm -f "${tmp_config}.bak"

    {
        echo "══════════════════════════════════════════"
        echo "[${current}/${total}] ${label}  seed=${seed}  device=${device}"
        echo "started: $(date)"
        echo "run_tag: ${run_tag}"
        echo "══════════════════════════════════════════"
    } | tee -a "$logfile"

    start_ts=$(date +%s)
    set +e
    MPLCONFIGDIR="$MPL_CONFIG_DIR" "${RUNNER[@]}" main.py --config "$tmp_config" >> "$logfile" 2>&1
    exit_code=$?
    set -e
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    printf "[%s/%s] %-9s seed=%-3s  device=%-4s  wall=%ss  exit=%s  %s\n" \
        "$current" "$total" "$label" "$seed" "$device" "$elapsed" "$exit_code" "$(date)" | tee -a "$SUMMARY"
    rm -f "$tmp_config"
    return "$exit_code"
}

EXPERIMENTS=(
    "SAC:$SAC_CONFIG:cuda"
    "PPO_CPU:$PPO_CONFIG:cpu"
    "PPO_GPU:$PPO_CONFIG:cuda"
)

TOTAL=$((${#EXPERIMENTS[@]} * ${#SEEDS[@]}))
CURRENT=0
FINAL_EXIT=0

for ENTRY in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r LABEL CFG DEV <<< "$ENTRY"
    for SEED in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        run_one "$LABEL" "$CFG" "$SEED" "$CURRENT" "$TOTAL" "$DEV" || FINAL_EXIT=1
        if [[ $CURRENT -lt $TOTAL ]]; then
            sleep "$SLEEP_SECONDS"
        fi
    done
done

echo "=== DONE at $(date) ===" | tee -a "$SUMMARY"
cat "$SUMMARY"
exit "$FINAL_EXIT"
