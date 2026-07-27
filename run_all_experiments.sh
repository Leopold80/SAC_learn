#!/usr/bin/env bash
# Matched SAC sampling-scale experiment for macOS and NVIDIA Ubuntu hosts.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# All four entries share the same algorithm and network. The runner patches
# their temporary YAMLs to a common transition/evaluation budget while retaining
# the established SAC warmup, so n_envs is the only intended experimental factor.
EXPERIMENTS="${EXPERIMENTS:-baseline parallel_2env parallel_8env parallel_64env}"
SEED_LIST="${SEED_LIST:-42 123 456}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-sac-sampling-scale}"
SLEEP_SECONDS="${SLEEP_SECONDS:-3}"
MATCHED_TIMESTEPS="${MATCHED_TIMESTEPS:-499968}"
MATCHED_EVAL_FREQUENCY="${MATCHED_EVAL_FREQUENCY:-9984}"
MATCHED_LEARNING_STARTS="${MATCHED_LEARNING_STARTS:-10000}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/sac_sampling_scale_${TIMESTAMP}"
MPL_CONFIG_DIR="$LOG_DIR/matplotlib"

first_existing_config() {
    for CANDIDATE in "$@"; do
        if [[ -f "$CANDIDATE" ]]; then
            printf '%s\n' "$CANDIDATE"
            return 0
        fi
    done
    return 1
}

config_for_experiment() {
    case "$1" in
        baseline)
            first_existing_config 'configs/sac/baseline.yaml' 'configs/baseline.yaml'
            ;;
        parallel_2env)
            first_existing_config 'configs/sac/parallel/parallel_2env.yaml' 'configs/parallel_2env.yaml'
            ;;
        parallel_8env)
            first_existing_config 'configs/sac/parallel/parallel_8env.yaml' 'configs/parallel_baseline.yaml'
            ;;
        parallel_64env)
            first_existing_config 'configs/sac/parallel/parallel_64env.yaml' 'configs/parallel_64env.yaml'
            ;;
        *)
            echo "Unknown experiment '$1'. Use baseline, parallel_2env, parallel_8env, or parallel_64env." >&2
            return 1
            ;;
    esac
}

read -r -a EXPERIMENT_NAMES <<< "$EXPERIMENTS"
read -r -a SEEDS <<< "$SEED_LIST"
if [[ ${#EXPERIMENT_NAMES[@]} -eq 0 || ${#SEEDS[@]} -eq 0 ]]; then
    echo "EXPERIMENTS and SEED_LIST must both contain at least one value." >&2
    exit 1
fi

for NAME in "${EXPERIMENT_NAMES[@]}"; do
    CONFIG_PATH="$(config_for_experiment "$NAME")"
    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "Config file not found for $NAME: $CONFIG_PATH" >&2
        exit 1
    fi
done

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
    echo "Matched SAC sampling-scale experiment started at $(date)"
    echo "host_os=$HOST_OS"
    echo "device=$SELECTED_DEVICE allow_cpu=$ALLOW_CPU"
    echo "gpu=$GPU_INFO"
    echo "experiments=${EXPERIMENT_NAMES[*]}"
    echo "seeds=${SEEDS[*]}"
    echo "timesteps=$MATCHED_TIMESTEPS eval_frequency=$MATCHED_EVAL_FREQUENCY learning_starts=$MATCHED_LEARNING_STARTS"
    echo "runner=${RUNNER[*]}"
    echo
} > "$SUMMARY"

TOTAL=$((${#EXPERIMENT_NAMES[@]} * ${#SEEDS[@]}))
CURRENT=0
FINAL_EXIT=0
for NAME in "${EXPERIMENT_NAMES[@]}"; do
    CONFIG_PATH="$(config_for_experiment "$NAME")"
    for SEED in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        LOGFILE="$LOG_DIR/${NAME}_seed${SEED}.log"
        TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/sac_scale_${NAME}_seed${SEED}.XXXXXX")"
        RUN_TAG="${RUN_TAG_PREFIX}-${TIMESTAMP}-${NAME}-seed${SEED}"

        cp "$CONFIG_PATH" "$TMP_CONFIG"
        sed -i.bak -E \
            -e "s/^(  timesteps: ).*/\\1${MATCHED_TIMESTEPS}/" \
            -e "s/^(  seed: ).*/\\1${SEED}/" \
            -e "s/^(  device: ).*/\\1${SELECTED_DEVICE}/" \
            -e "s/^(  allow_cpu: ).*/\\1${ALLOW_CPU}/" \
            -e "s/^(  progress_bar: ).*/\\1false/" \
            -e "s/^(  frequency: ).*/\\1${MATCHED_EVAL_FREQUENCY}/" \
            -e "s/^(  run_tag: ).*/\\1${RUN_TAG}/" \
            -e "s/^(  learning_starts: ).*/\\1${MATCHED_LEARNING_STARTS}/" \
            "$TMP_CONFIG"
        rm -f "${TMP_CONFIG}.bak"

        {
            echo "=============================================="
            echo "[$CURRENT/$TOTAL] $NAME seed=$SEED device=$SELECTED_DEVICE"
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

        echo "[$CURRENT/$TOTAL] $NAME seed=$SEED wall=${ELAPSED}s exit=$EXIT_CODE $(date)" | tee -a "$SUMMARY"
        rm -f "$TMP_CONFIG"

        if [[ $EXIT_CODE -ne 0 ]]; then
            FINAL_EXIT=1
            echo "$NAME seed=$SEED failed; continuing with the remaining matrix entries." | tee -a "$SUMMARY"
        fi
        if [[ $CURRENT -lt $TOTAL ]]; then
            sleep "$SLEEP_SECONDS"
        fi
    done
done

echo "=== COMPLETE at $(date) ===" | tee -a "$SUMMARY"
cat "$SUMMARY"
exit "$FINAL_EXIT"
