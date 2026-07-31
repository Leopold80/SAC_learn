#!/usr/bin/env bash
set -Eeuo pipefail

# 在 Ubuntu 训练机上顺序执行 Go2 PPO 稳定化实验。
# 默认复用已经生成的 runtime profile，避免中断续跑时改变实验设计。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE_PATH="${GO2_RUNTIME_PROFILE:-${REPO_ROOT}/outputs/go2_runtime_benchmark.json}"
FORWARD_CONFIG="${REPO_ROOT}/configs/go2/ppo_stabilized_forward.yaml"
REBENCHMARK=0

usage() {
    echo "Usage: $0 [--rebenchmark]"
    echo
    echo "Environment variables:"
    echo "  PYTHON_BIN             Python executable (default: python3)"
    echo "  GO2_RUNTIME_PROFILE    Runtime profile path"
}

while (($# > 0)); do
    case "$1" in
        --rebenchmark)
            REBENCHMARK=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd -- "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

on_error() {
    local exit_code=$?
    echo >&2
    echo "[失败] 第 ${BASH_LINENO[0]} 行退出，状态码 ${exit_code}。" >&2
    echo "已完成的实验不会被覆盖；修复问题后直接重新运行本脚本即可。" >&2
    exit "${exit_code}"
}
trap on_error ERR

directory_has_files() {
    local directory=$1
    [[ -d "${directory}" ]] && [[ -n "$(find "${directory}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

require_fresh_or_complete() {
    local label=$1
    local output_dir=$2
    local tensorboard_dir=$3

    if [[ -f "${output_dir}/experiment_summary.json" ]]; then
        return 0
    fi
    if directory_has_files "${output_dir}" || directory_has_files "${tensorboard_dir}"; then
        echo "[阻止覆盖] ${label} 存在不完整产物：" >&2
        echo "  output=${output_dir}" >&2
        echo "  tensorboard=${tensorboard_dir}" >&2
        echo "请先人工检查并移动这些目录，再重新运行。" >&2
        return 1
    fi
}

audit_run() {
    local output_dir=$1
    local tensorboard_dir=$2
    "${PYTHON_BIN}" scripts/check_training_artifacts.py \
        "${output_dir}" \
        "${tensorboard_dir}"
}

check_gate_health() {
    local output_dir=$1
    local tensorboard_dir=$2
    "${PYTHON_BIN}" scripts/check_go2_gate.py \
        "${output_dir}" \
        "${tensorboard_dir}"
}

run_gate() {
    local label=$1
    local config=$2
    local output_dir=$3
    local tensorboard_dir=$4

    require_fresh_or_complete "${label}" "${output_dir}" "${tensorboard_dir}"
    if [[ -f "${output_dir}/experiment_summary.json" ]]; then
        echo "[跳过] ${label} 已完成，重新审计现有产物。"
    else
        echo
        echo "========== ${label} =========="
        "${PYTHON_BIN}" main.py --config "${config}"
    fi
    audit_run "${output_dir}" "${tensorboard_dir}"
    check_gate_health "${output_dir}" "${tensorboard_dir}"
}

echo "Repository: ${REPO_ROOT}"
echo "Python:     ${PYTHON_BIN}"

run_gate \
    "Gate 1/2：旧 relative-error 奖励对照" \
    "${REPO_ROOT}/configs/go2/ppo_stabilized_gate_relative.yaml" \
    "${REPO_ROOT}/outputs/go2_ppo_stabilized_gate_relative" \
    "${REPO_ROOT}/runs/go2_ppo_stabilized_gate_relative"

run_gate \
    "Gate 2/2：smooth-baseline 奖励候选" \
    "${REPO_ROOT}/configs/go2/ppo_stabilized_gate_smooth.yaml" \
    "${REPO_ROOT}/outputs/go2_ppo_stabilized_gate_smooth" \
    "${REPO_ROOT}/runs/go2_ppo_stabilized_gate_smooth"

completed_seed_exists=0
for seed in 0 1 2; do
    if [[ -f "${REPO_ROOT}/outputs/go2_ppo_stabilized_forward/seed_${seed}/experiment_summary.json" ]]; then
        completed_seed_exists=1
    fi
done
if ((completed_seed_exists)) && { ((REBENCHMARK)) || [[ ! -f "${PROFILE_PATH}" ]]; }; then
    echo "[阻止混用] 已有正式 seed 完成，不能重新生成或替换 runtime profile。" >&2
    echo "请恢复这些 seed 使用的原始 profile；否则不同 seed 将不再可比。" >&2
    exit 1
fi

if [[ ! -f "${PROFILE_PATH}" ]] || ((REBENCHMARK)); then
    echo
    echo "========== 4070/i7 运行配置 benchmark =========="
    "${PYTHON_BIN}" main.py \
        --config "${FORWARD_CONFIG}" \
        --benchmark-runtime \
        --benchmark-output "${PROFILE_PATH}"
else
    echo
    echo "[复用] 已有 runtime profile：${PROFILE_PATH}"
    echo "如需重新测试，请使用 --rebenchmark；续跑不同种子时不应更换 profile。"
fi

# 在正式训练前显式验证 profile 的 YAML 哈希、主机、Python/CUDA 和 PPO 约束。
"${PYTHON_BIN}" -c \
    "from pathlib import Path; import sys; from sac_experiments.config import load_config; from sac_experiments.runtime_benchmark import apply_runtime_profile; apply_runtime_profile(load_config(Path(sys.argv[1])), Path(sys.argv[2]))" \
    "${FORWARD_CONFIG}" \
    "${PROFILE_PATH}"

for seed in 0 1 2; do
    output_dir="${REPO_ROOT}/outputs/go2_ppo_stabilized_forward/seed_${seed}"
    tensorboard_dir="${REPO_ROOT}/runs/go2_ppo_stabilized_forward/seed_${seed}"
    require_fresh_or_complete "正式训练 seed ${seed}" "${output_dir}" "${tensorboard_dir}"

    if [[ -f "${output_dir}/experiment_summary.json" ]]; then
        echo
        echo "[跳过] seed ${seed} 已完成，重新审计现有产物。"
    else
        echo
        echo "========== 正式训练 seed ${seed} / 2 =========="
        "${PYTHON_BIN}" main.py \
            --config "${FORWARD_CONFIG}" \
            --runtime-profile "${PROFILE_PATH}" \
            --seeds "${seed}"
    fi
    audit_run "${output_dir}" "${tensorboard_dir}"
done

echo
echo "========== 三种子统一验收 =========="
"${PYTHON_BIN}" scripts/summarize_go2_multiseed.py \
    "${REPO_ROOT}/outputs/go2_ppo_stabilized_forward"

echo
echo "[完成] Gate、benchmark、三个种子和产物审计均已执行。"
echo "验收报告：${REPO_ROOT}/outputs/go2_ppo_stabilized_forward/multiseed_summary.json"
