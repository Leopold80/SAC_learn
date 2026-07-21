#!/bin/bash
set -euo pipefail

CONDA_BASE="$HOME/anaconda3"
ENV_NAME="sac_sb3_demo"
REPO_DIR="$HOME/sac_learn"
LOG_DIR="$REPO_DIR/experiment_logs"
PERF_DIR="$LOG_DIR/perf"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/master_${TIMESTAMP}.log"

# 5 configs × 5 seeds = 25 runs
# 顺序: seed0跑完5个 → seed1跑完5个 → ...
CONFIGS=(
  "baseline.yaml"
  "parallel_baseline.yaml"
  "ppo_parallel.yaml"
  "ppo_parallel_large.yaml"
  "parallel_baseline_large_batch.yaml"
)
NAMES=(
  "单SAC"
  "多SAC"
  "多PPO_CPU"
  "多PPO_GPU大网"
  "多大batch_SAC"
)

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$PERF_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

# ── 性能监控 ──
PERF_PID=""
start_perf() {
  local tag="$1"
  local csv="$PERF_DIR/${tag}_${TIMESTAMP}.csv"
  echo "ts,gpu%,gpu_mem,cpu%,ram_mb" > "$csv"
  (
    while true; do
      local ts=$(date +%H:%M:%S)
      local gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "N/A,N/A")
      local cpu=$(top -bn1 | grep "Cpu(s)" | awk '{printf "%.1f", $2+$4}' 2>/dev/null || echo "N/A")
      local ram=$(free -m | awk '/Mem:/{print $3}' 2>/dev/null || echo "N/A")
      echo "$ts,$gpu,$cpu,$ram" >> "$csv"
      sleep 30
    done
  ) &
  PERF_PID=$!
}

stop_perf() {
  [ -n "$PERF_PID" ] && kill $PERF_PID 2>/dev/null || true
  PERF_PID=""
}

trap stop_perf EXIT

# ── 单次运行 ──
run_one() {
  local seed="$1" config_file="$2" name="$3" tag="$4"
  local config_path="$REPO_DIR/configs/$config_file"
  
  log "  ▶ $name (seed=$seed)"
  
  # 备份原文件 + 修改 seed / output / tensorboard / run_tag
  cp "$config_path" "${config_path}.bak"
  sed -i "s/^  seed: .*/  seed: $seed/" "$config_path"
  sed -i "s|^  directory: outputs/|  directory: outputs/${tag}/|" "$config_path"
  sed -i "s|^  tensorboard_log: runs/|  tensorboard_log: runs/${tag}/|" "$config_path"
  sed -i "s/run_tag:.*/run_tag: seed${seed}/" "$config_path"
  
  local start_ts=$(date +%s)
  export MPLCONFIGDIR=/tmp/matplotlib-sac-demo
  
  python "$REPO_DIR/main.py" --config "$config_path" >> "$MASTER_LOG" 2>&1
  local rc=$?
  
  local elapsed=$(($(date +%s) - start_ts))
  local m=$((elapsed / 60))
  local s=$((elapsed % 60))
  
  # 恢复原文件
  mv "${config_path}.bak" "$config_path"
  
  if [ $rc -eq 0 ]; then
    log "  ✅ ${name} seed=${seed} (${m}m${s}s)"
  else
    log "  ❌ ${name} seed=${seed} FAILED (${m}m${s}s)"
  fi
  return $rc
}

# ── 主流程 ──
TOTAL=$((5 * 5))
CURRENT=0

log "═══════════════════════════════════════"
log "多种子实验启动 | $(date)"
log "配置: 5 种子 × 5 实验 = $TOTAL 次"
log "策略: 每个种子跑完5个实验再换下一个"
log "═══════════════════════════════════════"

for seed in 0 1 2 3 4; do
  log ""
  log "━━━ SEED $seed ━━━"
  
  for i in "${!CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    config="${CONFIGS[$i]}"
    name="${NAMES[$i]}"
    tag="seed${seed}_$(echo $config | sed 's/.yaml//')"
    perf_tag="s${seed}_$i"
    
    log "  [$CURRENT/$TOTAL]"
    start_perf "$perf_tag"
    run_one "$seed" "$config" "$name" "$tag" || log "  ⚠️ 继续下一个"
    stop_perf
  done
done

log ""
log "═══════════════════════════════════════"
log "全部 $TOTAL 次实验完成 | $(date)"
log "═══════════════════════════════════════"
