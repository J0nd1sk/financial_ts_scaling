#!/bin/bash
#
# Phase 6A HPO Runner Script - Large Models Only (200M, 2B)
# Runs 200M and 2B HPO experiments (2M/20M run separately for parallelization)
#
# Experiments (in order):
#   1-3: 200M_h{1,3,5}
#   4-6: 2B_h{1,3,5}
#
# Note: 2M and 20M experiments excluded - run via run_phase6a_2M.sh and run_phase6a_20M.sh
#
# Usage:
#   tmux new -s hpo
#   ./scripts/run_phase6a_large.sh
#   # Ctrl+B, D to detach
#   # tmux attach -t hpo to reattach
#
# Options:
#   --dry-run    Print what would execute without running
#

set -o pipefail

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python"
EXPERIMENTS_DIR="${PROJECT_ROOT}/experiments/phase6a"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/hpo"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/phase6a_large_${TIMESTAMP}.log"
HARDWARE_LOG="${LOG_DIR}/hardware_monitor_large_${TIMESTAMP}.csv"
MONITOR_INTERVAL=300  # 5 minutes

# Parse arguments
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
    esac
done

# Background monitor PID (for cleanup)
MONITOR_PID=""

# Large model experiments only (200M and 2B)
EXPERIMENTS=(
    "hpo_200M_h1_threshold_1pct.py"
    "hpo_200M_h3_threshold_1pct.py"
    "hpo_200M_h5_threshold_1pct.py"
    "hpo_2B_h1_threshold_1pct.py"
    "hpo_2B_h3_threshold_1pct.py"
    "hpo_2B_h5_threshold_1pct.py"
)

# Results tracking
declare -a RESULTS
declare -a DURATIONS
declare -a SKIPPED

# ============================================================
# HELPER FUNCTIONS
# ============================================================

get_experiment_name() {
    local script="$1"
    local base="${script%.py}"
    echo "phase6a_${base#hpo_}"
}

is_complete() {
    local exp_name="$1"
    local output_dir="${OUTPUT_DIR}/${exp_name}"

    if [ ! -d "$output_dir" ]; then
        return 1
    fi

    local best_file=$(ls ${output_dir}/${exp_name}_*_best.json 2>/dev/null | head -1)
    if [ -z "$best_file" ] || [ ! -f "$best_file" ]; then
        return 1
    fi

    local trials=$("${PYTHON}" -c "import json; print(json.load(open('$best_file')).get('n_trials_completed', 0))" 2>/dev/null)
    if [ "$trials" = "50" ]; then
        return 0
    fi

    return 1
}

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================

preflight_check() {
    echo "Running pre-flight checks..."
    local failures=0

    echo -n "  MPS (Apple Silicon GPU): "
    if "${PYTHON}" -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        echo "OK"
    else
        echo "NOT AVAILABLE"
        ((failures++))
    fi

    echo -n "  Temperature monitoring: "
    if sudo -n true 2>/dev/null; then
        TEMP=$("${PYTHON}" -c "from src.training.thermal import get_macos_temperature; print(get_macos_temperature())" 2>/dev/null)
        if [ "$TEMP" != "-1.0" ] && [ -n "$TEMP" ]; then
            echo "OK (${TEMP}C)"
        else
            echo "limited (will continue)"
        fi
    else
        echo "limited - run 'sudo -v' for thermal monitoring"
    fi

    echo -n "  Memory (>16GB free for large models): "
    FREE_MEM_GB=$("${PYTHON}" -c "import psutil; print(f'{psutil.virtual_memory().available / (1024**3):.1f}')" 2>/dev/null)
    if [ -n "$FREE_MEM_GB" ]; then
        FREE_MEM_INT=${FREE_MEM_GB%.*}
        if [ "$FREE_MEM_INT" -ge 16 ]; then
            echo "OK (${FREE_MEM_GB}GB)"
        elif [ "$FREE_MEM_INT" -ge 8 ]; then
            echo "WARNING (${FREE_MEM_GB}GB - may be tight for 2B)"
        else
            echo "INSUFFICIENT (${FREE_MEM_GB}GB)"
            ((failures++))
        fi
    else
        echo "could not check"
    fi

    echo -n "  Data file: "
    if [ -f "${PROJECT_ROOT}/data/processed/v1/SPY_dataset_a25.parquet" ]; then
        echo "OK"
    else
        echo "NOT FOUND"
        ((failures++))
    fi

    echo -n "  Disk space (>10GB free): "
    FREE_DISK_GB=$(df -g "${PROJECT_ROOT}" | tail -1 | awk '{print $4}')
    if [ "$FREE_DISK_GB" -ge 10 ]; then
        echo "OK (${FREE_DISK_GB}GB)"
    else
        echo "LOW (${FREE_DISK_GB}GB)"
        ((failures++))
    fi

    echo ""
    if [ $failures -gt 0 ]; then
        echo "Pre-flight FAILED with $failures error(s)"
        return 1
    else
        echo "Pre-flight checks PASSED"
        return 0
    fi
}

# ============================================================
# BACKGROUND HARDWARE MONITOR
# ============================================================

start_hardware_monitor() {
    echo "Starting hardware monitor (interval: ${MONITOR_INTERVAL}s)..."
    echo "Hardware log: ${HARDWARE_LOG}"

    mkdir -p "${LOG_DIR}"
    echo "timestamp,cpu_percent,memory_percent,memory_used_gb,temperature" > "${HARDWARE_LOG}"

    (
        while true; do
            STATS=$("${PYTHON}" -c "
import psutil
from src.training.thermal import get_macos_temperature
cpu = psutil.cpu_percent(interval=1)
mem = psutil.virtual_memory()
temp = get_macos_temperature()
print(f'{cpu},{mem.percent},{mem.used / (1024**3):.1f},{temp}')
" 2>/dev/null)

            if [ -n "$STATS" ]; then
                echo "$(date -Iseconds),${STATS}" >> "${HARDWARE_LOG}"
            fi

            sleep ${MONITOR_INTERVAL}
        done
    ) &
    MONITOR_PID=$!
    echo "Monitor PID: ${MONITOR_PID}"
}

stop_hardware_monitor() {
    if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "Stopping hardware monitor..."
        kill "$MONITOR_PID" 2>/dev/null
        wait "$MONITOR_PID" 2>/dev/null
    fi
}

cleanup() {
    stop_hardware_monitor
}
trap cleanup EXIT INT TERM

# ============================================================
# MAIN EXECUTION
# ============================================================

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Phase 6A HPO Runner - Large Models (200M, 2B)"
echo "============================================================"
echo "Started: $(date)"
echo "Experiments: ${#EXPERIMENTS[@]}"
echo "Mode: $([ "$DRY_RUN" = true ] && echo 'DRY-RUN' || echo 'LIVE')"
echo "Log file: ${LOG_FILE}"
echo ""
echo "NOTE: 2M and 20M experiments run separately for parallelization"
echo "      Use run_phase6a_2M.sh and run_phase6a_20M.sh"
echo "============================================================"
echo ""

if ! preflight_check; then
    echo "Aborting due to pre-flight failures."
    exit 1
fi
echo ""

# Dry-run mode
if [ "$DRY_RUN" = true ]; then
    echo "============================================================"
    echo "DRY-RUN: Execution Plan"
    echo "============================================================"
    for i in "${!EXPERIMENTS[@]}"; do
        exp="${EXPERIMENTS[$i]}"
        exp_name=$(get_experiment_name "$exp")
        exp_num=$((i + 1))

        if is_complete "$exp_name"; then
            echo "[${exp_num}/${#EXPERIMENTS[@]}] SKIP (complete): ${exp}"
        else
            echo "[${exp_num}/${#EXPERIMENTS[@]}] WOULD RUN: ${exp}"
        fi
    done
    echo ""
    echo "To execute, run without --dry-run flag."
    exit 0
fi

# Start logging
exec > >(tee -a "${LOG_FILE}") 2>&1

# Start hardware monitor
start_hardware_monitor
echo ""

echo "============================================================"
echo "Starting Large Model Experiment Queue"
echo "============================================================"
echo ""

TOTAL_START=$(date +%s)
COMPLETED=0
FAILED=0
SKIPPED_COUNT=0

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    exp_name=$(get_experiment_name "$exp")
    exp_num=$((i + 1))

    echo ""
    echo "============================================================"
    echo "[${exp_num}/${#EXPERIMENTS[@]}] ${exp}"
    echo "Time: $(date)"
    echo "============================================================"

    if is_complete "$exp_name"; then
        echo "SKIPPED: Already complete (50 trials)"
        SKIPPED[$i]=true
        RESULTS[$i]="SKIP"
        DURATIONS[$i]=0
        ((SKIPPED_COUNT++))
        continue
    fi

    SKIPPED[$i]=false
    START=$(date +%s)

    echo "Running: ${PYTHON} ${EXPERIMENTS_DIR}/${exp}"
    echo ""

    if "${PYTHON}" "${EXPERIMENTS_DIR}/${exp}"; then
        RESULTS[$i]="PASS"
        ((COMPLETED++))
        echo ""
        echo "PASSED: ${exp}"
    else
        RESULTS[$i]="FAIL"
        ((FAILED++))
        echo ""
        echo "FAILED: ${exp}"
    fi

    END=$(date +%s)
    DURATION=$((END - START))
    DURATIONS[$i]=$DURATION

    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    echo "Duration: $(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)"
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================================"
echo "SUMMARY - Large Models (200M, 2B)"
echo "============================================================"
echo "Completed: $(date)"
echo ""

TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))
echo "Total duration: $(printf "%02d:%02d:%02d" $TOTAL_HOURS $TOTAL_MINUTES $TOTAL_SECONDS)"
echo ""

echo "Results:"
for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    result="${RESULTS[$i]}"
    duration="${DURATIONS[$i]}"

    HOURS=$((duration / 3600))
    MINUTES=$(((duration % 3600) / 60))
    SECONDS=$((duration % 60))
    DUR_FMT=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

    case "$result" in
        PASS) echo "  [PASS] ${exp} (${DUR_FMT})" ;;
        FAIL) echo "  [FAIL] ${exp} (${DUR_FMT})" ;;
        SKIP) echo "  [SKIP] ${exp} (already complete)" ;;
    esac
done

echo ""
echo "Passed: ${COMPLETED}/${#EXPERIMENTS[@]}"
echo "Failed: ${FAILED}/${#EXPERIMENTS[@]}"
echo "Skipped: ${SKIPPED_COUNT}/${#EXPERIMENTS[@]}"
echo ""
echo "Log: ${LOG_FILE}"
echo "Hardware: ${HARDWARE_LOG}"
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
