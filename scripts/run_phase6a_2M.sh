#!/bin/bash
#
# Phase 6A HPO Runner Script - 2M Budget Only
# Runs 2M HPO experiments (can run in parallel with larger models)
#
# Experiments:
#   1-3: 2M_h{1,3,5}
#
# Usage:
#   tmux new-window -t hpo -n 2M
#   ./scripts/run_phase6a_2M.sh
#   # Or in a new tmux session:
#   tmux new -s hpo-2M
#   ./scripts/run_phase6a_2M.sh
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
LOG_FILE="${LOG_DIR}/phase6a_2M_${TIMESTAMP}.log"

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

# 2M experiments only
EXPERIMENTS=(
    "hpo_2M_h1_threshold_1pct.py"
    "hpo_2M_h3_threshold_1pct.py"
    "hpo_2M_h5_threshold_1pct.py"
)

# Results tracking
declare -a RESULTS
declare -a DURATIONS

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
# PRE-FLIGHT CHECKS (lightweight for 2M)
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

    echo -n "  Memory (>4GB free for 2M): "
    FREE_MEM_GB=$("${PYTHON}" -c "import psutil; print(f'{psutil.virtual_memory().available / (1024**3):.1f}')" 2>/dev/null)
    if [ -n "$FREE_MEM_GB" ]; then
        FREE_MEM_INT=${FREE_MEM_GB%.*}
        if [ "$FREE_MEM_INT" -ge 4 ]; then
            echo "OK (${FREE_MEM_GB}GB)"
        else
            echo "LOW (${FREE_MEM_GB}GB)"
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
# MAIN EXECUTION
# ============================================================

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Phase 6A HPO Runner - 2M Budget Only"
echo "============================================================"
echo "Started: $(date)"
echo "Experiments: ${#EXPERIMENTS[@]}"
echo "Mode: $([ "$DRY_RUN" = true ] && echo 'DRY-RUN' || echo 'LIVE')"
echo "Log file: ${LOG_FILE}"
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

echo "============================================================"
echo "Starting 2M Experiment Queue"
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
        RESULTS[$i]="SKIP"
        DURATIONS[$i]=0
        ((SKIPPED_COUNT++))
        continue
    fi

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
echo "SUMMARY - 2M Budget"
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
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
