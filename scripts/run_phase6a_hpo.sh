#!/bin/bash
#
# Phase 6A HPO Runner Script
# Runs all 12 HPO experiments sequentially with logging
#
# Usage:
#   tmux new -s hpo
#   ./scripts/run_phase6a_hpo.sh
#   # Ctrl+B, D to detach
#   # tmux attach -t hpo to reattach
#

set -o pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python"
EXPERIMENTS_DIR="${PROJECT_ROOT}/experiments/phase6a"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/phase6a_hpo_${TIMESTAMP}.log"

# Experiments in order (smallest to largest)
EXPERIMENTS=(
    "hpo_2M_h1_threshold_1pct.py"
    "hpo_2M_h3_threshold_1pct.py"
    "hpo_2M_h5_threshold_1pct.py"
    "hpo_20M_h1_threshold_1pct.py"
    "hpo_20M_h3_threshold_1pct.py"
    "hpo_20M_h5_threshold_1pct.py"
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

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Header
echo "============================================================" | tee "${LOG_FILE}"
echo "Phase 6A HPO Runner" | tee -a "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Experiments: ${#EXPERIMENTS[@]}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run each experiment
TOTAL_START=$(date +%s)

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    exp_num=$((i + 1))

    echo "" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "[${exp_num}/${#EXPERIMENTS[@]}] Starting: ${exp}" | tee -a "${LOG_FILE}"
    echo "Time: $(date)" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    START=$(date +%s)

    # Run experiment, capture output to log
    if "${PYTHON}" "${EXPERIMENTS_DIR}/${exp}" 2>&1 | tee -a "${LOG_FILE}"; then
        RESULTS[$i]="PASS"
        echo "" | tee -a "${LOG_FILE}"
        echo "[${exp_num}/${#EXPERIMENTS[@]}] PASSED: ${exp}" | tee -a "${LOG_FILE}"
    else
        RESULTS[$i]="FAIL"
        echo "" | tee -a "${LOG_FILE}"
        echo "[${exp_num}/${#EXPERIMENTS[@]}] FAILED: ${exp}" | tee -a "${LOG_FILE}"
    fi

    END=$(date +%s)
    DURATION=$((END - START))
    DURATIONS[$i]=$DURATION

    # Format duration as HH:MM:SS
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    DURATION_FMT=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

    echo "Duration: ${DURATION_FMT}" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))
TOTAL_FMT=$(printf "%02d:%02d:%02d" $TOTAL_HOURS $TOTAL_MINUTES $TOTAL_SECONDS)

# Summary
echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "SUMMARY" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Completed: $(date)" | tee -a "${LOG_FILE}"
echo "Total duration: ${TOTAL_FMT}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

PASSED=0
FAILED=0

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    result="${RESULTS[$i]}"
    duration="${DURATIONS[$i]}"

    HOURS=$((duration / 3600))
    MINUTES=$(((duration % 3600) / 60))
    SECONDS=$((duration % 60))
    DUR_FMT=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

    if [ "$result" = "PASS" ]; then
        echo "  [PASS] ${exp} (${DUR_FMT})" | tee -a "${LOG_FILE}"
        ((PASSED++))
    else
        echo "  [FAIL] ${exp} (${DUR_FMT})" | tee -a "${LOG_FILE}"
        ((FAILED++))
    fi
done

echo "" | tee -a "${LOG_FILE}"
echo "Passed: ${PASSED}/${#EXPERIMENTS[@]}" | tee -a "${LOG_FILE}"
echo "Failed: ${FAILED}/${#EXPERIMENTS[@]}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

# Exit with failure if any experiment failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
