#!/bin/bash
#
# Run 10 supplementary training experiments for 2M cross-horizon analysis
#
# Purpose: Test h3-optimal config (d=64, L=32, h=2, dropout=0.10) on h1/h5,
# then vary n_heads and dropout to measure sensitivity.
#
# Usage:
#   ./scripts/run_supplementary_2M.sh
#
# Expected runtime: ~20-50 minutes total (2-5 min per script)
#

set -e
set -o pipefail  # Exit code from pipeline = rightmost failed command (not tee)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python"
SCRIPTS_DIR="${PROJECT_ROOT}/experiments/phase6a_supplementary"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/supplementary_2M_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

# Scripts to run (in order)
SCRIPTS=(
    # Set 1: Cross-horizon validation
    "train_h1_d64_L32_h2_drop010.py"
    "train_h5_d64_L32_h2_drop010.py"
    # Set 2: n_heads sensitivity
    "train_h1_d64_L32_h8_drop010.py"
    "train_h1_d64_L32_h16_drop010.py"
    "train_h5_d64_L32_h8_drop010.py"
    "train_h5_d64_L32_h16_drop010.py"
    # Set 3: dropout sensitivity
    "train_h1_d64_L32_h2_drop020.py"
    "train_h1_d64_L32_h2_drop030.py"
    "train_h5_d64_L32_h2_drop020.py"
    "train_h5_d64_L32_h2_drop030.py"
)

echo "============================================================" | tee "${LOG_FILE}"
echo "Supplementary 2M Training Runs" | tee -a "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Scripts: ${#SCRIPTS[@]}" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

TOTAL_START=$(date +%s)
PASSED=0
FAILED=0

for i in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$i]}"
    exp_num=$((i + 1))

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "[${exp_num}/${#SCRIPTS[@]}] ${script}" | tee -a "${LOG_FILE}"
    echo "Time: $(date)" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    START=$(date +%s)

    if "${PYTHON}" "${SCRIPTS_DIR}/${script}" 2>&1 | tee -a "${LOG_FILE}"; then
        END=$(date +%s)
        DURATION=$((END - START))
        echo "[${exp_num}/${#SCRIPTS[@]}] PASSED (${DURATION}s)" | tee -a "${LOG_FILE}"
        ((PASSED++))
    else
        END=$(date +%s)
        DURATION=$((END - START))
        echo "[${exp_num}/${#SCRIPTS[@]}] FAILED (${DURATION}s)" | tee -a "${LOG_FILE}"
        ((FAILED++))
    fi
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_MIN=$((TOTAL_DURATION / 60))

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "SUMMARY" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Completed: $(date)" | tee -a "${LOG_FILE}"
echo "Total time: ${TOTAL_MIN} min (${TOTAL_DURATION}s)" | tee -a "${LOG_FILE}"
echo "Passed: ${PASSED}/${#SCRIPTS[@]}" | tee -a "${LOG_FILE}"
echo "Failed: ${FAILED}/${#SCRIPTS[@]}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Results in: outputs/supplementary/" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
