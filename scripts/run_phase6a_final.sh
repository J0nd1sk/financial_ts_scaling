#!/bin/bash
#
# Phase 6A Final Training Runner Script
# Runs all 16 final training experiments sequentially with logging
#
# Usage:
#   tmux new -s final
#   ./scripts/run_phase6a_final.sh
#   ./scripts/run_phase6a_final.sh --start-from 5   # Start from experiment 5 (20M_h1)
#   # Ctrl+B, D to detach
#   # tmux attach -t final to reattach
#
# Graceful Stop:
#   Option 1: touch outputs/logs/STOP_FINAL
#   Option 2: Ctrl+C (SIGINT) - stops after current experiment
#
#   The runner checks for stop signals between experiments.
#   When triggered, it stops gracefully after the current experiment.
#

set -o pipefail

# ============================================================
# ARGUMENT PARSING
# ============================================================

START_FROM=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--start-from N]"
            echo ""
            echo "Phase 6A Final Training Runner"
            echo "Runs 16 final training experiments (4 budgets × 4 horizons)"
            echo ""
            echo "Options:"
            echo "  --start-from N   Start from experiment N (1-16)"
            echo "                   1-4: 2M, 5-8: 20M, 9-12: 200M, 13-16: 2B"
            echo ""
            echo "Experiments:"
            echo "   1: train_2M_h1      9: train_200M_h1"
            echo "   2: train_2M_h2     10: train_200M_h2"
            echo "   3: train_2M_h3     11: train_200M_h3"
            echo "   4: train_2M_h5     12: train_200M_h5"
            echo "   5: train_20M_h1    13: train_2B_h1"
            echo "   6: train_20M_h2    14: train_2B_h2"
            echo "   7: train_20M_h3    15: train_2B_h3"
            echo "   8: train_20M_h5    16: train_2B_h5"
            echo ""
            echo "Graceful stop: touch outputs/logs/STOP_FINAL or Ctrl+C"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Validate START_FROM
if [[ ! "$START_FROM" =~ ^[0-9]+$ ]] || [ "$START_FROM" -lt 1 ] || [ "$START_FROM" -gt 16 ]; then
    echo "Error: --start-from must be a number between 1 and 16"
    exit 1
fi

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python"
EXPERIMENTS_DIR="${PROJECT_ROOT}/experiments/phase6a_final"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/phase6a_final_${TIMESTAMP}.log"
HARDWARE_LOG="${LOG_DIR}/hardware_final_${TIMESTAMP}.csv"
MONITOR_INTERVAL=300  # 5 minutes
STOP_FILE="${LOG_DIR}/STOP_FINAL"  # Touch this file to stop after current experiment

# Background monitor PID (for cleanup)
MONITOR_PID=""

# Flag for graceful stop on signal (must use file for subshell visibility)
STOP_SIGNAL_FILE="${LOG_DIR}/.stop_signal_final_$$"

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================

preflight_check() {
    echo "Running pre-flight checks..."
    local failures=0

    # Check 1: MPS available
    echo -n "  MPS (Apple Silicon GPU): "
    if "${PYTHON}" -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        echo "✓ available"
    else
        echo "✗ NOT AVAILABLE"
        ((failures++))
    fi

    # Check 2: Temperature readable (sudo cached)
    echo -n "  Temperature (sudo cached): "
    if sudo -n true 2>/dev/null; then
        TEMP=$("${PYTHON}" -c "from src.training.thermal import get_macos_temperature; print(get_macos_temperature())" 2>/dev/null)
        if [ "$TEMP" != "-1.0" ] && [ -n "$TEMP" ]; then
            echo "✓ readable (${TEMP}°C)"
        else
            echo "⚠ not readable (will continue without thermal monitoring)"
        fi
    else
        echo "⚠ sudo not cached (run 'sudo -v' first for thermal monitoring)"
    fi

    # Check 3: Sufficient memory (>8GB free)
    echo -n "  Memory (>8GB free): "
    FREE_MEM_GB=$("${PYTHON}" -c "import psutil; print(f'{psutil.virtual_memory().available / (1024**3):.1f}')" 2>/dev/null)
    if [ -n "$FREE_MEM_GB" ]; then
        FREE_MEM_INT=${FREE_MEM_GB%.*}
        if [ "$FREE_MEM_INT" -ge 8 ]; then
            echo "✓ ${FREE_MEM_GB}GB available"
        else
            echo "✗ only ${FREE_MEM_GB}GB available (need 8GB+)"
            ((failures++))
        fi
    else
        echo "⚠ could not check memory"
    fi

    # Check 4: Data file exists (a20 tier for final training)
    echo -n "  Data file: "
    if [ -f "${PROJECT_ROOT}/data/processed/v1/SPY_dataset_a20.parquet" ]; then
        echo "✓ exists"
    else
        echo "✗ NOT FOUND (expected: data/processed/v1/SPY_dataset_a20.parquet)"
        ((failures++))
    fi

    # Check 5: All 16 experiment scripts exist
    echo -n "  Experiment scripts: "
    local missing=0
    for exp in "${EXPERIMENTS[@]}"; do
        if [ ! -f "${EXPERIMENTS_DIR}/${exp}" ]; then
            ((missing++))
        fi
    done
    if [ $missing -eq 0 ]; then
        echo "✓ all 16 found"
    else
        echo "✗ ${missing} scripts missing"
        ((failures++))
    fi

    echo ""
    if [ $failures -gt 0 ]; then
        echo "❌ Pre-flight failed with $failures error(s)"
        return 1
    else
        echo "✅ Pre-flight checks passed"
        return 0
    fi
}

# ============================================================
# BACKGROUND HARDWARE MONITOR
# ============================================================

start_hardware_monitor() {
    echo "Starting background hardware monitor (interval: ${MONITOR_INTERVAL}s)..."
    echo "Hardware log: ${HARDWARE_LOG}"

    # Create log file with header
    mkdir -p "${LOG_DIR}"
    echo "timestamp,cpu_percent,memory_percent,temperature" > "${HARDWARE_LOG}"

    # Start background monitoring loop
    (
        while true; do
            STATS=$("${PYTHON}" -c "
import psutil
from src.training.thermal import get_macos_temperature
cpu = psutil.cpu_percent(interval=1)
mem = psutil.virtual_memory().percent
temp = get_macos_temperature()
print(f'{cpu},{mem},{temp}')
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
        echo "Stopping hardware monitor (PID: ${MONITOR_PID})..."
        kill "$MONITOR_PID" 2>/dev/null
        wait "$MONITOR_PID" 2>/dev/null
    fi
}

# Handle interrupt signals gracefully
handle_signal() {
    echo ""
    echo "============================================================"
    echo "⚠️  Signal received - will stop after current experiment"
    echo "============================================================"
    touch "${STOP_SIGNAL_FILE}"
}

# Cleanup on exit
cleanup() {
    stop_hardware_monitor
    # Remove stop files (clean state for next run)
    rm -f "${STOP_FILE}" "${STOP_SIGNAL_FILE}"
}

# Set up signal handlers
trap handle_signal INT TERM
trap cleanup EXIT

# ============================================================
# EXPERIMENTS (16 total: 4 budgets × 4 horizons)
# ============================================================

# Order: smallest to largest (2M → 20M → 200M → 2B)
# Within each budget: h1, h2, h3, h5
EXPERIMENTS=(
    "train_2M_h1.py"
    "train_2M_h2.py"
    "train_2M_h3.py"
    "train_2M_h5.py"
    "train_20M_h1.py"
    "train_20M_h2.py"
    "train_20M_h3.py"
    "train_20M_h5.py"
    "train_200M_h1.py"
    "train_200M_h2.py"
    "train_200M_h3.py"
    "train_200M_h5.py"
    "train_2B_h1.py"
    "train_2B_h2.py"
    "train_2B_h3.py"
    "train_2B_h5.py"
)

# Results tracking
declare -a RESULTS
declare -a DURATIONS

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Run pre-flight checks
echo ""
if ! preflight_check; then
    echo "Aborting due to pre-flight failures."
    exit 1
fi
echo ""

# Start background hardware monitoring
start_hardware_monitor
echo ""

# Show graceful stop hint
echo "💡 To stop gracefully after current experiment: touch ${STOP_FILE}"
echo ""

# Header
echo "============================================================" | tee "${LOG_FILE}"
echo "Phase 6A Final Training Runner" | tee -a "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Hardware log: ${HARDWARE_LOG}" | tee -a "${LOG_FILE}"
echo "Experiments: ${#EXPERIMENTS[@]}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run each experiment
TOTAL_START=$(date +%s)

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    exp_num=$((i + 1))

    # Skip experiments before START_FROM
    if [ $exp_num -lt $START_FROM ]; then
        echo "[${exp_num}/${#EXPERIMENTS[@]}] Skipping: ${exp} (--start-from ${START_FROM})"
        RESULTS[$i]="SKIP"
        DURATIONS[$i]=0
        continue
    fi

    # Check for graceful stop request (file OR signal)
    if [ -f "${STOP_FILE}" ] || [ -f "${STOP_SIGNAL_FILE}" ]; then
        echo "" | tee -a "${LOG_FILE}"
        echo "============================================================" | tee -a "${LOG_FILE}"
        echo "⏹️  STOP REQUESTED" | tee -a "${LOG_FILE}"
        echo "Completed $((exp_num-1))/${#EXPERIMENTS[@]} experiments." | tee -a "${LOG_FILE}"
        echo "Resume with: $0 --start-from ${exp_num}" | tee -a "${LOG_FILE}"
        echo "============================================================" | tee -a "${LOG_FILE}"
        rm -f "${STOP_FILE}" "${STOP_SIGNAL_FILE}"
        break
    fi

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
        echo "[${exp_num}/${#EXPERIMENTS[@]}] ✅ PASSED: ${exp}" | tee -a "${LOG_FILE}"
    else
        RESULTS[$i]="FAIL"
        echo "" | tee -a "${LOG_FILE}"
        echo "[${exp_num}/${#EXPERIMENTS[@]}] ❌ FAILED: ${exp}" | tee -a "${LOG_FILE}"
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
SKIPPED=0

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    result="${RESULTS[$i]}"
    duration="${DURATIONS[$i]:-0}"

    HOURS=$((duration / 3600))
    MINUTES=$(((duration % 3600) / 60))
    SECONDS=$((duration % 60))
    DUR_FMT=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

    if [ "$result" = "PASS" ]; then
        echo "  [PASS] ${exp} (${DUR_FMT})" | tee -a "${LOG_FILE}"
        ((PASSED++))
    elif [ "$result" = "SKIP" ]; then
        echo "  [SKIP] ${exp}" | tee -a "${LOG_FILE}"
        ((SKIPPED++))
    elif [ -n "$result" ]; then
        echo "  [FAIL] ${exp} (${DUR_FMT})" | tee -a "${LOG_FILE}"
        ((FAILED++))
    fi
done

echo "" | tee -a "${LOG_FILE}"
echo "Passed: ${PASSED}, Failed: ${FAILED}, Skipped: ${SKIPPED}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Hardware log: ${HARDWARE_LOG}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

# Exit with failure if any experiment failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
