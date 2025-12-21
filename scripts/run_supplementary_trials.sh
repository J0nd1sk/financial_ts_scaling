#!/bin/bash
# Run Phase 6A Supplementary Trials
#
# 9 experiments testing:
# - Cross-horizon validation of best architectures
# - New architectures (d=1024 L=16, d=768 L=28)
#
# Estimated time: ~15-20 min per trial, ~3 hours total

set -e

SCRIPT_DIR="experiments/phase6a_supplementary"
LOG_DIR="outputs/supplementary/logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Phase 6A Supplementary Trials"
echo "========================================"
echo "Start time: $(date)"
echo ""

# List of experiments in recommended order
EXPERIMENTS=(
    # Cross-horizon validation first (known good architectures)
    "train_768_L24_h16_horizon1.py"    # h3 winner on h1
    "train_768_L24_h16_horizon5.py"    # h3 winner on h5
    "train_1024_L12_h16_horizon5.py"   # runner-up on h5

    # New architecture: d=1024, L=16 (201.8M - perfect budget)
    "train_1024_L16_h16_horizon1.py"
    "train_1024_L16_h16_horizon3.py"
    "train_1024_L16_h16_horizon5.py"

    # New architecture: d=768, L=28 (198.7M - near-perfect budget)
    "train_768_L28_h16_horizon1.py"
    "train_768_L28_h16_horizon3.py"
    "train_768_L28_h16_horizon5.py"
)

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

for exp in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    NAME="${exp%.py}"
    LOG_FILE="$LOG_DIR/${NAME}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "[$CURRENT/$TOTAL] Running: $exp"
    echo "Log: $LOG_FILE"
    echo "----------------------------------------"

    # Check thermal before starting
    TEMP=$(sudo powermetrics --samplers smc -n 1 2>/dev/null | grep -i "CPU die" | awk '{print $4}' | tr -d 'C' || echo "unknown")
    echo "CPU Temperature: ${TEMP}°C"

    if [[ "$TEMP" != "unknown" ]] && (( $(echo "$TEMP > 90" | bc -l) )); then
        echo "⚠️  Temperature too high, waiting 60s..."
        sleep 60
    fi

    # Run experiment
    START=$(date +%s)
    ./venv/bin/python "$SCRIPT_DIR/$exp" 2>&1 | tee "$LOG_FILE"
    END=$(date +%s)
    DURATION=$((END - START))

    echo "Duration: ${DURATION}s ($((DURATION / 60))m)"
    echo "----------------------------------------"
done

echo ""
echo "========================================"
echo "All experiments complete!"
echo "End time: $(date)"
echo "========================================"
echo ""
echo "Results saved to: outputs/supplementary/"
echo "Logs saved to: $LOG_DIR/"
