#!/bin/bash
# Run all extended experiment categories (114 experiments)
#
# Categories:
#   DA: Data Augmentation (24 experiments) - DA-P1 to DA-P5
#   NR: Noise-Robust Training (18 experiments) - NR-P1 to NR-P4
#   CL: Curriculum Learning (18 experiments) - CL-P1 to CL-P4
#   RD: Regime Detection (18 experiments) - RD-P1 to RD-P4
#   MS: Multi-Scale Temporal (18 experiments) - MS-P1 to MS-P4
#   CP: Contrastive Pre-training (18 experiments) - CP-P1 to CP-P4
#
# Usage:
#   ./scripts/run_extended_experiments.sh           # Run all
#   ./scripts/run_extended_experiments.sh --phase 1 # Run Phase 1 only (DA, NR)
#   ./scripts/run_extended_experiments.sh --phase 2 # Run Phase 2 only (CL, RD)
#   ./scripts/run_extended_experiments.sh --phase 3 # Run Phase 3 only (MS)
#   ./scripts/run_extended_experiments.sh --phase 4 # Run Phase 4 only (CP)
#   ./scripts/run_extended_experiments.sh --dry-run # Dry run all

set -e
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
RUNNER="experiments/feature_embedding/run_experiments.py"
DRY_RUN=""
PHASE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Extended Experiments: 114 total ==="
echo "Starting at $(date)"
echo ""

run_priority() {
    local priority=$1
    echo "--- Running priority: $priority ---"
    caffeinate $PYTHON $RUNNER --priority "$priority" $DRY_RUN
    echo ""
}

# Phase 1: Foundation (no dependencies)
run_phase_1() {
    echo "=== Phase 1: Foundation (DA + NR) ==="
    # Data Augmentation (24 experiments)
    for p in 1 2 3 4 5; do
        run_priority "DA-P${p}"
    done

    # Noise-Robust Training (18 experiments)
    for p in 1 2 3 4; do
        run_priority "NR-P${p}"
    done
}

# Phase 2: Training Modifications
run_phase_2() {
    echo "=== Phase 2: Training Modifications (CL + RD) ==="
    # Curriculum Learning (18 experiments)
    for p in 1 2 3 4; do
        run_priority "CL-P${p}"
    done

    # Regime Detection (18 experiments)
    for p in 1 2 3 4; do
        run_priority "RD-P${p}"
    done
}

# Phase 3: Architecture Extensions
run_phase_3() {
    echo "=== Phase 3: Architecture Extensions (MS) ==="
    # Multi-Scale Temporal (18 experiments)
    for p in 1 2 3 4; do
        run_priority "MS-P${p}"
    done
}

# Phase 4: Pre-training (depends on DA)
run_phase_4() {
    echo "=== Phase 4: Pre-training (CP) ==="
    # Contrastive Pre-training (18 experiments)
    for p in 1 2 3 4; do
        run_priority "CP-P${p}"
    done
}

# Run based on phase argument
case $PHASE in
    1)
        run_phase_1
        ;;
    2)
        run_phase_2
        ;;
    3)
        run_phase_3
        ;;
    4)
        run_phase_4
        ;;
    "")
        # Run all phases
        run_phase_1
        run_phase_2
        run_phase_3
        run_phase_4
        ;;
    *)
        echo "Unknown phase: $PHASE (valid: 1, 2, 3, 4)"
        exit 1
        ;;
esac

echo "=== All extended experiments complete at $(date) ==="
