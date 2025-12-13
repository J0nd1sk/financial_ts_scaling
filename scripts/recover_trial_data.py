#!/usr/bin/env python3
"""
Recover trial data from HPO log file.

Parses the phase6a_hpo_*.log file to extract trial information
and generates structured JSON files for experiments that didn't
save their data properly (due to TrialLogger bug or old script format).
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Regex patterns for parsing log lines
TRIAL_START_PATTERN = re.compile(
    r'\[I (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\] '
    r'Trial (\d+): arch_idx=(\d+), d_model=(\d+), n_layers=(\d+), params=([\d,]+), '
    r'lr=([\d.e-]+), epochs=(\d+), batch_size=(\d+)'
)

TRIAL_FINISH_PATTERN = re.compile(
    r'\[I (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\] '
    r'Trial (\d+) finished with value: ([\d.]+) and parameters: \{([^}]+)\}'
)

EXPERIMENT_START_PATTERN = re.compile(
    r'\[(\d+)/12\] Starting: hpo_(\w+)_threshold_1pct\.py'
)

EXPERIMENT_END_PATTERN = re.compile(
    r'\[(\d+)/12\] (PASSED|FAILED): hpo_(\w+)_threshold_1pct\.py'
)


def parse_params(params_str: str) -> dict:
    """Parse the parameters string from log."""
    params = {}
    # Match 'key': value patterns
    for match in re.finditer(r"'(\w+)': ([\d.e-]+)", params_str):
        key = match.group(1)
        value = match.group(2)
        # Convert to appropriate type
        if '.' in value or 'e' in value:
            params[key] = float(value)
        else:
            params[key] = int(value)
    return params


def parse_log_file(log_path: Path) -> dict:
    """Parse the log file and extract trial data for each experiment."""
    experiments = defaultdict(lambda: {
        'trials': [],
        'status': None,
        'start_time': None,
        'end_time': None,
    })

    current_experiment = None
    pending_trials = {}  # trial_num -> trial_start_data

    with open(log_path, 'r') as f:
        for line in f:
            # Check for experiment start
            exp_start = EXPERIMENT_START_PATTERN.search(line)
            if exp_start:
                current_experiment = exp_start.group(2)
                pending_trials = {}
                continue

            # Check for experiment end
            exp_end = EXPERIMENT_END_PATTERN.search(line)
            if exp_end:
                exp_name = exp_end.group(3)
                experiments[exp_name]['status'] = exp_end.group(2)
                continue

            if not current_experiment:
                continue

            # Check for trial start
            trial_start = TRIAL_START_PATTERN.search(line)
            if trial_start:
                timestamp = trial_start.group(1)
                trial_num = int(trial_start.group(2))
                pending_trials[trial_num] = {
                    'start_time': timestamp,
                    'arch_idx': int(trial_start.group(3)),
                    'd_model': int(trial_start.group(4)),
                    'n_layers': int(trial_start.group(5)),
                    'param_count': int(trial_start.group(6).replace(',', '')),
                    'lr': float(trial_start.group(7)),
                    'epochs': int(trial_start.group(8)),
                    'batch_size': int(trial_start.group(9)),
                }
                continue

            # Check for trial finish
            trial_finish = TRIAL_FINISH_PATTERN.search(line)
            if trial_finish:
                timestamp = trial_finish.group(1)
                trial_num = int(trial_finish.group(2))
                val_loss = float(trial_finish.group(3))
                params = parse_params(trial_finish.group(4))

                # Merge with start data if available
                trial_data = pending_trials.get(trial_num, {})
                trial_data.update({
                    'trial_number': trial_num,
                    'end_time': timestamp,
                    'val_loss': val_loss,
                    'params': params,
                })

                # Extract architecture from params if not in start data
                if 'arch_idx' not in trial_data and 'arch_idx' in params:
                    trial_data['arch_idx'] = params['arch_idx']

                experiments[current_experiment]['trials'].append(trial_data)
                continue

    return dict(experiments)


def find_best_trial(trials: list) -> dict:
    """Find the trial with the lowest val_loss."""
    if not trials:
        return None
    return min(trials, key=lambda t: t.get('val_loss', float('inf')))


def generate_output_files(experiments: dict, output_base: Path):
    """Generate JSON files for each experiment."""
    for exp_name, exp_data in experiments.items():
        trials = exp_data['trials']
        if not trials:
            print(f"  {exp_name}: No trials found, skipping")
            continue

        # Determine output directory
        # Map experiment names to directory names
        if exp_name.startswith('2M_'):
            budget = '2M'
            horizon = exp_name.split('_')[1]  # h1, h3, h5
        elif exp_name.startswith('20M_'):
            budget = '20M'
            horizon = exp_name.split('_')[1]
        elif exp_name.startswith('200M_'):
            budget = '200M'
            horizon = exp_name.split('_')[1]
        elif exp_name.startswith('2B_'):
            budget = '2B'
            horizon = exp_name.split('_')[1]
        else:
            print(f"  {exp_name}: Unknown budget format, skipping")
            continue

        dir_name = f"phase6a_{budget}_{horizon}_threshold_1pct"
        output_dir = output_base / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort trials by val_loss
        sorted_trials = sorted(trials, key=lambda t: t.get('val_loss', float('inf')))

        # Generate all_trials.json
        all_trials_path = output_dir / f"phase6a_{budget}_{horizon}_threshold_1pct_all_trials.json"
        all_trials_data = {
            'experiment': f"phase6a_{budget}_{horizon}_threshold_1pct",
            'budget': budget,
            'n_trials': len(trials),
            'trials': sorted_trials,
            'generated_from': 'log_recovery',
            'timestamp': datetime.now().isoformat(),
        }
        with open(all_trials_path, 'w') as f:
            json.dump(all_trials_data, f, indent=2)

        # Find best trial
        best = find_best_trial(trials)

        print(f"  {exp_name}: {len(trials)} trials recovered, best val_loss={best['val_loss']:.4f}")
        print(f"    -> {all_trials_path}")


def main():
    # Find the log file
    log_dir = Path("outputs/logs")
    log_files = list(log_dir.glob("phase6a_hpo_*.log"))

    if not log_files:
        print("ERROR: No phase6a_hpo_*.log files found in outputs/logs/")
        sys.exit(1)

    # Use the most recent log file
    log_path = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"Parsing log file: {log_path}")

    # Parse the log
    experiments = parse_log_file(log_path)

    print(f"\nFound {len(experiments)} experiments:")
    for exp_name, exp_data in sorted(experiments.items()):
        status = exp_data['status'] or 'IN_PROGRESS'
        print(f"  {exp_name}: {len(exp_data['trials'])} trials ({status})")

    # Generate output files
    print("\nGenerating output files:")
    output_base = Path("outputs/hpo")
    generate_output_files(experiments, output_base)

    print("\nDone!")


if __name__ == '__main__':
    main()
