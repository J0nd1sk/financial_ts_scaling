# HPO Logging Infrastructure Plan

**Status**: APPROVED, waiting for implementation
**Created**: 2025-12-28
**Memory Entity**: `HPO_Logging_Infrastructure_Plan`
**Blocked By**: Smoke test running - cannot regenerate scripts until complete

## Problem

HPO scripts use `print()` statements only. No automatic file logging.
Runner script uses external `tee` - but direct script execution has no logs.

## Solution

Add Python `logging` module to generated HPO scripts:
- Dual output: console + file
- Log file: `outputs/hpo/{experiment}/{experiment}.log`
- Timestamps on all entries

## Design

Add to generated scripts:

```python
import logging

def setup_logging():
    """Configure logging to both console and file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / f"{EXPERIMENT}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logging()
```

Replace `print(f"...")` with `logger.info("...")`.

## Files to Modify

| File | Changes |
|------|---------|
| `src/experiments/templates.py` | Add logging setup (~30 lines) |
| `tests/experiments/test_templates.py` | Add 3 tests (~15 lines) |
| `experiments/phase6a/hpo_*.py` (12) | Regenerate after template change |

## Tests to Add

1. `test_hpo_script_imports_logging`: Assert `import logging` present
2. `test_hpo_script_creates_log_file`: Assert log file setup code exists
3. `test_hpo_script_uses_logger`: Assert `logger.info()` usage

## Implementation Steps

1. Write failing tests (TDD RED)
2. Modify `templates.py` to add logging
3. Run `make test` (TDD GREEN)
4. Regenerate all 12 HPO scripts
5. Verify a script creates log file when run

## Prerequisites

- Task 6 smoke test must complete first
- N_TRIALS reverted to 50
- HPO Time Optimization stage marked complete

---

*This plan will be implemented after current smoke test completes.*
