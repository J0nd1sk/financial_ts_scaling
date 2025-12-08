# Decision Log

## 2025-11-26 PatchTST-Only Scaling Architecture

**Context**: Need a single architecture to isolate scaling-law effects across parameter budgets.

**Decision**: Use Hugging Face PatchTST as the exclusive model for all experiments; scale only via parameter count and feature/data dimensions.

**Rationale**: PatchTST has proven TS performance, clean parameterization, and avoids architectural confounds.

**Alternatives Considered**:
- Lag-Llama or other sequence models — rejected for higher complexity and lack of mature tooling.
- Multi-architecture comparison — rejected until baseline scaling behavior is established.

**Implications**: All training/evaluation code must target PatchTST; any architectural change requires a new approval gate.

## 2025-11-26 Fixed Dataset Matrix & Splits

**Context**: Scaling experiments require consistent data ranges and dataset definitions.

**Decision**: Lock train/val/test splits to ≤2020 / 2021-2022 / 2023+, and adopt the 5×4 asset-quality dataset matrix (Aa..Ed) with defined feature tiers.

**Rationale**: Prevents leakage, keeps comparisons fair, and enables reproducible scaling curves.

**Alternatives Considered**:
- Rolling or adaptive splits — rejected for added variance.
- Ad-hoc dataset definitions — rejected due to poor reproducibility.

**Implications**: All data work must respect the matrix and date ranges; deviations require documented approval.

## 2025-12-07 Context File Enforcement

**Context**: Rules review showed `.claude/context` artifacts existed but lacked explicit usage requirements.

**Decision**: Added mandatory read/write conditions for `session_context.md`, `phase_tracker.md`, and `decision_log.md` across both Claude and Cursor rule sets.

**Rationale**: Ensure every session start, end, phase change, and architectural decision is captured, keeping restore/handoff workflows reliable.

**Alternatives Considered**:
- Rely on skills alone — rejected because Cursor lacks Superpowers automation.
- Update docs only — rejected since enforcement belongs in active rules.

**Implications**: Agents must now update context files whenever the defined triggers occur; omissions are treated as process violations.

## 2025-12-08 Data Version Manifest System

**Context**: Need reproducible tracking of raw/processed datasets before Phase 2 pipeline work.

**Decision**: Introduced JSON manifests (`data/raw/manifest.json`, `data/processed/manifest.json`) plus `scripts/manage_data_versions.py` and `make verify` integration to register artifacts with MD5 checksums.

**Rationale**: Guarantees provenance for every dataset, enables automated verification, and ensures session handoffs capture latest data state.

**Alternatives Considered**:
- Git LFS or DVC — rejected for added tooling overhead at this stage.
- Manual spreadsheets — rejected due to high risk of drift and poor automation.

**Implications**: All future data downloads/processing steps must register manifest entries; handoff/restore summaries must note latest entries; verification failures block merges.
