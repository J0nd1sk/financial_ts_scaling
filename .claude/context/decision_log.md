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

## 2025-12-08 Dataset Naming Convention

**Context**: Registering first SPY download in manifest required choosing a dataset identifier format.

**Decision**: Adopt hierarchical naming: `{TICKER}.{DATA_TYPE}.{FREQUENCY}` (e.g., `SPY.OHLCV.daily`).

**Rationale**: Enables future expansion to multiple data types (fundamentals, options, sentiment) and frequencies (1min, hourly, weekly) while maintaining clear provenance.

**Alternatives Considered**:
- Simple ticker name (`SPY`) — rejected as ambiguous when multiple data types exist.
- Filename-based (`SPY.parquet`) — rejected as it conflates storage with logical identity.

**Implications**: All manifest registrations must follow this convention; scripts should validate format on registration.

## 2025-12-08 Testing & Manifest Automation Lessons

**Context**: Phase 2 implementation had three critical issues identified by code review:
1. Tests called real yfinance API (slow, flaky, non-reproducible)
2. Each test downloaded full SPY history independently (8× redundant downloads)
3. Manifest registration was manual, not automatic in download script

**Decision**: Establish mandatory patterns:
- **Data download tests must use mocks/fixtures**: Never call real APIs in tests; mock external dependencies for speed and reproducibility
- **Shared test fixtures**: Use pytest fixtures to download/mock data once, reuse across tests
- **Automatic manifest registration**: Download scripts must register artifacts programmatically, not rely on manual post-download steps

**Rationale**: Tests must be fast, deterministic, and offline-capable per PRD reproducibility requirements. Manual registration steps violate automation principles and invite manifest drift.

**Alternatives Considered**:
- Live API tests in separate suite — adds complexity, still flaky
- Optional manifest registration flag — easy to forget, defeats automation purpose

**Implications**:
- All future data-fetching code must include mocking strategy in test plan
- Download scripts must integrate manifest registration before merge
- CI/CD must run offline without external API dependencies
