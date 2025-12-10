# Financial TS Transformer Scaling: Phase Plans

**Status:** Active Development Plan  
**Version:** 2.0  
**Last Updated:** 2025-11-26  
**Previous:** phase_plans_v1.md (superseded)

**Key Changes in v2:**
- Added Section 0: Development Discipline & Agentic Workflow
- Integrated SpecKit (constitution + selective commands)
- Integrated Superpowers skills system
- All code samples marked as propositions requiring review
- RACI matrix for human-LLM collaboration
- Test-driven development requirements
- Git branching strategy
- Context management architecture

---

# Section 0: Development Discipline & Agentic Workflow

## 0.1 Human-LLM RACI Matrix

| Activity | Human (Alex) | LLM/Agent |
|----------|--------------|-----------|
| **Planning & Design** |
| Experimental design | **Lead**, Accountable | Consult, Inform |
| Architecture decisions | **Lead**, Accountable | Propose, Consult |
| Phase planning | **Lead**, Approve | **Execute** (with SpecKit) |
| Task breakdown | Review, Approve | **Propose** |
| Branching strategy | Approve | **Propose** |
| **Development** |
| Writing tests | Review, Approve | **Execute** |
| Writing implementation | Review, Approve | **Execute** |
| Refactoring | Approve | **Propose** |
| Code review | **Execute**, Accountable | Inform |
| Git operations (commit/merge) | Approve | **Execute** |
| **Execution** |
| Data downloads | Monitor | **Execute** |
| Feature calculation | Review results | **Execute** |
| Model training | Monitor thermal, review | **Execute** |
| Hyperparameter optimization | Review strategy | **Execute** |
| **Quality & Validation** |
| Test execution | Review results | **Execute** |
| Probability scoring validation | **Review**, Accountable | **Execute**, Inform |
| Experimental results analysis | **Execute**, Accountable | Inform |
| Publication preparation | **Execute**, Accountable | Consult |

**Key Principles:**
- **Lead** = Makes decisions and drives work
- **Execute** = Does the actual work  
- **Review** = Evaluates quality and correctness
- **Approve** = Final sign-off required before proceeding
- **Accountable** = Ultimately responsible for outcomes
- **Consult** = Provides input and expertise
- **Inform** = Keeps others updated on progress

## 0.2 Development Workflow Standards

### 0.2.1 Planning Protocol (REQUIRED Before ANY Code)

**For ALL tasks (even "simple" ones):**

1. **Planning Session** (5-30 minutes depending on complexity)
   - Define objective clearly
   - Identify success criteria
   - Surface assumptions explicitly
   - Estimate complexity and scope

2. **Task Breakdown**
   - Decompose into reviewable sub-tasks
   - Assign RACI for each sub-task
   - Identify dependencies
   - Estimate time per sub-task

3. **Technical Design** (for complex tasks)
   - Propose approach
   - Identify alternatives considered
   - Justify technology choices
   - Highlight risks

4. **Branching Strategy Proposal**
   - Branch name following convention
   - Base branch (usually `staging`)
   - Merge target and criteria
   - Testing requirements

5. **Test Plan**
   - List test cases to be written
   - Define validation criteria
   - Identify edge cases

6. **Approval Gate**
   - Present summary to Alex
   - WAIT for explicit "yes" / "go ahead" / "approved"
   - Do NOT proceed on assumption or silence

### 0.2.2 Test-Driven Development (MANDATORY)

**Protocol:**
1. Write failing test FIRST
2. Run test, confirm it fails with expected error
3. Write MINIMAL code to make test pass
4. Run test, confirm it passes
5. Refactor if needed (keeping tests green)
6. Commit with test + implementation together

**Test Naming Convention:**
```python
def test_[function]_[scenario]_[expected_result]():
    """Test that [function] [expected_result] when [scenario]."""
```

**Examples:**
```python
def test_download_ohlcv_invalid_ticker_raises_ValueError():
    """Test that download_ohlcv raises ValueError when ticker is invalid."""

def test_calculate_rsi_returns_array_with_correct_shape():
    """Test that calculate_rsi returns array matching input length."""

def test_patchtst_forward_pass_output_shape_matches_prediction_length():
    """Test that PatchTST forward pass output has correct prediction length."""
```

**Coverage Requirements:**
- Data pipeline: >80%
- Feature calculation: >80%
- Model architecture: >60%
- Training loops: >50%
- Utility functions: >90%

**Testing Framework:**
```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

### 0.2.3 Code Review Process

**All code requires review before merge:**

1. **LLM generates review checklist:**
   - Changes summary
   - Files modified
   - Tests added/modified
   - Potential breaking changes
   - Performance considerations
   - Thermal impact (for training code)

2. **Alex reviews:**
   - Functional correctness
   - Alignment with experimental design
   - Code quality and maintainability
   - Test coverage adequacy
   - Documentation completeness

3. **Approval types:**
   - âœ… **Approved** - Merge immediately
   - ðŸ”„ **Approved with changes** - Implement feedback then merge
   - âŒ **Rejected** - Do not merge, rework required

### 0.2.4 Git Branching Strategy

**Branch Types:**
```
main                    # Production: published results only, protected
staging                 # Integration: all tests pass, ready for main
feature/phase-N-*       # Phase-specific work
experiment/[name]       # Individual experiment runs  
hotfix/*               # Critical fixes to main
```

**Workflow:**
```
1. Create feature branch from staging
   git checkout staging
   git pull origin staging
   git checkout -b feature/phase-2-ide-setup

2. Work in feature branch
   - Make changes
   - Write tests
   - Commit atomically
   - Push regularly

3. Create experiment branches from feature
   git checkout -b experiment/2M-daily-direction

4. PR to staging with review
   - All tests pass
   - Code review approved
   - Merge to staging

5. Staging â†’ main only after phase completion
   - Full phase validated
   - All experiments documented
   - Results reproducible
```

**Experiment Branch Naming:**
```
experiment/[param_budget]-[timescale]-[task]

Examples:
experiment/2M-daily-direction
experiment/20M-weekly-price-regression
experiment/200M-monthly-threshold-3pct
```

**Commit Message Format:**
```
[type]: Brief description (50 chars max)

Detailed explanation if needed (wrap at 72 chars):
- What changed
- Why it changed
- Impact on experiments

Related: #issue-number, phase-N-task-M
```

**Commit Types:**
- `feat:` New feature or capability
- `fix:` Bug fix
- `refactor:` Code restructuring without behavior change
- `test:` Adding or modifying tests
- `docs:` Documentation updates
- `data:` Data acquisition or processing
- `exp:` Experiment run or configuration
- `perf:` Performance improvements

## 0.3 SpecKit Integration

### 0.3.1 Installation & Setup

```bash
# Install SpecKit
git clone https://github.com/github/spec-kit
cd spec-kit
# Follow installation instructions

# Create constitution
mkdir -p .speckit
touch .speckit/constitution.md
```

### 0.3.2 Constitution File

**Location:** `.speckit/constitution.md`

```markdown
# Financial TS Transformer Scaling - Constitution

## Experimental Integrity (Non-Negotiable)

### Parameter Constraints
- Parameter budgets: 2M, 20M, 200M ONLY
- No intermediate values permitted
- Clean isolation: PatchTST architecture exclusively
- One model per task: 48 models per dataset matrix
- Batch size re-tuning: REQUIRED when parameter budget changes
- Batch size re-tuning: RECOMMENDED when dataset changes
- Batch size re-tuning: OPTIONAL when feature count changes

### Probability Scoring Requirements
- Binary classification: Sigmoid activation for probability output
- Regression tasks: Conformal prediction for uncertainty quantification
- All predictions must include confidence/uncertainty estimates
- Threshold classifications: 1%, 2%, 3%, 5% movement thresholds

### Reproducibility Standards
- Fixed random seeds for all experiments
- Deterministic operations (cudnn.deterministic = True)
- Version-pinned dependencies
- Environment snapshots saved with results

## Thermal Constraints (M4 MacBook Pro 128GB)

### Temperature Thresholds
- **Normal**: <70Â°C - Full operation permitted
- **Acceptable**: 70-85Â°C - Continue with monitoring
- **Warning**: 85-95Â°C - Consider pause, reduce batch size
- **Critical**: >95Â°C - IMMEDIATE STOP, terminate training

### Thermal Management
- Basement cooling required: 50-60Â°F ambient
- Monitor every epoch during training
- Log temperature with experiment results
- Automatic shutdown hook at 95Â°C
- No overnight unattended training without thermal monitoring

## Data Quality Requirements

### Storage Standards
- Raw data: Parquet format only
- Processed data: Parquet with version suffix (v1, v2, etc.)
- LLM debugging: CSV samples (1000 rows max) in `data/samples/`
- Checksums: MD5 for all raw downloads
- Backups: External SSD mirror of `data/raw/`

### Validation Requirements
- Schema validation before processing
- Null value checks and handling policy
- Outlier detection and treatment
- Date range verification
- Volume checks (expected row counts)

### Dataset Quality Matrix
```
Rows (Assets):
A: SPY only
B: +DIA, QQQ
C: +individual stocks (AAPL, MSFT, GOOGL, AMZN, TSLA)
D: +sentiment (SF Fed News Sentiment Index)
E: +economic indicators (FRED)

Columns (Quality Tiers):
a: OHLCV + basic indicators (SMA, EMA, RSI, MACD)
b: +sentiment indicators
c: +VIX correlation features
d: +trend analysis features

Example: Dataset "C-c" = SPY+DIA+QQQ+stocks, with OHLCV+indicators+sentiment+VIX
```

## Development Standards

### Test-Driven Development
- No implementation without failing test first
- Test naming: `test_[function]_[scenario]_[expected]`
- Coverage minimums: data (80%), features (80%), models (60%)
- Integration tests for full pipeline
- Smoke tests for trained models

### Code Quality
- Black formatting (line length: 88)
- isort for import sorting
- mypy type checking (strict mode)
- Absolute imports only: `from src.module import function`
- Never relative imports: `from .module` or `from ..module`
- Docstrings: NumPy style for all public functions
- Max function length: 50 lines (exceptions require comment)
- Max file length: 500 lines (refactor if exceeded)

### Code Propositions Policy
- All code samples in plans are PROPOSITIONS
- No copy-paste without planning review
- Planning session required before implementation
- Code review required before merge
- Tests required before approval

## Prohibited Actions (Require Explicit Approval)

### Experimental Parameters
- Changing parameter budgets mid-experiment
- Modifying model architecture during comparison
- Altering train/val/test split dates
- Changing loss functions without documentation

### Development
- Committing directly to main branch
- Merging without passing tests
- Deleting data files
- Training without thermal monitoring
- Pushing large files (models/data) to git

### Infrastructure
- Modifying Python version
- Changing core dependencies without version lock
- Altering directory structure
- Removing backup files

## Approval Gates (Must Pause for Human Review)

### Automatic Triggers
- Any refactoring >50 lines
- Architecture changes
- Dependency additions/upgrades
- Branch merges to staging or main
- Parameter budget changes
- Phase transitions
- Model training with new configuration
- Data deletion or overwriting

### Approval Protocol
1. Generate summary: What, Why, Impact, Risks, Alternatives
2. Present to Alex with explicit request
3. WAIT for approval: "yes", "no", or "modify"
4. Proceed only on explicit "yes" or "go ahead"
5. Never assume silence = approval
```

### 0.3.3 Selective Command Usage

**When to Use SpecKit Commands:**

| Situation | Command | Output | Use Case |
|-----------|---------|--------|----------|
| Complex phase with unclear requirements | `/specify` | Structured specification | Phase 3: Pipeline Design |
| Major architectural decisions | `/plan` | Technical design doc | Feature calculation engine |
| Many parallel or sequential tasks | `/tasks` | Numbered task breakdown | Phase 5: Data Acquisition |
| Simple/repetitive work | Skip SpecKit | Use Superpowers only | Bug fixes, small refactors |

**Decision Tree:**
```
Starting new work?
â”‚
â”œâ”€ Is it simple/repetitive?
â”‚  â””â”€ Use Superpowers skills only
â”‚
â”œâ”€ Is it complex with unclear requirements?
â”‚  â””â”€ Use SpecKit: /specify â†’ review/edit â†’ save to .speckit/specs/
â”‚
â”œâ”€ Need architectural design?
â”‚  â””â”€ Use SpecKit: /plan â†’ review/edit â†’ save to .speckit/specs/
â”‚
â””â”€ Need task breakdown?
   â””â”€ Use SpecKit: /tasks â†’ review/edit â†’ save to .speckit/specs/
```

**Example: Phase 5 (Data Acquisition)**

```bash
# Step 1: Specification
/specify "Download and validate OHLCV data for SPY, DIA, QQQ from 1990-2025 with retry logic and rate limiting"

# Review generated spec, edit as needed, save to:
# .speckit/specs/phase-5-data-acquisition.md

# Step 2: Technical planning
/plan "Implement resilient download system with exponential backoff retry"

# Review technical design, edit, append to spec file

# Step 3: Task breakdown  
/tasks "Execute Phase 5: Data Acquisition"

# Review task list, edit, append to spec file

# Step 4: Execute using Superpowers workflow (NOT SpecKit /implement)
# See Section 0.4 for Superpowers integration
```

**What We Skip:**
- âŒ `/implement` - Use our own execution workflow
- âŒ Full SpecKit review gates - Use our RACI process
- âŒ SpecKit PR integration - Use our branching strategy
- âŒ Forced adherence to all commands - Cherry-pick what helps

## 0.4 Superpowers Skills System

### 0.4.1 Installation

```bash
# In Claude Code session
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace

# Restart Claude Code
# Verify installation
# Should see: "You have Superpowers" in session start hook
```

### 0.4.2 Skills Directory Structure

- REMOVED INTENTIONALLY BECAUSE OF ERRORS

### 0.4.3 Core Skills (Implement First)

#### Priority 1: Essential Skills (Implement in Phase 2)

**1. session_handoff.md** - Context management
```markdown
# Skill: Session Handoff

## When to Use
- Context window >80% full
- User says "end session", "done for today", or similar
- Before switching phases or major tasks
- Natural session end after 2-4 hours work

## Protocol
1. Check context usage: must be >80% to warrant handoff
2. Generate HANDOFF.md with exactly these sections:
   
   ```markdown
   # Session Handoff - [YYYY-MM-DD HH:MM]
   
   ## Phase Status
   **Current:** Phase N - [Name]
   **Progress:** [X]% complete
   
   ## Last Session Summary
   **Objective:** [One sentence]
   **Completed:** [3-5 bullets, specific accomplishments]
   **Decisions Made:** [Key choices, max 3-5]
   **Files Modified:** [Paths only, no diffs]
   **Next Session Starts With:** [1-3 immediate actions]
   
   ## Active Branches
   - main: [status]
   - staging: [status]  
   - feature/[name]: current work
   
   ## Thermal Status
   Last recorded: [X]Â°C ([normal/acceptable/warning])
   
   ## Open Questions
   [2-3 questions maximum, or "None"]
   
   ## Token Budget This Session
   Context used: [X]K tokens
   Available next session: ~[Y]K tokens
   ```

3. Save to root: `HANDOFF.md`
4. Confirm to user: "Session state saved to HANDOFF.md"

## Token Budget
Keep entire handoff <500 tokens

## Never Include
- Code snippets or implementations
- Full file contents  
- Detailed explanations
- Speculation or uncertainty
- Repeated project rules (those are in CLAUDE.md)

## Validation
Before saving, verify:
- Decisions are specific, not vague
- Next steps are actionable
- Token count is <500
```

**2. test_first.md** - TDD enforcement
```markdown
# Skill: Test-Driven Development

## Protocol (MANDATORY)
1. **NEVER write implementation code first**
2. **ALWAYS follow RED â†’ GREEN â†’ REFACTOR**

## Process for ANY New Functionality
Step 1 (RED): Write failing test
- Define expected behavior in test
- Run test: `pytest tests/[module]/test_[function].py -v`
- Confirm: Test FAILS with expected error message
- If test passes unexpectedly, test is wrong

Step 2 (GREEN): Write minimal code
- Write ONLY enough code to make test pass
- No "future-proofing" or "while we're here" additions
- Run test: `pytest tests/[module]/test_[function].py -v`
- Confirm: Test PASSES

Step 3 (REFACTOR): Clean up if needed
- Improve code quality (readability, performance)
- Keep tests passing throughout
- Run test after each change
- Confirm: Tests still PASS

## Test Naming Convention
```python
def test_[function]_[scenario]_[expected_result]():
    """Test that [function] [expected_result] when [scenario]."""
```

## Before ANY Implementation
Ask user: "What test validates this works correctly?"
Write that test first.
Do NOT proceed to implementation until test is written and failing.

## For ML Models
Tests validate:
- Input shape handling: `assert output.shape == expected_shape`
- Output probability range: `assert torch.all((output >= 0) & (output <= 1))`
- Gradient flow: `assert not torch.isnan(loss).any()`
- Reproducibility: `assert torch.allclose(output1, output2)`

## Enforcement
If implementation code appears before test code:
1. STOP immediately
2. Ask: "What test validates this implementation?"
3. Write test first
4. Then implement
```

**3. approval_gate.md** - Prevent runaway changes
```markdown
# Skill: Approval Gate

## Trigger Before (Automatic)
- Any refactoring >50 lines
- Architecture changes
- Dependency additions/upgrades
- Branch merges (to staging or main)
- Parameter budget changes
- Phase transitions
- Model training with new configuration
- File/data deletion

## Protocol
1. **PAUSE execution immediately**
2. Generate approval summary:

```markdown
# Approval Request

## What
[One sentence: what change is proposed]

## Why  
[1-2 sentences: what problem this solves]

## Impact
- Files affected: [list paths]
- Scope: [small/medium/large]
- Reversible: [yes/no/with-effort]

## Risks
[1-3 specific risks, or "None identified"]

## Alternatives Considered
[1-2 alternatives, or "None - straightforward decision"]

## Recommendation
[Proceed / Reconsider / Needs discussion]
```

3. Present to user with explicit request:
   "âš ï¸  APPROVAL REQUIRED âš ï¸
    
    [Summary above]
    
    **Proceed?** Reply with:
    - 'yes' or 'approved' â†’ I'll proceed
    - 'no' or 'stop' â†’ I'll abort
    - 'modify [details]' â†’ I'll adjust approach"

4. **WAIT for explicit approval**
   - Do NOT proceed on assumption
   - Do NOT proceed on silence
   - Do NOT proceed on "probably okay"

5. Parse response:
   - "yes", "go ahead", "approved", "proceed" â†’ Execute
   - "no", "stop", "don't", "wait" â†’ Abort
   - Anything else â†’ Ask for clarification

## Never Proceed Without Approval For
- Git operations (merge, rebase, push to staging/main)
- Deleting files or data
- Changing experimental parameters
- Model training with new budget
- Breaking changes to interfaces
- Refactoring >100 lines

## If Approval Denied
1. Acknowledge: "Understood, aborting [action]"
2. Ask: "Would you like me to propose an alternative approach?"
3. Wait for direction
```

**4. thermal_management.md** - M4 monitoring
```markdown
# Skill: Thermal Management (M4 MacBook Pro)

## When to Use
- Before starting model training
- Every epoch during training
- After long computations
- When system feels hot
- Before overnight processes

## Temperature Check Protocol
```python
# Read CPU temperature
import subprocess
result = subprocess.run(
    ['sudo', 'powermetrics', '-n', '1', '-i', '1000', '--samplers', 'smc'],
    capture_output=True,
    text=True
)
# Parse temperature from output
# Format: "CPU Thermal level: X"
```

## Temperature Thresholds
- **<70Â°C**: âœ… Normal - Full operation
- **70-85Â°C**: âš ï¸  Acceptable - Monitor closely
- **85-95Â°C**: âš ï¸  Warning - Consider:
  - Reducing batch size
  - Pausing for cooldown
  - Checking ambient temperature
- **>95Â°C**: ðŸ›‘ CRITICAL - STOP IMMEDIATELY
  - Terminate training
  - Save checkpoint
  - Alert user
  - Do NOT resume until <70Â°C

## Before Training Protocol
1. Check temperature
2. If >70Â°C, wait for cooldown
3. Log starting temperature
4. Set up thermal monitoring hook
5. Proceed only if <70Â°C

## During Training Protocol
1. Check temperature every epoch
2. Log to: `outputs/thermal_log.txt`
3. Format: `[timestamp] Epoch [N]: [X]Â°C`
4. If >85Â°C: Display warning
5. If >95Â°C: STOP and save

## Thermal Hook Code
```python
class ThermalCallback:
    def on_epoch_end(self, epoch, logs=None):
        temp = get_cpu_temperature()
        log_temperature(epoch, temp)
        
        if temp > 95:
            print(f"ðŸ›‘ CRITICAL: {temp}Â°C - STOPPING")
            self.model.stop_training = True
            save_checkpoint(self.model, f"emergency_{epoch}")
        elif temp > 85:
            print(f"âš ï¸  WARNING: {temp}Â°C - Monitor closely")
```

## Ambient Temperature
- Basement cooling required: 50-60Â°F
- Check ambient before long training
- If >65Â°F ambient, reduce batch size

## Logging
Log every temperature check to:
`outputs/thermal_log.txt`

Include in experiment results:
- Starting temperature
- Average training temperature  
- Peak temperature
- Any thermal events (warnings/stops)
```

#### Priority 2: Development Workflow (Implement in Phase 3)

**5. planning_session.md** - Structured pre-coding
```markdown
# Skill: Planning Session

## When to Use
Before ANY code is written, even for "simple" tasks

## Protocol
1. Read context:
   - CLAUDE.md for project rules
   - .speckit/specs/[feature].md if exists
   - HANDOFF.md for session state

2. Interactive planning questions:
   ```
   Let's plan this task:
   
   1. Objective: [One sentence - what are we building?]
   
   2. Success criteria: [How do we know it's done?]
   
   3. Assumptions: [What are we assuming about data/environment/dependencies?]
   
   4. Scope:
      - In scope: [What WILL we do]
      - Out of scope: [What we WON'T do]
   
   5. Approach: [High-level technical approach in 2-3 sentences]
   
   6. Risks: [What could go wrong?]
   
   7. Alternatives: [Other approaches we considered?]
   ```

3. Break down into sub-tasks (use task_breakdown.md skill)

4. Propose branching strategy (use branch_strategy.md skill)

5. Define test plan:
   - What tests will we write?
   - What are we validating?
   - Edge cases to cover?

6. Generate approval request (use approval_gate.md skill)

7. WAIT for approval before proceeding

## Complexity Estimation
- Simple: <1 hour, single file, <100 lines
- Medium: 1-4 hours, 2-5 files, <500 lines
- Complex: >4 hours, >5 files, >500 lines, or architectural changes

## For Complex Tasks
Also use SpecKit:
- `/specify` for requirements
- `/plan` for technical design
- `/tasks` for detailed breakdown

## Output
Save planning summary to:
`.claude/context/planning_[feature].md`

Include:
- All answers above
- Task breakdown
- Branch strategy
- Test plan
- Approval status
```

**6. task_breakdown.md** - RACI decomposition
```markdown
# Skill: Task Breakdown

## When to Use
After planning session, before implementation

## Protocol
1. Take high-level objective
2. Decompose into sub-tasks
3. Assign RACI for each
4. Identify dependencies
5. Estimate time
6. Number sequentially

## Sub-Task Format
```markdown
### Task [N]: [Brief description]
**Assignee:** [Human/LLM]
**RACI:** [R/A/C/I for Human and LLM]
**Depends on:** [Task numbers, or "None"]
**Estimated time:** [X hours/minutes]
**Deliverable:** [Specific output]

**Steps:**
1. [Specific action]
2. [Specific action]
3. [Specific action]

**Tests required:**
- [Test case 1]
- [Test case 2]

**Review criteria:**
- [Criterion 1]
- [Criterion 2]
```

## Example
```markdown
### Task 1: Setup test directory
**Assignee:** LLM
**RACI:** Human (Approve), LLM (Responsible, Execute)
**Depends on:** None
**Estimated time:** 10 minutes
**Deliverable:** tests/unit/data/ directory with __init__.py

**Steps:**
1. Create directory: tests/unit/data/
2. Create __init__.py with module docstring
3. Verify structure

**Tests required:**
- None (infrastructure only)

**Review criteria:**
- Directory exists
- __init__.py has docstring
```

### Task 2: Write test for download_ohlcv
**Assignee:** LLM
**RACI:** Human (Review, Approve), LLM (Responsible, Execute)
**Depends on:** Task 1
**Estimated time:** 30 minutes
**Deliverable:** tests/unit/data/test_download.py

**Steps:**
1. Write test_download_ohlcv_valid_ticker_returns_dataframe
2. Write test_download_ohlcv_invalid_ticker_raises_ValueError
3. Run tests (should fail - no implementation yet)

**Tests required:**
- Test executes and fails appropriately
- Coverage check shows missing implementation

**Review criteria:**
- Test names follow convention
- Tests have clear docstrings
- Assertions are specific
```

## Parallel Tasks
Mark with [P] if can run concurrently:
```markdown
### Task 3 [P]: Download SPY data
### Task 4 [P]: Download DIA data  
### Task 5 [P]: Download QQQ data
### Task 6: Validate all downloads (depends on 3,4,5)
```

## Output
Save to: `.claude/context/tasks_[feature].md`
```

### 0.4.4 Skills Implementation Timeline

**Phase 2: IDE Setup & Rules** (Implement 4 skills)
- âœ… session_handoff.md
- âœ… test_first.md
- âœ… approval_gate.md
- âœ… thermal_management.md

**Phase 3: Pipeline Design** (Implement 3 skills)
- âœ… planning_session.md
- âœ… task_breakdown.md
- âœ… branch_strategy.md

**Phase 4: Boilerplate** (Implement 2 skills)
- âœ… code_review_prep.md
- âœ… holistic_review.md

**Phase 5+: As Needed** (Implement remaining skills iteratively)
- session_restore.md
- context_budget.md
- experimental_discipline.md
- dataset_validation.md
- probability_scoring.md
- etc.

## 0.5 Context Management Architecture

### 0.5.1 Document Separation

**Static Documents (Version Controlled):**
- `CLAUDE.md` - Project rules and constitution (~1.5K tokens)
- `.speckit/constitution.md` - SpecKit configuration
- `.claude/skills/*.md` - Superpowers skills
- `.claude/context/phase_tracker.md` - Experimental progress

**Dynamic Documents (Not Committed):**
- `HANDOFF.md` - Session state, regenerated each session
- `.claude/context/decision_log.md` - Append-only log

**Add to `.gitignore`:**
```gitignore
# Session state (dynamic)
HANDOFF.md

# Temporary context
.claude/context/planning_*.md
.claude/context/tasks_*.md
```

### 0.5.2 CLAUDE.md Structure

**Location:** `CLAUDE.md` (root directory)

```markdown
# Financial Time-Series Transformer Scaling Experiments

## Project Objective
Experimentally test whether neural scaling laws apply to transformer models 
trained on financial time-series data with rigorous methodology for publication.

## Constitution
See: `.speckit/constitution.md` for complete rules

## Development Workflow
See: Section 0.2 in phase_plans_v2.md

## Tech Stack
- Python 3.12, PyTorch MPS, Optuna HPO
- W&B + MLflow (local), pandas-ta + TA-Lib  
- Parquet storage, CSV samples for LLM debugging

## Current Phase
See: `HANDOFF.md` for session state
See: `.claude/context/phase_tracker.md` for progress

## Skills
Superpowers installed. See `.claude/skills/` directory.

## Code Propositions
All code samples in phase_plans_v2.md are PROPOSITIONS requiring:
1. Planning session
2. Task breakdown with RACI
3. Branching strategy
4. Test-first development
5. Code review and approval
```

### 0.5.3 Session Handoff Protocol

**At Session Start:**
1. LLM reads `HANDOFF.md` (if exists)
2. Confirms phase and task continuation with user
3. Loads relevant context from `.claude/context/`
4. Proceeds with work

**At Session End (>80% context):**
1. LLM triggers `session_handoff.md` skill
2. Generates `HANDOFF.md` with current state
3. Updates `.claude/context/phase_tracker.md`
4. Logs decisions to `.claude/context/decision_log.md`

**Token Budget:**
- CLAUDE.md: ~1.5K tokens (static)
- HANDOFF.md: <500 tokens (dynamic)
- Skills: ~2K tokens (auto-loaded by Superpowers)
- Available for work: ~196K tokens (200K window)

## 0.6 Coding Standards

### 0.6.1 Python Style
- **Formatter:** Black (line length: 88)
- **Import sorting:** isort
- **Type checking:** mypy (strict mode)
- **Linting:** flake8 + pylint

### 0.6.2 Import Rules
**ALWAYS:**
```python
from src.data.download import download_ohlcv
from src.features.indicators import calculate_rsi
```

**NEVER:**
```python
from .download import download_ohlcv  # Relative import
from ..features import calculate_rsi  # Relative import
```

### 0.6.3 Docstring Format (NumPy Style)
```python
def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index.

    Parameters
    ----------
    prices : np.ndarray
        Array of closing prices, shape (n_samples,)
    period : int, default=14
        RSI calculation period in bars

    Returns
    -------
    np.ndarray
        RSI values, shape (n_samples,)
        Values range from 0 to 100

    Raises
    ------
    ValueError
        If prices array is shorter than period
        
    Examples
    --------
    >>> prices = np.array([100, 102, 101, 103, 105])
    >>> rsi = calculate_rsi(prices, period=3)
    >>> len(rsi) == len(prices)
    True
    """
```

### 0.6.4 Function Length Limits
- **Target:** <30 lines per function
- **Maximum:** 50 lines (exception requires comment explaining why)
- **If exceeded:** Refactor into helper functions

### 0.6.5 File Length Limits  
- **Target:** <300 lines per file
- **Maximum:** 500 lines (exception requires comment)
- **If exceeded:** Split into multiple modules

### 0.6.6 Configuration
**File:** `pyproject.toml`
```toml
[tool.black]
line-length = 88
target-version = ['py312']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "--cov=src --cov-report=term-missing --cov-report=html"
```

## 0.7 Acceptance Criteria for Section 0

- [ ] SpecKit installed and `.speckit/constitution.md` created
- [ ] Superpowers installed with 4 core skills implemented
- [ ] CLAUDE.md created with project rules
- [ ] HANDOFF.md template tested
- [ ] `.gitignore` updated for session state
- [ ] Git branching strategy documented
- [ ] RACI matrix reviewed and approved
- [ ] TDD protocol tested with sample function
- [ ] Approval gate tested with sample refactoring
- [ ] Code formatting tools configured and tested

---

# Phase 1: Environment, Directory, Repository

## 1.1 Objectives

- Functional Python 3.12 environment with all dependencies
- GitHub repository with proper structure (including .claude/ and .speckit/)
- Verified MPS (Apple Silicon GPU) acceleration
- API keys obtained and tested
- Backup infrastructure ready
- SpecKit and Superpowers integrated

## 1.2 Tasks

### 1.2.1 Create GitHub Repository

> **NOTE:** This phase now executed AFTER Section 0 (Development Discipline) is complete.

| Item | Detail |
|------|--------|
| Repo name | `financial-ts-scaling` |
| Visibility | Private (until publication) |
| License | MIT (add at publication) |
| Initial commit | README.md + .gitignore + CLAUDE.md + .speckit/ |

**README.md initial content:**
```markdown
# Financial Time-Series Transformer Scaling Experiments

Testing whether neural scaling laws apply to transformer models trained on financial time-series data.

## Status
Environment setup phase - implementing development discipline (Section 0).

## Experimental Framework
- **Objective:** Validate/refute scaling law hypothesis with rigorous methodology
- **Architecture:** PatchTST for clean parameter scaling isolation
- **Budgets:** 2M, 20M, 200M parameters
- **Tasks:** 48 models (6 tasks Ã— 8 timescales) per dataset

## Development Workflow
- **Agentic:** SpecKit + Superpowers for AI-assisted development
- **TDD:** Test-driven development mandatory
- **RACI:** Clear human-LLM responsibility matrix
- **Thermal:** M4 MacBook Pro with basement cooling

## Structure
See `docs/phase_plans_v2.md` for full documentation.

## Installation
See `docs/setup.md` for environment setup.
```

### 1.2.2 Create Directory Structure

```bash
mkdir -p financial-ts-scaling/{.claude/{skills/{core,context,development,quality},context},.speckit/{specs},configs/{phase1,phase2,phase3,phase4,phase5,phase6},data/{raw/{ohlcv,fred,sentiment},processed/v1,samples},src/{data,features,models,training,evaluation,utils},tests/{unit,integration},notebooks,scripts,outputs/{checkpoints,results,figures,thermal},docs/runbooks}
```

**Resulting structure:**
- REMOVED BECAUSE OF ERRORS

### 1.2.3 Create .gitignore

```gitignore
# Models and checkpoints
*.pt
*.pth
*.onnx
*.safetensors

# Data files
*.parquet
*.csv
!data/samples/*.csv
data/raw/
data/processed/

# Experiment tracking
outputs/checkpoints/
outputs/results/
mlruns/
wandb/

# Environment
.venv/
__pycache__/
*.pyc
.env

# IDE
.cursor/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Secrets
*.key
credentials.json

# Session state (dynamic, not committed)
HANDOFF.md

# Temporary context files
.claude/context/planning_*.md
.claude/context/tasks_*.md
```

### 1.2.4 Create Python Environment

> **PROPOSITION ONLY**: This code is a starting point for planning discussion.
> Actual implementation requires:
> 1. Planning session to review approach
> 2. Test-first development (write test before code)
> 3. Code review and approval
> 4. Do not copy-paste without review

```bash
cd financial-ts-scaling
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 1.2.5 Create requirements.txt

> **PROPOSITION ONLY**: Review dependencies, versions may need updating.

```text
# Core ML
torch>=2.6.0
torchvision>=0.21.0
torchaudio>=2.6.0
transformers>=4.48.0
datasets>=3.0.0
accelerate>=1.2.0

# Data processing
pandas>=2.2.0
polars>=1.0.0
pyarrow>=18.0.0
fastparquet>=2024.11.0

# Data acquisition
yfinance>=0.2.50
pandas-datareader>=0.10.0
fredapi>=0.5.2

# Technical indicators
pandas-ta>=0.3.14b0
TA-Lib>=0.4.32

# HPO and tracking
optuna>=4.0.0
optuna-dashboard>=0.17.0
wandb>=0.18.0
mlflow>=2.19.0

# ML utilities
scikit-learn>=1.6.0
mapie>=0.9.0
catboost>=1.2.7

# Visualization
matplotlib>=3.9.0
plotly>=5.24.0
seaborn>=0.13.0

# Development
pytest>=8.3.0
pytest-cov>=6.0.0
black>=24.10.0
isort>=5.13.0
mypy>=1.13.0
flake8>=7.0.0
pylint>=3.0.0
ipython>=8.30.0
jupyter>=1.1.0
```

### 1.2.6 Install Dependencies

> **PROPOSITION ONLY**: Review installation steps.

```bash
# TA-Lib C library (required before pip install)
brew install ta-lib

# Python packages
pip install -r requirements.txt

# Verify installation
pip list | grep -E "torch|transformers|optuna"
```

### 1.2.7 Create Verification Script

> **PROPOSITION ONLY**: This script will need review for completeness.

**File:** `scripts/verify_environment.py`

```python
#!/usr/bin/env python3
"""Verify development environment is correctly configured.

This script validates:
- Python version (3.12.x)
- PyTorch with MPS support
- PatchTST model loading
- Optuna functionality
- Data libraries
- Technical indicator libraries
- Experiment tracking libraries
"""

import sys


def check_python_version():
    """Require Python 3.12.x"""
    major, minor = sys.version_info[:2]
    assert major == 3 and minor == 12, f"Need Python 3.12, got {major}.{minor}"
    print(f"âœ“ Python {major}.{minor}")


def check_pytorch_mps():
    """Verify MPS (Apple Silicon GPU) available."""
    import torch
    
    assert torch.backends.mps.is_available(), "MPS not available"
    assert torch.backends.mps.is_built(), "MPS not built"
    
    # Quick tensor test
    x = torch.randn(10, 10, device="mps")
    y = x @ x.T
    assert y.shape == (10, 10)
    print(f"âœ“ PyTorch {torch.__version__} with MPS")


def check_transformers():
    """Verify PatchTST loads."""
    from transformers import PatchTSTConfig, PatchTSTForPrediction
    
    config = PatchTSTConfig(
        num_input_channels=1,
        context_length=64,
        prediction_length=1,
        patch_length=16,
        d_model=32,
        num_hidden_layers=2,
        num_attention_heads=2,
    )
    model = PatchTSTForPrediction(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"âœ“ PatchTST loads ({param_count:,} params in test config)")


def check_optuna():
    """Verify Optuna works."""
    import optuna
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return x ** 2
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=3, show_progress_bar=False)
    print(f"âœ“ Optuna {optuna.__version__}")


def check_data_libs():
    """Verify data libraries."""
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    import yfinance as yf
    
    print(f"âœ“ pandas {pd.__version__}, polars {pl.__version__}, pyarrow {pa.__version__}")


def check_indicators():
    """Verify indicator libraries."""
    import pandas_ta as ta
    import talib
    
    print(f"âœ“ pandas-ta {ta.version}, TA-Lib available")


def check_tracking():
    """Verify tracking libraries import."""
    import wandb
    import mlflow
    
    print(f"âœ“ wandb {wandb.__version__}, mlflow {mlflow.__version__}")


def main():
    """Run all verification checks."""
    print("Environment Verification")
    print("=" * 40)
    
    checks = [
        check_python_version,
        check_pytorch_mps,
        check_transformers,
        check_optuna,
        check_data_libs,
        check_indicators,
        check_tracking,
    ]
    
    failed = []
    for check in checks:
        try:
            check()
        except Exception as e:
            failed.append((check.__name__, str(e)))
            print(f"âœ— {check.__name__}: {e}")
    
    print("=" * 40)
    if failed:
        print(f"FAILED: {len(failed)} checks")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

### 1.2.8 Obtain API Keys

| Service | URL | Storage |
|---------|-----|---------|
| FRED | https://fred.stlouisfed.org/docs/api/api_key.html | `.env` as `FRED_API_KEY` |
| Weights & Biases | https://wandb.ai/authorize | `wandb login` (stores in ~/.netrc) |

**Create `.env` file (do not commit):**
```bash
FRED_API_KEY=your_key_here
```

### 1.2.9 Setup Backup Infrastructure

**External SSD structure:**
```
/Volumes/Backup/financial-ts-scaling/
â”œâ”€â”€ raw_data/           # Mirror of data/raw/
â”œâ”€â”€ checkpoints/        # Training checkpoints
â””â”€â”€ checksums.md5       # Verification file
```

**Initial backup script:** `scripts/backup.sh`

> **PROPOSITION ONLY**: Review backup strategy.

```bash
#!/bin/bash
set -e

BACKUP_DIR="/Volumes/Backup/financial-ts-scaling"
PROJECT_DIR="$(dirname "$0")/.."

# Check backup drive mounted
if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup drive not mounted at $BACKUP_DIR"
    exit 1
fi

# Sync raw data
rsync -av --progress "$PROJECT_DIR/data/raw/" "$BACKUP_DIR/raw_data/"

# Sync checkpoints
rsync -av --progress "$PROJECT_DIR/outputs/checkpoints/" "$BACKUP_DIR/checkpoints/"

echo "Backup complete: $(date)"
```

### 1.2.10 Install Agentic Tools

**SpecKit:**
```bash
# Install SpecKit
git clone https://github.com/github/spec-kit
cd spec-kit
# Follow installation instructions

# Return to project
cd ../financial-ts-scaling

# Create constitution
mkdir -p .speckit
# Copy constitution content from Section 0.3.2
```

**Superpowers:**
```bash
# In Claude Code session
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace

# Restart Claude Code
# Verify: should see "You have Superpowers" message
```

### 1.2.11 Create Core Documents

**CLAUDE.md:** (See Section 0.5.2 for content)

**HANDOFF.md template:** (Generated by session_handoff.md skill)

**.claude/context/phase_tracker.md:**
```markdown
# Experimental Phase Tracker

## Phase 0: Development Discipline âœ… COMPLETE
- RACI matrix defined: 2025-11-26
- SpecKit installed: 2025-11-26
- Superpowers installed: 2025-11-26
- Core skills implemented: 2025-11-26

## Phase 1: Environment Setup â³ IN PROGRESS (0%)
- Repository: â¸ï¸ Pending
- Dependencies: â¸ï¸ Pending
- MPS verified: â¸ï¸ Pending
- Agentic tools: â¸ï¸ Pending

## Phase 2: IDE Rules â¸ï¸ NOT STARTED

## Phase 3: Pipeline Design â¸ï¸ NOT STARTED

## Phase 4: Boilerplate â¸ï¸ NOT STARTED

## Phase 5: Data Acquisition â¸ï¸ NOT STARTED

## Phase 6: Experiments â¸ï¸ NOT STARTED

## Decision Log
**2025-11-26:** PatchTST for parameter scaling isolation
**2025-11-26:** Sigmoid heads for binary classification
**2025-11-26:** SpecKit + Superpowers for agentic development
**2025-11-26:** TDD mandatory, no exceptions
**2025-11-26:** HANDOFF.md for session state, CLAUDE.md for rules
```

## 1.3 Execution Order

1. Create GitHub repository
2. Setup directory structure
3. Install SpecKit and Superpowers
4. Create .claude/skills/ (4 core skills from Section 0)
5. Create .speckit/constitution.md
6. Create CLAUDE.md and phase_tracker.md
7. Create Python environment
8. Install dependencies
9. Run verification script
10. Obtain API keys
11. Setup backup infrastructure
12. Test session handoff protocol

## 1.4 Acceptance Criteria

- [ ] Repository created with proper structure
- [ ] .claude/ and .speckit/ directories configured
- [ ] Python 3.12 environment created
- [ ] All dependencies installed (requirements.txt)
- [ ] MPS verification passes (`scripts/verify_environment.py`)
- [ ] FRED API key obtained and tested
- [ ] W&B logged in successfully
- [ ] External SSD backup tested
- [ ] SpecKit constitution.md created
- [ ] Superpowers 4 core skills implemented
- [ ] CLAUDE.md created
- [ ] HANDOFF.md protocol tested
- [ ] phase_tracker.md initialized
- [ ] All checks green on `make test` (once Phase 2 complete)

## 1.5 Estimated Time

6-8 hours (including agentic tool setup and skills implementation)

---

# Phase 2: Data Pipeline Foundation

## 2.1 Objectives

- Minimal SPY OHLCV download pipeline
- Parquet storage with data splits (train/val/test)
- Comprehensive data validation tests
- Foundation for multi-asset and indicator expansion

## 2.2 Planning Session Output

**Date:** 2025-12-07
**Status:** Approved for implementation

### Objective
Create a minimal, testable data pipeline for downloading and validating SPY OHLCV data, establishing the foundation for scaling experiments.

### In Scope
- Data directory structure (raw/, processed/, samples/)
- SPY OHLCV download script using yfinance
- Basic data validation (completeness, date ranges, missing values)
- Parquet storage format
- Fixed data splits (train: through 2020, val: 2021-2022, test: 2023+)

### Out of Scope
- Multi-asset data (DIA, QQQ, stocks, econ) - Phase 5
- Indicator calculations - separate task after download works
- Data versioning system - separate task
- Advanced validation (outliers, anomalies)
- Data visualization/EDA tools

### Success Criteria
- [ ] `make test` passes with data pipeline tests
- [ ] SPY data downloads successfully via yfinance
- [ ] Data stored in parquet format at `data/raw/SPY.parquet`
- [ ] Data contains expected columns: Date, Open, High, Low, Close, Volume
- [ ] Data covers date range from earliest available through present
- [ ] Data splits correctly into train/val/test by date boundaries
- [ ] No missing values in OHLCV data
- [ ] Script is idempotent (safe to re-run)

### Assumptions
1. yfinance 0.2.66 works for SPY historical data
2. Yahoo Finance has SPY data from ~1993 to present
3. Yahoo Finance OHLCV data is complete (no gaps in trading days)
4. 2020/2021/2023 boundaries are appropriate for train/val/test
5. Parquet format is best for this data
6. Internet connection available for data downloads
7. `data/`, `scripts/`, `tests/` directories exist

### Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| yfinance API changes/breaks | Medium | Error handling, version pin, document alternatives |
| Missing data (market holidays) | Low | Validation checks, document expected gaps |
| Date split boundaries incorrect | Medium | Make boundaries configurable, validate split sizes |
| Download too slow for testing | Medium | Add caching, use small date ranges for tests |
| Parquet write fails | Low | Error handling, validate write before return |
| Network failures during download | High | Retry logic, timeout handling, clear error messages |

## 2.3 Tasks

### 2.3.1 Create Data Directory Structure

> **PROPOSITION ONLY**: Requires approval before execution.

**Task:** Create subdirectories for data organization
**RACI:** Human (Approve), LLM (Execute)
**Estimated time:** 5 minutes

```bash
mkdir -p data/raw data/processed data/samples
touch data/raw/.gitkeep data/processed/.gitkeep data/samples/.gitkeep
```

**Deliverable:**
- `data/raw/` exists
- `data/processed/` exists
- `data/samples/` exists
- Each has `.gitkeep` file for git tracking

### 2.3.2 Write Tests for Data Download

> **PROPOSITION ONLY**: TDD - Write tests BEFORE implementation.

**File:** `tests/test_data_download.py`

**Test Cases:**

1. **`test_download_spy_basic()`**
   - Downloads SPY data
   - Asserts: `data/raw/SPY.parquet` exists after download
   - Asserts: File size > 0

2. **`test_spy_data_columns()`**
   - Validates column structure
   - Asserts: Columns are ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
   - Asserts: Date column is datetime type
   - Asserts: OHLCV columns are numeric

3. **`test_spy_data_completeness()`**
   - Validates data coverage
   - Asserts: Date range starts before 2000
   - Asserts: Date range ends within last 7 days
   - Asserts: No null values in OHLCV columns

4. **`test_spy_data_splits()`**
   - Validates train/val/test split
   - Asserts: Train data ends at 2020-12-31
   - Asserts: Val data covers 2021-01-01 to 2022-12-31
   - Asserts: Test data starts at 2023-01-01

5. **`test_download_idempotent()`**
   - Validates safe re-runs
   - Runs download twice
   - Asserts: Second run doesn't error

**Test for Directory Structure:**

**File:** `tests/test_data_directories.py`

1. **`test_data_directories_exist()`**
   - Asserts: `data/raw/` exists
   - Asserts: `data/processed/` exists
   - Asserts: `data/samples/` exists

### 2.3.3 Implement SPY Download Script

> **PROPOSITION ONLY**: Implement AFTER tests are written and failing.

**File:** `scripts/download_ohlcv.py`

**Function signature:**
```python
def download_spy(output_path: str = "data/raw/SPY.parquet") -> pd.DataFrame:
    """Download SPY OHLCV data from Yahoo Finance.

    Parameters
    ----------
    output_path : str
        Path to save parquet file

    Returns
    -------
    pd.DataFrame
        OHLCV data with columns: Date, Open, High, Low, Close, Volume

    Raises
    ------
    ValueError
        If download fails or data is invalid
    """
```

**Implementation requirements:**
- Use yfinance to download SPY data
- Handle network errors with clear messages
- Validate data before saving
- Save to parquet format
- Return DataFrame for testing
- Log download progress
- ~80 lines of code

### 2.3.4 Run Tests (RED phase)

```bash
pytest tests/test_data_download.py -v
pytest tests/test_data_directories.py -v
```

**Expected:** All tests FAIL (no implementation yet)

### 2.3.5 Implement Download Script (GREEN phase)

Write minimal code to make tests pass.

**Execution:**
```bash
python scripts/download_ohlcv.py
```

**Expected output:**
- `data/raw/SPY.parquet` created
- All tests pass

### 2.3.6 Run Tests (Verify GREEN)

```bash
make test
```

**Expected:** All tests PASS

### 2.3.7 Refactor (if needed)

Clean up code while keeping tests green.

## 2.4 Test Plan Summary

**New Files:**
- `tests/test_data_directories.py` (~20 lines)
- `tests/test_data_download.py` (~120 lines)
- `scripts/download_ohlcv.py` (~80 lines)
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `data/samples/.gitkeep`

**Total:** 6 files, ~220 lines

**Edge Cases Covered:**
- Network timeout during download
- yfinance returns partial data
- File already exists
- Invalid date ranges

## 2.5 Execution Order

1. Create data directories (2.3.1)
2. Write directory structure tests (2.3.2 - directories)
3. Write download tests (2.3.2 - download)
4. Run tests - verify RED (2.3.4)
5. Implement download script (2.3.3, 2.3.5)
6. Run tests - verify GREEN (2.3.6)
7. Refactor if needed (2.3.7)
8. Commit with tests + implementation

## 2.6 Branching Strategy

**Branch name:** `feature/phase-2-data-pipeline`
**Base:** `staging` (or `main` if staging doesn't exist yet)
**Merge target:** `staging`

**Workflow:**
```bash
git checkout main
git checkout -b feature/phase-2-data-pipeline
# Work on feature
git add -A
git commit -m "feat: add SPY data download pipeline with tests"
# When complete:
# PR to main (or staging if it exists)
```

## 2.7 Acceptance Criteria

- [ ] Data directories created with .gitkeep files
- [ ] `tests/test_data_directories.py` exists and passes
- [ ] `tests/test_data_download.py` exists with 5 test cases
- [ ] All tests pass: `make test`
- [ ] `scripts/download_ohlcv.py` implements `download_spy()`
- [ ] SPY data downloads successfully
- [ ] Data saved to `data/raw/SPY.parquet`
- [ ] Data has correct columns and types
- [ ] Data coverage validated (pre-2000 to present)
- [ ] Data splits validated (train/val/test boundaries)
- [ ] Script is idempotent (safe to re-run)
- [ ] Code committed to feature branch

## 2.8 Estimated Time

**Total:** 2-3 hours

- Planning & approval: 30 min (complete)
- Directory setup: 5 min
- Test writing: 45 min
- Implementation: 60 min
- Testing & debugging: 30 min
- Refactoring: 15 min
- Documentation & commit: 15 min

## 2.9 Next Steps (After Phase 2)

1. **Indicator Calculation** - Separate planning session
   - Use pandas-ta + TA-Lib
   - Start with 20 basic indicators
   - Save to `data/processed/`

2. **Multi-Asset Downloads** - Phase 5
   - Extend to DIA, QQQ
   - Add individual stocks
   - Add economic indicators (FRED)

3. **Data Versioning** - Separate task
   - Version schema for processed data
   - Checksums for validation

---

# Phase 3: Pipeline Design ✅ COMPLETE

**Status:** Complete (2025-12-08)

Deliverables:
- Feature engineering: `src/features/tier_a20.py` with 20 indicators
- Build script: `scripts/build_features_a20.py`
- Manifest integration for processed data versioning
- Training infrastructure decisions documented below

---

# Phase 4: Training Boilerplate ✅ COMPLETE

**Status:** Complete (2025-12-09)
**Tests:** 88 → 94 tests passing

## 4.1 Deliverables Completed

| Component | Files | Tests |
|-----------|-------|-------|
| Config System | `src/config/experiment.py` | 5 |
| Dataset Class | `src/data/dataset.py` | 8 |
| PatchTST Model | `src/models/patchtst.py` | 6 |
| Parameter Configs | `configs/model/patchtst_{2m,20m,200m,2b}.yaml` | 4 |
| Thermal Callback | `src/training/thermal.py` | 10 |
| Tracking Integration | `src/training/tracking.py` | 9 |
| Training Script | `scripts/train.py`, `src/training/trainer.py` | 7 |
| Batch Size Discovery | `scripts/find_batch_size.py` | 10 |

## 4.2 Parameter Budgets

| Budget | d_model | n_heads | n_layers | d_ff | Actual Params |
|--------|---------|---------|----------|------|---------------|
| 2M | 192 | 6 | 4 | 768 | ~1.82M |
| 20M | 512 | 8 | 6 | 2048 | ~19M |
| 200M | 1024 | 16 | 16 | 4096 | ~202M |
| 2B | 2304 | 24 | 32 | 9216 | ~2.04B |

---

# Phase 5: Data Acquisition 🔄 IN PROGRESS

**Status:** Task 1/8 complete (2025-12-09)
**Plan:** See `docs/phase5_data_acquisition_plan.md`

## 5.1 Tasks

1. ✅ Generalize download script (`download_ticker` with retry logic)
2. ⏸️ Download DIA + QQQ
3. ⏸️ Download VIX
4. ⏸️ Generalize feature pipeline
5. ⏸️ Build DIA/QQQ features (tier a20)
6. ⏸️ VIX feature engineering (tier c)
7. ⏸️ Combined dataset builder
8. ⏸️ Multi-asset builder (optional)

---

# Phase 5.5: Experiment Setup ⏸️ PROPOSED

**Objective:** Prepare infrastructure for systematic scaling experiments

## 5.5.1 Scope

| Task | Purpose | Files | Est. Lines |
|------|---------|-------|------------|
| 5.5.1 | Config templates for 4 threshold tasks | `configs/experiments/*.yaml` | +100 |
| 5.5.2 | HPO infrastructure | `scripts/run_hpo.py` | +150 |
| 5.5.3 | Timescale resampling | `scripts/resample_timescales.py` | +100 |
| 5.5.4 | Horizon config variants | `configs/experiments/{2d,3d,5d,weekly}/*.yaml` | +80 |
| 5.5.5 | Scaling curve analysis | `src/analysis/scaling_curves.py` | +150 |
| 5.5.6 | Result aggregation | `src/analysis/aggregate_results.py` | +100 |

## 5.5.2 Task Definitions

All experiments use threshold prediction tasks:

| Task | Target Definition | Output |
|------|-------------------|--------|
| threshold_1pct | P(max_price[t+h] > close[t] × 1.01) | sigmoid |
| threshold_2pct | P(max_price[t+h] > close[t] × 1.02) | sigmoid |
| threshold_3pct | P(max_price[t+h] > close[t] × 1.03) | sigmoid |
| threshold_5pct | P(max_price[t+h] > close[t] × 1.05) | sigmoid |

Where `h` = horizon in trading days (1, 2, 3, 5, or weekly ~5 days).

## 5.5.3 Success Criteria

- [ ] All 4 task configs exist and load successfully
- [ ] HPO script integrates with Optuna + W&B/MLflow
- [ ] Timescale resampling produces valid weekly data
- [ ] Analysis scripts generate scaling curves from mock data
- [ ] All tests pass: `make test`

---

# Phase 6: Scaling Experiments

**Research Question:** Do neural scaling laws apply to transformer models trained on financial time-series data?

## 6.0 Experimental Design Overview

### Constants (All Experiments)
- **Tasks:** 4 (threshold 1%, 2%, 3%, 5%)
- **Parameter budgets:** 4 (2M, 20M, 200M, 2B)
- **Data:** SPY baseline (data scaling deferred to 6D)

### Scaling Dimensions
1. **Parameters:** 2M → 20M → 200M → 2B
2. **Horizons:** 1-day → 2-day → 3-day → 5-day → weekly
3. **Features:** 20 → 100 → 250 → 500+
4. **Data:** SPY → multi-asset → cross-domain (Phase 6D)

### Experiment Totals

| Phase | HPO Runs | Final Runs | Total |
|-------|----------|------------|-------|
| 6A: Parameter Scaling | 16 | 16 | 32 |
| 6B: Horizon Scaling | 0 | 64 | 64 |
| 6C: Feature × Horizon | 48 | 240 | 288 |
| **Total (6A-6C)** | **64** | **320** | **384** |

### Runtime Estimate

| Model | Avg Time | Count | Hours |
|-------|----------|-------|-------|
| 2M | ~30 min | 96 | 48 |
| 20M | ~1.5 hr | 96 | 144 |
| 200M | ~4 hr | 96 | 384 |
| 2B | ~12 hr | 96 | 1,152 |
| **Total** | | **384** | **~1,728 hrs** |

Realistic timeline: **2-3 months** with thermal constraints and scheduling.

---

# Phase 6A: Parameter Scaling Baseline

**Objective:** Establish whether error follows power law with model parameters

## 6A.1 Experimental Matrix

**Hold Constant:**
- Features: 20 (tier a20)
- Horizon: 1-day
- Data: SPY only

**Vary:** Parameter budget (2M, 20M, 200M, 2B)

### Experiment IDs

| ID | Params | Task | Type | Purpose |
|----|--------|------|------|---------|
| 6A-HPO-2M-T1 | 2M | 1% | HPO | Find optimal hyperparams |
| 6A-HPO-2M-T2 | 2M | 2% | HPO | Find optimal hyperparams |
| 6A-HPO-2M-T3 | 2M | 3% | HPO | Find optimal hyperparams |
| 6A-HPO-2M-T5 | 2M | 5% | HPO | Find optimal hyperparams |
| 6A-HPO-20M-T1 | 20M | 1% | HPO | Find optimal hyperparams |
| 6A-HPO-20M-T2 | 20M | 2% | HPO | Find optimal hyperparams |
| 6A-HPO-20M-T3 | 20M | 3% | HPO | Find optimal hyperparams |
| 6A-HPO-20M-T5 | 20M | 5% | HPO | Find optimal hyperparams |
| 6A-HPO-200M-T1 | 200M | 1% | HPO | Find optimal hyperparams |
| 6A-HPO-200M-T2 | 200M | 2% | HPO | Find optimal hyperparams |
| 6A-HPO-200M-T3 | 200M | 3% | HPO | Find optimal hyperparams |
| 6A-HPO-200M-T5 | 200M | 5% | HPO | Find optimal hyperparams |
| 6A-HPO-2B-T1 | 2B | 1% | HPO | Find optimal hyperparams |
| 6A-HPO-2B-T2 | 2B | 2% | HPO | Find optimal hyperparams |
| 6A-HPO-2B-T3 | 2B | 3% | HPO | Find optimal hyperparams |
| 6A-HPO-2B-T5 | 2B | 5% | HPO | Find optimal hyperparams |
| 6A-2M-T1 | 2M | 1% | Final | Baseline evaluation |
| 6A-2M-T2 | 2M | 2% | Final | Baseline evaluation |
| 6A-2M-T3 | 2M | 3% | Final | Baseline evaluation |
| 6A-2M-T5 | 2M | 5% | Final | Baseline evaluation |
| 6A-20M-T1 | 20M | 1% | Final | Mid-scale evaluation |
| 6A-20M-T2 | 20M | 2% | Final | Mid-scale evaluation |
| 6A-20M-T3 | 20M | 3% | Final | Mid-scale evaluation |
| 6A-20M-T5 | 20M | 5% | Final | Mid-scale evaluation |
| 6A-200M-T1 | 200M | 1% | Final | High-scale evaluation |
| 6A-200M-T2 | 200M | 2% | Final | High-scale evaluation |
| 6A-200M-T3 | 200M | 3% | Final | High-scale evaluation |
| 6A-200M-T5 | 200M | 5% | Final | High-scale evaluation |
| 6A-2B-T1 | 2B | 1% | Final | Max-scale evaluation |
| 6A-2B-T2 | 2B | 2% | Final | Max-scale evaluation |
| 6A-2B-T3 | 2B | 3% | Final | Max-scale evaluation |
| 6A-2B-T5 | 2B | 5% | Final | Max-scale evaluation |

**Total: 32 runs (16 HPO + 16 final)**

## 6A.2 Analysis Outputs

1. **Parameter Scaling Curve:** Plot log(error) vs log(params) for each task
2. **Power Law Fit:** error ∝ N^(-α), report α and R² per task
3. **Task Difficulty Comparison:** Which thresholds benefit most from scaling?

## 6A.3 Success Criteria

- [ ] All 32 experiments complete with logged results
- [ ] Parameter scaling curves plotted for all 4 tasks
- [ ] Power law exponent (α) computed with confidence intervals
- [ ] Draft finding: "Scaling [does/does not] improve financial prediction"

## 6A.4 Minimum Publishable Result

After Phase 6A completion:
> "Empirical Test of Transformer Scaling Laws on Financial Time Series"

---

# Phase 6B: Horizon Scaling

**Objective:** Test whether scaling effectiveness varies with prediction horizon

## 6B.1 Experimental Matrix

**Hold Constant:**
- Features: 20 (tier a20)
- Data: SPY only

**Vary:**
- Parameter budget (2M, 20M, 200M, 2B)
- Horizon (2-day, 3-day, 5-day, weekly)

**Note:** Reuse HPO results from Phase 6A (same param budgets)

### Experiment Count

4 new horizons × 4 params × 4 tasks = **64 runs**

### Experiment ID Pattern

`6B-{PARAMS}-H{HORIZON}-T{THRESHOLD}`

Examples:
- `6B-2M-H2-T1`: 2M params, 2-day horizon, 1% threshold
- `6B-200M-H5-T3`: 200M params, 5-day horizon, 3% threshold
- `6B-2B-HW-T5`: 2B params, weekly horizon, 5% threshold

## 6B.2 Analysis Outputs

1. **Horizon × Parameter Heatmap:** Error rates across horizons and param budgets
2. **Scaling Curve by Horizon:** Does α vary with prediction horizon?
3. **Optimal Horizon Identification:** Which horizon benefits most from scaling?

## 6B.3 Success Criteria

- [ ] All 64 experiments complete
- [ ] Scaling curves computed for each horizon
- [ ] Horizon effect quantified with statistical significance

---

# Phase 6C: Feature × Horizon Scaling

**Objective:** Test feature scaling across all horizon combinations

## 6C.1 Experimental Matrix

**Hold Constant:**
- Data: SPY only

**Vary:**
- Features: 100, 250, 500+ (3 new tiers)
- Horizons: all 5 (1d, 2d, 3d, 5d, weekly)
- Parameters: all 4 (2M, 20M, 200M, 2B)

### Experiment Count

**HPO runs:** 3 tiers × 4 params × 4 tasks = 48 (at 1-day horizon)
**Final runs:** 3 tiers × 5 horizons × 4 params × 4 tasks = 240

**Total: 288 runs**

### Experiment ID Pattern

`6C-{PARAMS}-F{FEATURES}-H{HORIZON}-T{THRESHOLD}`

Examples:
- `6C-HPO-20M-F100-T2`: HPO for 20M, 100 features, 2% threshold
- `6C-2B-F500-H3-T1`: 2B params, 500 features, 3-day horizon, 1% threshold

## 6C.2 Feature Tiers

| Tier | Features | Indicator Categories |
|------|----------|---------------------|
| a20 | 20 | Basic (SMA, EMA, RSI, MACD, BBands, ATR, OBV, etc.) |
| a100 | 100 | + Extended momentum, volatility, volume |
| a250 | 250 | + Pattern recognition, multi-timeframe |
| a500 | 500+ | + All pandas-ta indicators |

## 6C.3 Analysis Outputs

1. **Feature × Parameter Surface:** 3D plot of error vs features vs params
2. **Feature Scaling Curves:** Does more features help or hurt?
3. **Interaction Effects:** Feature × Horizon × Parameter interactions
4. **Optimal Configuration:** Best feature/param/horizon combination

## 6C.3 Success Criteria

- [ ] All 288 experiments complete
- [ ] Feature scaling effect quantified
- [ ] Interaction effects documented
- [ ] Best configuration identified for Phase 6D

---

# Phase 6D: Data Scaling ⏸️ GATED

**Gate Condition:** Proceed only after Phase 6A-6C results show scaling promise

**Objective:** Test whether data diversity improves scaling

## 6D.1 Scope (Preliminary)

**Additional Data Sources:**
- Row B: DIA, QQQ (ETFs)
- Row C: AAPL, MSFT, GOOGL, AMZN, TSLA (stocks)
- Row D: SF Fed News Sentiment
- Row E: FRED economic indicators

**Approach:**
- Use best configuration from Phase 6C
- Test data scaling with fixed features/params/horizon
- Quantify data diversity effect on scaling

## 6D.2 Success Criteria

- [ ] Data scaling effect quantified
- [ ] Cross-asset generalization tested
- [ ] Publication-ready comprehensive scaling analysis

---

# Summary: Total Estimated Time

| Phase | Hours | Notes |
|-------|-------|-------|
| 0. Development Discipline | 4-6 | SpecKit + Superpowers setup, skills creation |
| 1. Environment | 6-8 | Includes agentic tools |
| 2. Data Pipeline Foundation | 2-3 | SPY OHLCV download with TDD |
| 3. Pipeline Design | 4-6 | With SpecKit planning |
| 4. Boilerplate | 10-14 | With TDD |
| 5. Data Acquisition | 5-7 | With task breakdown |
| 6. Experiments (initial) | Variable | Per experiment |
| **Setup Total** | **33-47** | Before first experiment |

---

# Next Steps After Phase 5

1. Implement feature calculation for all tiers (Phase 3 output)
2. Implement data normalization pipeline (Phase 3 output)
3. Implement dataset/dataloader classes (Phase 4 output)
4. Run batch size discovery for each param budget (Phase 4 output)
5. Begin Phase 6 Experiment 1: 2M-daily-direction

---

## Training Infrastructure Decisions (2025-12-08)

- **Configuration format**: Plain YAML files with lightweight `dataclass` loaders. We deliberately avoid OmegaConf or other heavy frameworks to keep configs simple, reviewable, and version-controlled alongside code.
- **Execution model**: No generic launcher scripts. Every experiment is run explicitly via CLI (e.g., `python scripts/train.py --config configs/daily/threshold_1pct.yaml`) to preserve one-task-per-model discipline and make approval gates obvious.
- **Tracking**: All runs log to both Weights & Biases (dashboards/sweeps) and local MLflow (artifacts + long-term retention). Config templates must include metadata for both systems.
- **Thermal/MPS setup**: PyTorch 2.x with MPS backend enabled by default; the training loop includes the powermetrics-based thermal callback (warn ≥85 °C, abort ≥95 °C) before scaling to larger parameter budgets.
- **Batch-size discovery**: `python scripts/find_batch_size.py --param-budget {2M|20M|200M}` remains mandatory whenever we change parameter budgets or significantly alter dataset/feature composition.

### Training Stack & Infrastructure

| Component | Decision |
|-----------|----------|
| Model | Hugging Face PatchTST, three config presets (~2M/20M/200M params) |
| Framework | PyTorch 2.x with MPS backend; `torch.backends.mps` checked before every run |
| Optimizer | AdamW + cosine schedule (configurable per YAML) |
| Tracking | W&B (dashboards/sweeps) + MLflow (local `mlruns/` mirror) every run |
| HPO | Optuna integrated with both trackers; trials respect approval gates |
| Thermal | Powermetrics hook each epoch (warn ≥85 °C, stop ≥95 °C, log to W&B/MLflow) |

### Task Explosion & Organization

- Each binary question (e.g., “Will next day’s high exceed today’s close by 1%?”) is a separate model. For daily horizon alone: 5 thresholds × 1 horizon = 5 models per parameter budget. Extending to horizons 2d/3d/5d multiplies that out (20 tasks). Coupled with the 3 parameter budgets, the daily grid alone reaches 60 training runs.
- Config layout: `configs/{horizon}/{threshold}.yaml`. YAML includes:
  - Feature tier (a20 initially)
  - Target definition (threshold %, horizon)
  - Train/val/test ranges (per PRD)
  - PatchTST parameter preset (2M/20M/200M)
  - Logging + artifact paths (W&B project/run name, MLflow experiment)
- Branching follows the experiment naming convention (`experiment/{budget}-{timescale}-{task}`) to keep approvals granular.

### Config Schema & Example

```yaml
# configs/daily/threshold_1pct.yaml
dataset:
  features_path: data/processed/v1/SPY_features_a20.parquet
  manifest_dataset: SPY.features.a20
  feature_tier: a20
target:
  type: threshold
  horizon_days: 1            # 2 / 3 / 5 for other horizons
  pct_threshold: 0.01        # 0.02 / 0.03 / 0.05 variants
training:
  param_budget: 2M           # 20M / 200M configs reuse same schema
  patchtst_config: configs/model/patchtst_2m.json
  epochs: 100
  batch_size: 32
  optimizer: adamw
  lr: 1e-4
tracking:
  wandb:
    project: financial-ts-scaling
    run_name: 2M-daily-threshold-1pct
  mlflow:
    experiment: financial-ts-scaling
    run_name: 2M-daily-threshold-1pct
```

The loader (`src/config/training.py`) validates these fields via dataclasses; new configs must pass unit tests before use.

### Target Construction Rules

- For each example at index *t*, compute `future_max = max(close[t+1 : t+horizon])` using trading days only. Label = 1 if `future_max >= close[t] * (1 + pct_threshold)` else 0.
- Horizons: 1-day (next trading day), 2-day, 3-day, 5-day. Additional horizons/timescales will follow the same rule.
- Regression or multi-head tasks will have dedicated schemas later; Phase 2 focuses on binary thresholds.

### First Task Checklist (Cold Start)

1. Restore context via session_restore skill; confirm Phase 2 status in `session_context.md`.
2. Activate environment + verification: `source venv/bin/activate`, `make test`, `make verify`.
3. Ensure features exist: `python scripts/build_features_a20.py` (rebuild if manifest missing entry).
4. Run batch-size discovery for the target budget: `python scripts/find_batch_size.py --param-budget {2M|20M|200M}`.
5. Create or reuse the correct config YAML (per schema above).
6. Launch training: `python scripts/train.py --config configs/daily/threshold_1pct.yaml`.
7. Monitor console for thermal warnings; ensure W&B + MLflow runs are created.
8. After completion, update manifests (if new processed artifacts), session_context, and phase_tracker.
9. Repeat for the next task/horizon with the appropriate config/branch.

### Testing & Optimization Strategy

1. **Unit tests**: dataset builder (label generation per horizon/threshold), PatchTST wrapper (input/output shapes), thermal callback (mock powermetrics), config loader (YAML→dataclass).
2. **Smoke tests**: micro-training on <30 days of data, ensuring forward/backward pass on MPS works for each parameter budget before long runs.
3. **Batch-size discovery**: scripted command per budget before new feature/data tiers. Results logged to both trackers.
4. **Regression checks**: `make test` covers data pipeline + config loader; `make verify` ensures environment and manifests are intact before training runs.
5. **Thermal validation**: dedicated tests assert the callback pauses/aborts when simulated temp crosses thresholds; runs log the thermal state to both trackers for audit.

### Tooling Notes

- No centralized launcher: we prefer explicit CLI invocation for clarity, approvals, and to avoid hiding configuration changes.
- Configs are plain YAML; we use standard `yaml.safe_load` + dataclasses to avoid OmegaConf complexity.
- Any automation (e.g., shell scripts queuing multiple tasks) must still call the explicit training script per run and log to both W&B/MLflow; such scripts require their own approval gate.

---

# Appendix A: Code Proposition Disclaimer

**ALL code in this document is a PROPOSITION requiring:**

1. **Planning session** using `planning_session.md` skill
2. **Task breakdown** using `task_breakdown.md` skill with RACI
3. **Branching strategy** proposed and approved
4. **Test-first development** using `test_first.md` skill
5. **Code review** by Alex with explicit approval
6. **Approval gate** for any changes >50 lines

**Do NOT copy-paste code directly. Always:**
- Start with planning discussion
- Write tests before implementation
- Request approval before proceeding
- Follow the RACI matrix

---

*Document version: v2*  
*Status: Active development plan*  
*Created: 2025-11-26*  
*Supersedes: phase_plans_v1.md*