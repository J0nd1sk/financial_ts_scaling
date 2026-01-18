> **ARCHIVED** - This document has been superseded. See docs/project_history.md and docs/project_phase_plans.md for current information.

# Financial Time-Series Transformer Scaling Experiments
## Comprehensive Project Plan

---

# Part I: Project Overview

## 1. Research Objectives

### 1.1 Primary Research Questions

1. **Parameter Scaling**: Does increasing model parameters (2M â†’ 20M â†’ 200M) improve prediction accuracy following a power law?
2. **Feature Scaling**: Does increasing feature count (20 â†’ 100 â†’ 500 â†’ 1000 â†’ 2000) improve accuracy?
3. **Data Scaling**: Does increasing training data diversity (single asset â†’ multiple assets â†’ cross-asset) improve accuracy?
4. **Data Quality Scaling**: Does adding richer data types (price-only â†’ +sentiment â†’ +volatility â†’ +search trends) improve accuracy?
5. **Interaction Effects**: How do these scaling dimensions interact?

### 1.2 Success Criteria

**Minimum Viable:**
- Complete Phase 1 (parameter scaling) experiments
- Generate scaling curves
- Determine if power law holds
- Publish findings (Medium minimum)

**Full Success:**
- Complete all phases
- Clear scaling law characterization
- Publish on arXiv or SSRN
- Reproducible code released
- Community engagement

### 1.3 Publication Targets

- **Primary**: Medium (Towards Data Science)
- **Secondary**: arXiv (Quantitative Finance / Machine Learning)
- **Tertiary**: SSRN
- **Code**: GitHub with full reproducibility

---

# Part II: Environment & Infrastructure

## 2. Development Environment

### 2.1 Hardware

- **Machine**: MacBook Pro M4 Max, 128GB Unified Memory
- **Storage**: Internal SSD + External SSD for backups
- **Cooling**: Basement environment (50-60Â°F ambient), elevated hard surface

### 2.2 Software Stack

| Category | Tool | Version | Notes |
|----------|------|---------|-------|
| Python | Python | 3.12.x | Stable middle ground |
| ML Framework | PyTorch | 2.6+ | MPS backend for M4 |
| Transformers | Hugging Face Transformers | Latest | PatchTST implementation |
| HPO | Optuna | Latest | Bayesian optimization |
| Experiment Tracking | Weights & Biases | Latest | Primary visualization |
| Experiment Tracking | MLflow | Latest | Local, model registry learning |
| Data Processing | pandas, polars | Latest | Data manipulation |
| Data Storage | PyArrow | Latest | Parquet I/O |
| Indicators | pandas-ta + TA-Lib | Latest | Speed + convenience |
| Uncertainty | MAPIE or crepes | Latest | Conformal prediction |
| Gradient Boosting | CatBoost | Latest | Meta-module (tentative) |

### 2.3 Python Environment Setup

```bash
# Create environment
python3.12 -m venv .venv
source .venv/bin/activate

# Core ML
pip install torch torchvision torchaudio
pip install transformers datasets accelerate

# Data & Features
pip install pandas polars pyarrow fastparquet
pip install yfinance pandas-datareader fredapi
pip install pandas-ta
# TA-Lib requires separate C library installation:
# brew install ta-lib
pip install TA-Lib

# HPO & Tracking
pip install optuna optuna-dashboard
pip install wandb mlflow

# Utilities
pip install scikit-learn matplotlib plotly
pip install mapie  # conformal prediction
pip install catboost

# Development
pip install pytest pytest-cov black isort mypy
pip install ipython jupyter
```

### 2.4 Verification Checklist

```python
# verification_script.py
import torch
import transformers
import optuna
import wandb
import mlflow
import pandas as pd
import pyarrow.parquet as pq
import pandas_ta as ta

# Check MPS
assert torch.backends.mps.is_available(), "MPS not available"
print(f"PyTorch: {torch.__version__}, MPS: âœ“")

# Check transformers
from transformers import PatchTSTConfig, PatchTSTForPrediction
config = PatchTSTConfig(num_input_channels=1, context_length=64, prediction_length=1)
model = PatchTSTForPrediction(config)
print(f"PatchTST: âœ“ ({sum(p.numel() for p in model.parameters())} params)")

# Check Optuna
study = optuna.create_study()
print(f"Optuna: âœ“")

# Check tracking
print(f"W&B: âœ“")
print(f"MLflow: âœ“")

print("\nâœ… All verifications passed")
```

---

## 3. Infrastructure & Accounts

### 3.1 Required Accounts

| Service | Purpose | Required? |
|---------|---------|-----------|
| GitHub | Code repository | Yes |
| Weights & Biases | Experiment tracking | Yes |
| FRED API | Economic data | Yes (free key) |
| Hugging Face | Model hosting (later) | Optional |
| Google Cloud / AWS | Backup storage | Optional |

### 3.2 Local Infrastructure

- External SSD for data/checkpoint backup
- Laptop elevated on hard surface for airflow
- Temperature monitoring tools installed
- Basement environment (natural cooling)

### 3.3 Directory Structure

```
financial-ts-scaling/
â”œâ”€â”€ .venv/                    # Virtual environment
â”œâ”€â”€ .git/                     # Git repository
â”œâ”€â”€ configs/                  # Experiment configurations
â”‚   â”œâ”€â”€ phase1/              # Parameter scaling configs
â”‚   â”œâ”€â”€ phase2/              # Feature scaling configs
â”‚   â”œâ”€â”€ phase3/              # Data scaling configs
â”‚   â”œâ”€â”€ phase4/              # Quality scaling configs
â”‚   â””â”€â”€ phase5/              # Interaction configs
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                 # Immutable raw downloads
â”‚   â”‚   â”œâ”€â”€ ohlcv/          # Price data parquet files
â”‚   â”‚   â”œâ”€â”€ fred/           # Economic data
â”‚   â”‚   â””â”€â”€ sentiment/      # Sentiment data
â”‚   â”œâ”€â”€ processed/          # Versioned processed data
â”‚   â”‚   â””â”€â”€ v1/
â”‚   â””â”€â”€ samples/            # Small CSV samples for LLM review
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data/               # Download, clean, validate
â”‚   â”œâ”€â”€ features/           # Indicator calculations
â”‚   â”œâ”€â”€ models/             # PatchTST configs, wrappers
â”‚   â”œâ”€â”€ training/           # Train loops, HPO
â”‚   â”œâ”€â”€ evaluation/         # Metrics, plotting
â”‚   â””â”€â”€ utils/              # Logging, checkpoints, thermal
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ unit/
â”‚   â””â”€â”€ integration/
â”œâ”€â”€ notebooks/               # Exploration only
â”œâ”€â”€ scripts/                 # CLI entry points
â”œâ”€â”€ outputs/
â”‚   â”œâ”€â”€ checkpoints/        # Model checkpoints
â”‚   â”œâ”€â”€ results/            # Experiment results
â”‚   â””â”€â”€ figures/            # Generated plots
â”œâ”€â”€ docs/
â”‚   â””â”€â”€ runbooks/           # Operational runbooks
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ pyproject.toml
â””â”€â”€ README.md
```

---

# Part III: Data Pipeline

## 4. Data Acquisition

### 4.1 Data Sources Overview

| Data Type | Source | Frequency | History Start | Format |
|-----------|--------|-----------|---------------|--------|
| OHLCV (SPY, DOW, QQQ) | Yahoo Finance | Daily | 1990s | Parquet |
| OHLCV (Major Stocks) | Yahoo Finance | Daily | Varies | Parquet |
| Treasury Yields | FRED | Daily | 1960s | Parquet |
| Fed Funds Rate | FRED | Daily | 1954 | Parquet |
| VIX | Yahoo/CBOE | Daily | 1990 | Parquet |
| SF Fed News Sentiment | SF Fed | Daily | 1980 | Parquet |
| Google Trends | pytrends | Weekly | 2004 | Parquet |

### 4.2 Asset Universe

**Tier 1 (Core):**
- SPY (S&P 500 ETF)
- DIA (Dow Jones ETF)
- QQQ (NASDAQ 100 ETF)

**Tier 2 (Major Stocks):**
- AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA

**Tier 3 (Economic):**
- ^TNX (10-Year Treasury)
- ^VIX (Volatility Index)
- LIBOR rates (via FRED)
- Fed Funds Rate (via FRED)

### 4.3 Parquet File Format

**What is Parquet:**
- Columnar binary storage format
- Compressed (2-10x smaller than CSV)
- Schema-aware (preserves dtypes)
- Supports partial column reads
- Industry standard for data engineering

**Why use Parquet:**
- 10-100x faster to load than CSV
- Type safety (no dtype inference bugs)
- Efficient for large datasets

**Note on LLMs:**
- LLMs cannot read Parquet directly (binary format)
- Create small CSV samples (~1000 rows) in `data/samples/` for code review sessions
- Use `df.head()`, `.describe()`, `.info()` to show data structure

### 4.4 Download Scripts

```python
# src/data/download_ohlcv.py
import yfinance as yf
import pandas as pd
from pathlib import Path

def download_ohlcv(symbol: str, start: str, end: str, output_dir: Path) -> Path:
    """Download OHLCV data and save as parquet."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end)
    
    # Validate
    assert not df.empty, f"No data for {symbol}"
    assert not df.isnull().all().any(), f"All-null columns in {symbol}"
    
    # Save
    output_path = output_dir / f"{symbol.lower()}_daily.parquet"
    df.to_parquet(output_path, compression="snappy")
    
    return output_path
```

```python
# src/data/download_fred.py
from fredapi import Fred
import pandas as pd
from pathlib import Path

def download_fred_series(series_id: str, api_key: str, output_dir: Path) -> Path:
    """Download FRED series and save as parquet."""
    fred = Fred(api_key=api_key)
    df = fred.get_series(series_id)
    df = df.to_frame(name=series_id.lower())
    
    output_path = output_dir / f"fred_{series_id.lower()}.parquet"
    df.to_parquet(output_path, compression="snappy")
    
    return output_path
```

### 4.5 Backup Protocol

**Raw Data (Immutable):**
- Never modify after download
- Store in `data/raw/`
- Backup to external SSD
- Optional: cloud backup (S3/GCS)

**Processed Data (Versioned):**
- Store in `data/processed/v{N}/`
- Document processing parameters
- Regenerate from raw as needed

**Backup Frequency:**
- After initial download: full backup
- After each processing version: incremental backup

**Checksums:**
```bash
# Generate checksums for raw data
find data/raw -name "*.parquet" -exec md5sum {} \; > data/raw/checksums.md5

# Verify checksums
md5sum -c data/raw/checksums.md5
```

---

## 5. Feature Engineering Pipeline

### 5.1 Pipeline Stages

```
Raw OHLCV â†’ Cleaning â†’ Indicator Calculation â†’ Multi-Timeframe Alignment â†’ 
Sentiment Integration â†’ Normalization â†’ Windowing â†’ DataLoader
```

### 5.2 Indicator Categories

**Price-Derived (Tier 1 - always included):**
- Returns (1d, 2d, 5d, 10d, 20d)
- Log returns
- OHLC ratios
- Range (High-Low)
- Gap (Open vs prev Close)

**Moving Averages (configurable count):**
- SMA: 5, 10, 20, 50, 100, 200 periods
- EMA: 5, 10, 20, 50, 100, 200 periods
- WMA, DEMA, TEMA variants
- MA crossover signals
- Price distance from MAs

**Momentum:**
- RSI (14, 21, 50)
- MACD (12/26/9)
- Stochastic (14, 3, 3)
- ROC (various periods)
- Williams %R
- CCI

**Volatility:**
- ATR (14, 21)
- Bollinger Bands (20, 2)
- Keltner Channels
- Historical volatility (10, 20, 50 day)
- Parkinson volatility
- Garman-Klass volatility

**Volume:**
- OBV
- Volume SMA ratios
- VWAP
- Money Flow Index
- Accumulation/Distribution

**Trend:**
- ADX
- Aroon
- Parabolic SAR
- Supertrend

### 5.3 Feature Scaling Tiers

| Tier | Feature Count | Composition |
|------|---------------|-------------|
| Minimal | ~20 | Basic OHLCV + returns + 3 MAs + RSI + MACD |
| Small | ~50 | + more MAs + momentum + basic volatility |
| Medium | ~100 | + volume indicators + trend + cross-signals |
| Large | ~200 | + multiple periods per indicator + derived |
| XL | ~500 | + all indicator variants + lagged features |
| XXL | ~1000 | + interaction features + polynomials |
| Max | ~2000 | + all possible combinations |

### 5.4 Indicator Calculation with pandas-ta + TA-Lib

```python
# src/features/calculate_indicators.py
import pandas as pd
import pandas_ta as ta

def calculate_indicators(df: pd.DataFrame, tier: str = "medium") -> pd.DataFrame:
    """Calculate technical indicators using pandas-ta with TA-Lib backend."""
    
    # Use TA-Lib backend for speed when available
    df.ta.cores = 0  # Use all cores
    
    if tier in ["minimal", "small", "medium", "large", "xl", "xxl", "max"]:
        # Basic returns
        df["return_1d"] = df["Close"].pct_change(1)
        df["return_5d"] = df["Close"].pct_change(5)
        df["return_20d"] = df["Close"].pct_change(20)
        
        # Moving averages (use talib=True for speed)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{period}"] = ta.sma(df["Close"], length=period, talib=True)
            df[f"ema_{period}"] = ta.ema(df["Close"], length=period, talib=True)
        
        # RSI
        df["rsi_14"] = ta.rsi(df["Close"], length=14, talib=True)
        
        # MACD
        macd = ta.macd(df["Close"], talib=True)
        df = pd.concat([df, macd], axis=1)
    
    if tier in ["small", "medium", "large", "xl", "xxl", "max"]:
        # Bollinger Bands
        bbands = ta.bbands(df["Close"], length=20, std=2, talib=True)
        df = pd.concat([df, bbands], axis=1)
        
        # ATR
        df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14, talib=True)
        
        # Stochastic
        stoch = ta.stoch(df["High"], df["Low"], df["Close"], talib=True)
        df = pd.concat([df, stoch], axis=1)
    
    # Continue for higher tiers...
    
    return df
```

### 5.5 Multi-Timeframe Feature Construction

For the 8th timescale (daily with hierarchical features):

```python
# src/features/multi_timeframe.py
import pandas as pd

def add_higher_timeframe_features(daily_df: pd.DataFrame, 
                                   weekly_df: pd.DataFrame,
                                   monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Add weekly and monthly indicators to daily data."""
    
    # Forward-fill weekly indicators to daily
    for col in weekly_df.columns:
        if col not in ["Open", "High", "Low", "Close", "Volume"]:
            daily_df[f"weekly_{col}"] = weekly_df[col].reindex(
                daily_df.index, method="ffill"
            )
    
    # Previous week's OHLC
    daily_df["prev_week_high"] = weekly_df["High"].shift(1).reindex(
        daily_df.index, method="ffill"
    )
    daily_df["prev_week_low"] = weekly_df["Low"].shift(1).reindex(
        daily_df.index, method="ffill"
    )
    daily_df["prev_week_close"] = weekly_df["Close"].shift(1).reindex(
        daily_df.index, method="ffill"
    )
    
    # Same for monthly
    for col in monthly_df.columns:
        if col not in ["Open", "High", "Low", "Close", "Volume"]:
            daily_df[f"monthly_{col}"] = monthly_df[col].reindex(
                daily_df.index, method="ffill"
            )
    
    daily_df["prev_month_high"] = monthly_df["High"].shift(1).reindex(
        daily_df.index, method="ffill"
    )
    # ... etc
    
    return daily_df
```

### 5.6 Sentiment Features

```python
# src/features/sentiment_indicators.py
import pandas as pd

def calculate_sentiment_indicators(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived sentiment indicators."""
    
    # Sentiment momentum
    sentiment_df["sentiment_momentum_5d"] = sentiment_df["sentiment"].diff(5)
    sentiment_df["sentiment_momentum_20d"] = sentiment_df["sentiment"].diff(20)
    
    # Sentiment moving averages
    sentiment_df["sentiment_sma_10"] = sentiment_df["sentiment"].rolling(10).mean()
    sentiment_df["sentiment_sma_50"] = sentiment_df["sentiment"].rolling(50).mean()
    
    # Sentiment regime (above/below average)
    sentiment_df["sentiment_regime"] = (
        sentiment_df["sentiment"] > sentiment_df["sentiment"].rolling(200).mean()
    ).astype(int)
    
    # Sentiment rate of change
    sentiment_df["sentiment_roc_10"] = sentiment_df["sentiment"].pct_change(10)
    
    return sentiment_df
```

### 5.7 Normalization

```python
# src/features/normalize.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

def normalize_features(df: pd.DataFrame, 
                       fit: bool = True,
                       scaler_path: Path = None) -> tuple[pd.DataFrame, StandardScaler]:
    """Z-score normalize features."""
    
    feature_cols = [c for c in df.columns if c not in ["Date", "target"]]
    
    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        if scaler_path:
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
    else:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df, scaler
```

---

# Part IV: Experimental Design

## 6. Dataset Matrix

### 6.1 Two-Dimensional Dataset Naming

**Asset Dimension (rows):**
- A = SPY only
- B = SPY + DIA
- C = SPY + DIA + QQQ
- D = SPY + DIA + QQQ + Major Stocks (AAPL, MSFT, GOOGL, NVDA, etc.)
- E = D + Economic (Treasury yields, LIBOR, Fed Funds)

**Feature Quality Dimension (columns):**
- a = OHLCV + calculated indicators only
- b = a + SF Fed Sentiment (+ sentiment-derived indicators)
- c = b + VIX (+ volatility regime indicators)
- d = c + Google Trends (+ search momentum indicators)

**History Constraints by Column:**
- Column a: Longest history (1950s+ for some assets)
- Column b: 1980+ (SF Fed Sentiment start)
- Column c: 1990+ (VIX start)
- Column d: 2004+ (Google Trends start)

**Full Matrix:**
```
     a    b    c    d
A   Aa   Ab   Ac   Ad
B   Ba   Bb   Bc   Bd
C   Ca   Cb   Cc   Cd
D   Da   Db   Dc   Dd
E   Ea   Eb   Ec   Ed
```

### 6.2 Feature Count Dimension (within column 'a')

Feature count is a separate variable tested in Phase 2:
- aâ‚‚â‚€ = 20 indicators
- aâ‚…â‚€ = 50 indicators
- aâ‚â‚€â‚€ = 100 indicators
- aâ‚‚â‚€â‚€ = 200 indicators
- aâ‚…â‚€â‚€ = 500 indicators
- aâ‚â‚€â‚€â‚€ = 1000 indicators
- aâ‚‚â‚€â‚€â‚€ = 2000 indicators

---

## 7. Timescales & Tasks

### 7.1 Timescales (8 total)

| # | Timescale | Bar Period | Notes |
|---|-----------|------------|-------|
| 1 | Daily | 1 day | Base timescale |
| 2 | 2-Day | 2 days | |
| 3 | 3-Day | 3 days | |
| 4 | 5-Day | 5 days | Trading week |
| 5 | Weekly | 7 days | |
| 6 | 2-Week | 14 days | |
| 7 | Monthly | ~21 trading days | |
| 8 | **Daily + Multi-Resolution** | 1 day | Daily with weekly/monthly indicators as features |

### 7.2 Tasks (6 per timescale)

| Task | Type | Output | Head | Loss |
|------|------|--------|------|------|
| Direction | Binary Classification | P(up) | Sigmoid | BCE |
| Threshold >1% | Binary Classification | P(>1%) | Sigmoid | BCE |
| Threshold >2% | Binary Classification | P(>2%) | Sigmoid | BCE |
| Threshold >3% | Binary Classification | P(>3%) | Sigmoid | BCE |
| Threshold >5% | Binary Classification | P(>5%) | Sigmoid | BCE |
| Price Regression | Regression | Predicted price/return | Linear | MSE |

**Note:** Each task = separate model. One model per task per timescale.

### 7.3 Models per Dataset

- 8 timescales Ã— 6 tasks = **48 models per dataset**
- With 3 parameter budgets = 144 training runs per dataset (in full expansion)

---

## 8. Experimental Phases

### Phase 1: Parameter Scaling

**Objective:** Test if model capacity improves accuracy (power law?)

**Control Variables:**
- Dataset: Aa (SPY only, OHLCV + indicators)
- Features: ~50 indicators
- Timescale: Daily
- Task: Direction (binary)

**Independent Variable:**
- Parameters: 2M â†’ 20M â†’ 200M

**Experiments:**
| ID | Params | Features | Data | Timescale | Task |
|----|--------|----------|------|-----------|------|
| P1-2M | 2M | 50 | Aa | Daily | Direction |
| P1-20M | 20M | 50 | Aa | Daily | Direction |
| P1-200M | 200M | 50 | Aa | Daily | Direction |

**Analysis:**
- Plot accuracy vs log(params)
- Fit power law: error âˆ params^(-Î±)
- Estimate scaling exponent Î±
- Statistical significance tests

---

### Phase 2: Feature Scaling

**Objective:** Test if more features improve accuracy

**Control Variables:**
- Dataset: Aa (SPY only)
- Parameters: 20M (middle tier)
- Timescale: Daily
- Task: Direction

**Independent Variable:**
- Features: 20 â†’ 50 â†’ 100 â†’ 200 â†’ 500 â†’ 1000 â†’ 2000

**Experiments:**
| ID | Params | Features | Data | Timescale | Task |
|----|--------|----------|------|-----------|------|
| P2-20 | 20M | 20 | Aa | Daily | Direction |
| P2-50 | 20M | 50 | Aa | Daily | Direction |
| P2-100 | 20M | 100 | Aa | Daily | Direction |
| P2-200 | 20M | 200 | Aa | Daily | Direction |
| P2-500 | 20M | 500 | Aa | Daily | Direction |
| P2-1000 | 20M | 1000 | Aa | Daily | Direction |
| P2-2000 | 20M | 2000 | Aa | Daily | Direction |

**Analysis:**
- Plot accuracy vs log(features)
- Identify saturation point (if any)
- Compare to Phase 1 scaling rate

---

### Phase 3: Data Scaling

**Objective:** Test if more training data (asset diversity) improves accuracy

**Control Variables:**
- Features: ~100 indicators per asset
- Parameters: 20M
- Timescale: Daily
- Task: Direction
- Feature quality: Column 'a' (OHLCV + indicators only)

**Independent Variable:**
- Data scope: A â†’ B â†’ C â†’ D â†’ E

**Experiments:**
| ID | Params | Features | Data | Timescale | Task |
|----|--------|----------|------|-----------|------|
| P3-A | 20M | 100/asset | Aa | Daily | Direction |
| P3-B | 20M | 100/asset | Ba | Daily | Direction |
| P3-C | 20M | 100/asset | Ca | Daily | Direction |
| P3-D | 20M | 100/asset | Da | Daily | Direction |
| P3-E | 20M | 100/asset | Ea | Daily | Direction |

**Analysis:**
- Plot accuracy vs dataset size/diversity
- Test if cross-asset training helps or hurts

---

### Phase 4: Data Quality Scaling

**Objective:** Test if richer data types improve accuracy beyond quantity

**Control Variables:**
- Asset scope: A (SPY only)
- Parameters: 20M
- Timescale: Daily
- Task: Direction

**Independent Variable:**
- Data quality: a â†’ b â†’ c â†’ d (progressively richer)

**Experiments:**
| ID | Params | Features | Data | Timescale | Task |
|----|--------|----------|------|-----------|------|
| P4-a | 20M | 100+ | Aa | Daily | Direction |
| P4-b | 20M | 100+ | Ab | Daily | Direction |
| P4-c | 20M | 100+ | Ac | Daily | Direction |
| P4-d | 20M | 100+ | Ad | Daily | Direction |

**Note:** Feature count varies as sentiment, VIX, and trends add derived indicators.

**Analysis:**
- Plot accuracy vs data quality tier
- Quantify value of each additional data source

---

### Phase 5: Interaction Effects

**Objective:** Test how scaling dimensions interact

**Selected Combinations:**
| ID | Params | Features | Data | Timescale | Task |
|----|--------|----------|------|-----------|------|
| P5-1 | 2M | 20 | Aa | Daily | Direction |
| P5-2 | 2M | 500 | Aa | Daily | Direction |
| P5-3 | 200M | 20 | Aa | Daily | Direction |
| P5-4 | 200M | 500 | Aa | Daily | Direction |
| P5-5 | 2M | 100 | Ea | Daily | Direction |
| P5-6 | 200M | 100 | Ea | Daily | Direction |
| P5-7 | 20M | 500 | Ed | Daily | Direction |
| P5-8 | 200M | 500 | Ed | Daily | Direction |

**Analysis:**
- 2Ã—2 interaction plots
- Test for synergy vs diminishing returns
- Identify optimal configuration

---

### Phase 6: Full Expansion

**Objective:** Apply best configuration to all tasks and timescales

**Using best configuration from Phases 1-5:**
- Best parameter count
- Best feature count
- Best data scope
- Best data quality

**Train 48 models:**
- 8 timescales Ã— 6 tasks
- Optionally re-validate with all 3 parameter budgets (144 models)

---

## 9. Train/Validation/Test Split

### 9.1 Temporal Split (Strict)

| Split | Date Range | Purpose |
|-------|------------|---------|
| Train | Start â†’ 2020-12-31 | Model training |
| Validation | 2021-01-01 â†’ 2022-12-31 | HPO, early stopping |
| Test | 2023-01-01 â†’ Present | Final evaluation only |

**Critical:** No future leakage. All features must be computed with only past data.

### 9.2 Walk-Forward Validation (Optional)

For robustness, implement rolling window validation:
- Train on years 1-10, validate on year 11
- Train on years 2-11, validate on year 12
- etc.

---

## 10. Metrics

### 10.1 Classification Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | When predicting up, how often correct |
| Recall | TP/(TP+FN) | What fraction of ups did we catch |
| F1 | 2Ã—(PÃ—R)/(P+R) | Balanced metric |
| ROC-AUC | Area under ROC curve | Discrimination ability |
| Brier Score | Mean squared prob error | Probability calibration |

### 10.2 Regression Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| MSE | Mean(pred - actual)Â² | Standard loss |
| MAE | Mean(|pred - actual|) | Robust to outliers |
| MAPE | Mean(|pred - actual|/actual) | Percentage error |
| RÂ² | 1 - SS_res/SS_tot | Variance explained |

### 10.3 Calibration Metrics

| Metric | Description |
|--------|-------------|
| Expected Calibration Error (ECE) | Avg gap between confidence and accuracy |
| Reliability Diagram | Plot predicted prob vs actual freq |
| Conformal Coverage | % of true values in prediction intervals |

### 10.4 Trading Metrics (Secondary)

| Metric | Description |
|--------|-------------|
| Directional Accuracy | % correct up/down predictions |
| Hypothetical Returns | Simulated P&L from signals |
| Sharpe Ratio | Risk-adjusted returns |
| Max Drawdown | Worst peak-to-trough |

---

# Part V: Model Architecture

## 11. PatchTST Configuration

### 11.1 Base Architecture

PatchTST = Patch Time Series Transformer
- Encoder-only transformer
- Patches input time series into tokens
- Channel-independent (each feature series shares backbone)

### 11.2 Parameter Budget Configurations

| Budget | d_model | num_layers | num_heads | ffn_dim | ~Params |
|--------|---------|------------|-----------|---------|---------|
| 2M | 64 | 3 | 4 | 256 | ~2M |
| 20M | 256 | 6 | 8 | 1024 | ~20M |
| 200M | 512 | 12 | 16 | 2048 | ~200M |

**Note:** Exact values TBD via parameter counting. These are estimates.

### 11.3 Configuration Code

```python
# src/models/patchtst_configs.py
from transformers import PatchTSTConfig

def get_patchtst_config(
    param_budget: str,
    num_input_channels: int,
    context_length: int = 64,
    prediction_length: int = 1,
    patch_length: int = 16
) -> PatchTSTConfig:
    """Get PatchTST config for parameter budget."""
    
    configs = {
        "2M": {
            "d_model": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "ffn_dim": 256,
        },
        "20M": {
            "d_model": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "ffn_dim": 1024,
        },
        "200M": {
            "d_model": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 16,
            "ffn_dim": 2048,
        },
    }
    
    cfg = configs[param_budget]
    
    return PatchTSTConfig(
        num_input_channels=num_input_channels,
        context_length=context_length,
        prediction_length=prediction_length,
        patch_length=patch_length,
        d_model=cfg["d_model"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        ffn_dim=cfg["ffn_dim"],
        dropout=0.1,
        attention_dropout=0.1,
    )
```

### 11.4 Output Heads

**Classification (Direction, Thresholds):**
```python
class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int = 1):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, d_model) - pooled representation
        return self.sigmoid(self.linear(x))
```

**Regression (Price):**
```python
class RegressionHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
    
    def forward(self, x):
        return self.linear(x)
```

### 11.5 Probability Scoring

**Classification:** Sigmoid output is P(positive class). Use directly.

**Regression:** Use conformal prediction for uncertainty quantification:
```python
from mapie.regression import MapieRegressor

# Wrap trained model
mapie = MapieRegressor(model, method="plus", cv="prefit")
mapie.fit(X_calib, y_calib)

# Predict with intervals
y_pred, y_pis = mapie.predict(X_test, alpha=0.1)  # 90% interval
```

---

## 12. Meta-Module (Tentative)

### 12.1 Purpose

Combine outputs from multiple Base Modules (different timescales, tasks) into final trading signal.

### 12.2 Architecture

- **Input:** Concatenated predictions from Base Modules + selected raw features
- **Model:** CatBoost classifier/regressor
- **Output:** Final signal with calibrated probability

### 12.3 Training Protocol

- Train on held-out fold to prevent leakage
- Use predictions from models trained on different data split
- Or: Use cross-validated predictions (careful implementation)

**Note:** This is tentative. Implement only after validating Base Module performance.

---

# Part VI: Training Protocol

## 13. Training Configuration

### 13.1 Optimizer Settings

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 (tunable) |
| Weight Decay | 0.01 |
| Betas | (0.9, 0.999) |
| Gradient Clipping | 1.0 |

### 13.2 Learning Rate Schedule

- **Warmup:** Linear warmup for first 10% of steps
- **Decay:** Cosine annealing to 1e-6

### 13.3 Early Stopping

- **Monitor:** Validation loss
- **Patience:** 10 epochs
- **Mode:** Minimize

### 13.4 Training Limits

- **Max Epochs:** 100 (usually stops earlier)
- **Batch Size:** Determined per architecture (see Section 15)

---

## 14. Hyperparameter Optimization

### 14.1 Optuna Configuration

```python
# src/training/hpo.py
import optuna

def create_objective(param_budget: str, dataset: str, task: str):
    def objective(trial: optuna.Trial) -> float:
        # Architecture (within param budget constraint)
        d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 2, 12)
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
        ffn_dim = trial.suggest_categorical("ffn_dim", [256, 512, 1024, 2048])
        
        # Verify param count within budget
        param_count = estimate_params(d_model, num_layers, num_heads, ffn_dim)
        budget_limits = {"2M": 3e6, "20M": 30e6, "200M": 300e6}
        if param_count > budget_limits[param_budget]:
            raise optuna.TrialPruned()
        
        # Training hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        context_length = trial.suggest_categorical("context_length", [32, 64, 128])
        patch_length = trial.suggest_categorical("patch_length", [8, 16, 32])
        
        # Train and return validation metric
        val_loss = train_model(...)
        return val_loss
    
    return objective
```

### 14.2 HPO Budget

| Phase | Trials per Experiment |
|-------|----------------------|
| Phase 1-4 | 50 trials |
| Phase 5 | 30 trials (reduced for interactions) |
| Phase 6 | 20 trials (apply best settings) |

### 14.3 Pruning

- **Algorithm:** Median Pruner
- **Start:** After 10 epochs
- **Prune if:** Below median of completed trials

### 14.4 Reporting

- Report best config
- Report best, median, worst metrics
- Save Optuna study for analysis

---

## 15. Batch Size Optimization

### 15.1 When to Tune

| Trigger | Action |
|---------|--------|
| New parameter budget (2M â†’ 20M â†’ 200M) | **Required** - re-tune |
| New dataset (Aa â†’ Ba) | Recommended - quick check |
| New feature count (50 â†’ 100) | Optional - usually minimal impact |

### 15.2 Tuning Method

```python
# src/training/batch_size.py
import torch

def find_max_batch_size(
    model: torch.nn.Module,
    sample_input_shape: tuple,
    start: int = 8,
    max_attempts: int = 10
) -> int:
    """Binary search for maximum batch size that fits in memory."""
    
    model = model.to("mps")
    batch_size = start
    max_working = start
    
    for _ in range(max_attempts):
        try:
            # Create dummy batch
            x = torch.randn(batch_size, *sample_input_shape, device="mps")
            
            # Forward + backward
            model.train()
            out = model(x)
            loss = out.mean()
            loss.backward()
            
            # Clear memory
            del x, out, loss
            torch.mps.empty_cache()
            
            # Success - try larger
            max_working = batch_size
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.mps.empty_cache()
                break
            raise
    
    # Use 80% of max for safety margin
    return int(max_working * 0.8)
```

### 15.3 Recording

Log batch size in experiment metadata for reproducibility.

---

## 16. Checkpointing

### 16.1 Strategy

- Save every N epochs (default: 5)
- Save on validation improvement
- Save on interrupt (SIGINT handler)
- Keep last K checkpoints (default: 3)

### 16.2 Checkpoint Contents

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_loss": best_val_loss,
    "config": config_dict,
    "metrics_history": metrics_history,
}
torch.save(checkpoint, f"checkpoints/exp_{exp_id}_epoch_{epoch}.pt")
```

### 16.3 Resume Training

```python
def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]
```

---

## 17. Experiment Tracking

### 17.1 Weights & Biases (Primary)

```python
import wandb

wandb.init(
    project="financial-ts-scaling",
    name=f"{exp_id}_{param_budget}_{dataset}",
    config={
        "param_budget": param_budget,
        "dataset": dataset,
        "task": task,
        "timescale": timescale,
        **hyperparameters
    }
)

# Log per epoch
wandb.log({
    "epoch": epoch,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "val_accuracy": val_accuracy,
    "learning_rate": lr,
})

# Log final results
wandb.summary["test_accuracy"] = test_accuracy
wandb.summary["test_f1"] = test_f1
```

### 17.2 MLflow (Secondary - Local)

```python
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("financial-ts-scaling")

with mlflow.start_run(run_name=f"{exp_id}"):
    # Log params
    mlflow.log_params({
        "param_budget": param_budget,
        "dataset": dataset,
        **hyperparameters
    })
    
    # Log metrics (batch per epoch to reduce overhead)
    for epoch, metrics in enumerate(metrics_history):
        mlflow.log_metrics(metrics, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

### 17.3 What to Log

| Category | Items |
|----------|-------|
| Config | All hyperparameters, data paths, seeds |
| Training | Loss curves, LR schedule |
| Validation | All metrics per epoch |
| Test | Final metrics with confidence intervals |
| System | Batch size, GPU utilization (optional) |
| Artifacts | Best checkpoint, predictions sample |

---

# Part VII: System Operations Runbooks

## 18. Pre-Training Runbook

### 18.1 Environment Preparation

```markdown
## Pre-Training Checklist

[ ] 1. Fresh reboot
    - Close all applications
    - Restart machine
    - Wait for system to settle (2 min)

[ ] 2. Close background applications
    - Browser (all tabs)
    - Slack/Discord
    - Email client
    - Any heavy applications

[ ] 3. Disable interruptions
    - Enable Focus mode (macOS)
    - Disable automatic updates
    - Disable notifications

[ ] 4. Verify system state
    - Check disk space (need ~50GB free for checkpoints)
    - Verify power connected (not battery)
    - Check ambient temperature

[ ] 5. Position hardware
    - Laptop on hard, flat surface
    - Elevated for airflow (books, wire rack)
    - Small fan pointed at intake (if ambient > 60Â°F)

[ ] 6. Activate environment
    ```bash
    cd ~/projects/financial-ts-scaling
    source .venv/bin/activate
    ```

[ ] 7. Start monitoring
    ```bash
    # Terminal 1: htop
    htop -d 20
    
    # Terminal 2: Temperature monitoring
    ./scripts/thermal_monitor.sh &
    ```

[ ] 8. Verify GPU/MPS available
    ```python
    import torch
    assert torch.backends.mps.is_available()
    ```

[ ] 9. Run batch size discovery (if new architecture)
    ```bash
    python scripts/find_batch_size.py --param-budget 20M
    ```
```

---

## 19. During Training Runbook

### 19.1 Monitoring Protocol

```markdown
## During Training Monitoring

Every 30 minutes:
[ ] Check terminal for errors
[ ] Check htop for CPU/memory usage
[ ] Check temperature log

Warning Signs:
- CPU > 85Â°C sustained
- Memory > 90% used
- Training loss not decreasing after 10 epochs
- Validation loss increasing for 5+ epochs

If Warning:
1. Note the issue
2. Let current epoch complete
3. Save checkpoint
4. Address issue before continuing
```

### 19.2 Temperature Monitoring Script

```bash
#!/bin/bash
# scripts/thermal_monitor.sh

LOG_FILE="outputs/thermal_$(date +%Y%m%d_%H%M%S).log"
WARNING_TEMP=85
CRITICAL_TEMP=95

echo "Thermal monitoring started. Log: $LOG_FILE"

while true; do
    # Get temperature
    TEMP=$(sudo powermetrics --samplers smc -n 1 2>/dev/null | grep "CPU die temperature" | awk '{print $4}')
    
    if [ -n "$TEMP" ]; then
        TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
        echo "$TIMESTAMP: CPU ${TEMP}Â°C" >> "$LOG_FILE"
        
        # Check thresholds
        TEMP_INT=${TEMP%.*}
        if [ "$TEMP_INT" -ge "$CRITICAL_TEMP" ]; then
            echo "âš ï¸ CRITICAL: CPU temp ${TEMP}Â°C - consider pausing training"
        elif [ "$TEMP_INT" -ge "$WARNING_TEMP" ]; then
            echo "âš ï¸ WARNING: CPU temp ${TEMP}Â°C"
        fi
    fi
    
    sleep 60
done
```

---

## 20. Thermal Management Runbook

### 20.1 Temperature Thresholds (Apple Silicon)

| Temperature | Status | Action |
|-------------|--------|--------|
| < 70Â°C | Normal | None |
| 70-85Â°C | Acceptable | Monitor closely |
| 85-95Â°C | Warning | Consider reducing load |
| > 95Â°C | Critical | Pause immediately |

### 20.2 Mitigation Strategies

**If Temperature Sustained > 70Â°C:**
1. Verify airflow
   - Laptop elevated
   - Vents not blocked
   - Fan pointed at intake (if needed)
2. Check ambient temperature
   - Basement should be 50-60Â°F
   - Move to cooler location if needed
3. Monitor for stabilization

**If Temperature Sustained > 85Â°C:**
1. Reduce batch size by 50%
2. Add 1-second delay between batches
3. If still high: pause training
4. Let cool to < 60Â°C before resuming
5. Consider overnight training (cooler ambient)

**If Temperature Hits > 95Â°C:**
1. Training script should auto-pause
2. Save checkpoint immediately
3. Close all processes
4. Let machine cool for 30+ minutes
5. Investigate cause before resuming
6. Reduce batch size significantly

### 20.3 Automated Thermal Guard

```python
# src/utils/thermal.py
import subprocess
import time
import logging

logger = logging.getLogger(__name__)

def get_cpu_temp() -> float | None:
    """Get CPU temperature on macOS."""
    try:
        result = subprocess.run(
            ["sudo", "powermetrics", "--samplers", "smc", "-n", "1"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split('\n'):
            if 'CPU die temperature' in line:
                return float(line.split(':')[1].strip().replace(' C', ''))
    except Exception as e:
        logger.warning(f"Could not get temperature: {e}")
    return None

def thermal_check(
    max_temp: float = 85.0,
    critical_temp: float = 95.0,
    cooldown_time: int = 300
) -> bool:
    """Check temperature and pause if too hot. Returns True if paused."""
    temp = get_cpu_temp()
    
    if temp is None:
        return False
    
    if temp >= critical_temp:
        logger.critical(f"CPU temp {temp}Â°C >= {critical_temp}Â°C - CRITICAL")
        raise SystemExit("Critical temperature reached - stopping training")
    
    if temp >= max_temp:
        logger.warning(f"CPU temp {temp}Â°C >= {max_temp}Â°C - pausing {cooldown_time}s")
        time.sleep(cooldown_time)
        return True
    
    return False
```

### 20.4 Integration with Training Loop

```python
def train_epoch(model, dataloader, optimizer, epoch, check_interval=100):
    for batch_idx, batch in enumerate(dataloader):
        # Training step
        loss = train_step(model, batch, optimizer)
        
        # Thermal check every N batches
        if batch_idx % check_interval == 0:
            if thermal_check(max_temp=85.0):
                logger.info("Resumed after thermal pause")
        
        # Checkpoint if needed
        if should_checkpoint(batch_idx):
            save_checkpoint(model, optimizer, epoch, batch_idx)
```

---

## 21. Post-Training Runbook

```markdown
## Post-Training Checklist

[ ] 1. Verify completion
    - Check training finished successfully
    - Review final metrics

[ ] 2. Save artifacts
    - Final checkpoint saved
    - Metrics logged to W&B and MLflow
    - Optuna study saved

[ ] 3. Backup results
    - Copy checkpoints to external SSD
    - Verify backup integrity

[ ] 4. Document experiment
    - Record any anomalies
    - Note actual training time
    - Log batch size used

[ ] 5. Cool-down
    - Close training processes
    - Let machine idle for 10 min
    - Check temperature back to normal

[ ] 6. Review results
    - Check W&B dashboard
    - Compare to previous experiments
    - Note any issues for next run
```

---

## 22. Failure Recovery Runbook

### 22.1 Training Crash Recovery

```markdown
## If Training Crashes

1. Check error message
   - Out of memory: Reduce batch size
   - CUDA/MPS error: Restart Python process
   - NaN loss: Check data, reduce LR

2. Find latest checkpoint
   ```bash
   ls -la outputs/checkpoints/ | grep $EXP_ID
   ```

3. Resume from checkpoint
   ```bash
   python scripts/train.py --resume outputs/checkpoints/exp_XXX_epoch_YY.pt
   ```

4. If checkpoint corrupted:
   - Load previous checkpoint
   - If all corrupted: restart experiment

5. Log incident for debugging
```

### 22.2 System Crash Recovery

```markdown
## If System Crashes/Restarts

1. After reboot, check thermal log
   - Was temperature issue the cause?

2. Check disk space
   - Ensure checkpoints saved properly

3. Verify checkpoint integrity
   ```python
   checkpoint = torch.load("checkpoint.pt")
   print(checkpoint.keys())  # Should have expected keys
   ```

4. Resume training
   - Follow pre-training checklist
   - Resume from latest valid checkpoint
```

---

# Part VIII: Development Methodology

## 23. Agentic IDE Rules

### 23.1 Mandatory Stops

```markdown
The LLM assistant MUST STOP and wait for confirmation after:

- Every file modification
- Before any file deletion
- Before creating new files
- Before running tests
- Before running training scripts
- After any error
```

### 23.2 Branching Protocol

```markdown
## Git Workflow

1. Create feature branch before any change
   - Name: `feature/{task-id}-{short-description}`
   - Example: `feature/001-download-ohlcv`

2. Never commit directly to main

3. One logical change per commit
   - "Logical change" = single function, single test, single config
   - If touching 3+ functions, split into multiple commits

4. Pull request required for merge (even self-review)
```

### 23.3 Test-First Flow

```markdown
## TDD Workflow

1. Write failing test
2. STOP â€” confirm test failure with user
3. Write minimal code to pass
4. STOP â€” confirm test passes
5. Refactor if needed
6. STOP â€” confirm tests still pass
7. Commit
```

### 23.4 Error Recovery

```markdown
## On Any Error

1. STOP immediately
2. Do NOT auto-fix
3. Present error and proposed fix
4. Wait for user confirmation
5. Apply fix
6. STOP â€” verify fix worked
```

### 23.5 Forbidden Actions

```markdown
## Never Do

- Bulk refactors without explicit plan approval
- "While we're here, let's also..." scope creep
- Assumptions about intent â€” ask if unclear
- Auto-running code without confirmation
- Modifying multiple files in one action
- Deleting files without explicit request
```

---

## 24. Code Organization Rules

### 24.1 File Size Limits

- No file > 300 lines
- Functions < 50 lines
- Classes < 200 lines

### 24.2 Required Elements

- Type hints on all functions
- Docstrings on all public functions
- Unit tests for all functions
- No magic numbers (use constants)
- No hardcoded paths (use config)

### 24.3 Module Structure

```python
# Each module should have:
# 1. Module docstring
# 2. Imports (stdlib, third-party, local)
# 3. Constants
# 4. Type definitions
# 5. Functions/Classes
# 6. Main block (if script)

"""Module docstring describing purpose."""

# Standard library
import os
from pathlib import Path

# Third party
import pandas as pd
import torch

# Local
from src.utils.logging import get_logger

# Constants
DEFAULT_BATCH_SIZE = 32
MAX_EPOCHS = 100

# Types
PathLike = str | Path

# Functions
def my_function(arg: int) -> str:
    """Docstring describing function."""
    pass

# Main
if __name__ == "__main__":
    pass
```

---

# Part IX: Scripts Inventory

## 25. Data Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `download_ohlcv.py` | Fetch price data from Yahoo | Symbol, dates | Parquet file |
| `download_fred.py` | Fetch economic data from FRED | Series ID | Parquet file |
| `download_sentiment.py` | Fetch SF Fed sentiment | None | Parquet file |
| `download_google_trends.py` | Fetch search trends | Keywords | Parquet file |
| `validate_data.py` | Check data quality | Parquet files | Report |

## 26. Feature Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `calculate_indicators.py` | Compute technical indicators | OHLCV parquet | Features parquet |
| `resample_timeframes.py` | Create weekly/monthly bars | Daily parquet | Multi-TF parquet |
| `add_multi_timeframe.py` | Add higher TF features to daily | Daily + Weekly + Monthly | Combined parquet |
| `build_feature_matrix.py` | Assemble all features | Multiple parquets | Final matrix |
| `normalize_features.py` | Z-score normalization | Features parquet | Normalized parquet |

## 27. Training Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `train.py` | Main training entry | Config, data | Checkpoints, logs |
| `hpo_sweep.py` | Optuna HPO | Config | Best params, study |
| `resume_training.py` | Resume from checkpoint | Checkpoint path | Continued training |
| `find_batch_size.py` | Discover max batch size | Model config | Batch size |

## 28. Evaluation Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `evaluate.py` | Compute all metrics | Model, test data | Metrics JSON |
| `plot_scaling_curves.py` | Generate scaling plots | Results | Figures |
| `calibration_analysis.py` | Analyze probability calibration | Predictions | Calibration report |
| `generate_report.py` | Create summary report | All results | Markdown report |

## 29. Utility Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `thermal_monitor.sh` | Monitor CPU temperature | None | Log file |
| `backup_data.sh` | Backup to external drive | None | Backup |
| `validate_checkpoint.py` | Check checkpoint integrity | Checkpoint path | Pass/fail |
| `create_csv_sample.py` | Create LLM-readable sample | Parquet | Small CSV |

---

# Part X: Action Items Checklist

## 30. One-Time Setup

```markdown
[ ] Create GitHub repo
[ ] Set up virtual environment (Python 3.12)
[ ] Install all libraries (create requirements.txt)
[ ] Verify PyTorch MPS works
[ ] Create W&B account, get API key
[ ] Create MLflow local tracking
[ ] Get FRED API key
[ ] Set up Cursor/VS Code with rules
[ ] Create project directory structure
[ ] Install htop, configure powermetrics
[ ] Acquire/setup external SSD for backups
```

## 31. Data Acquisition

```markdown
[ ] Download all OHLCV data (SPY, DIA, QQQ, major stocks)
[ ] Download all FRED data (yields, rates)
[ ] Download VIX data
[ ] Download SF Fed sentiment data
[ ] Download Google Trends data (if using)
[ ] Validate and clean all data
[ ] Create backup to external SSD
[ ] Generate checksums
```

## 32. Pipeline Development

```markdown
[ ] Build download scripts with validation
[ ] Build feature calculation pipeline (all indicator tiers)
[ ] Build multi-timeframe aggregation
[ ] Build normalization pipeline
[ ] Build dataset/dataloader
[ ] Verify shapes and dtypes correct
[ ] Create small CSV samples for debugging
```

## 33. Model Development

```markdown
[ ] Implement PatchTST wrapper with configurable params
[ ] Implement classification head (sigmoid)
[ ] Implement regression head (linear)
[ ] Implement training loop with checkpointing
[ ] Implement thermal monitoring integration
[ ] Implement evaluation metrics
[ ] Verify toy training works end-to-end
```

## 34. HPO Development

```markdown
[ ] Integrate Optuna
[ ] Define search space per parameter budget
[ ] Implement param count constraint
[ ] Implement pruning
[ ] Test HPO on small run
[ ] Integrate with W&B and MLflow
```

## 35. Experiment Execution

```markdown
[ ] Run batch size discovery for each param budget
[ ] Run Phase 1 experiments (parameter scaling)
[ ] Analyze Phase 1 results
[ ] Run Phase 2 experiments (feature scaling)
[ ] Analyze Phase 2 results
[ ] Run Phase 3 experiments (data scaling)
[ ] Analyze Phase 3 results
[ ] Run Phase 4 experiments (quality scaling)
[ ] Analyze Phase 4 results
[ ] Run Phase 5 experiments (interactions)
[ ] Analyze Phase 5 results
[ ] Determine best configuration
[ ] Run Phase 6 experiments (full expansion)
[ ] Aggregate all results
```

## 36. Analysis & Publication

```markdown
[ ] Generate scaling curves for all phases
[ ] Fit power law models
[ ] Create interaction plots
[ ] Statistical significance tests
[ ] Draft paper/article
[ ] Create visualizations
[ ] Review and edit
[ ] Publish to Medium
[ ] Publish to arXiv/SSRN (optional)
[ ] Release code on GitHub
```

---

# Part XI: Risk & Contingency

## 37. Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Thermal throttling | Medium | Medium | Batch reduction, overnight runs, basement cooling |
| Training instability | Medium | Medium | Gradient clipping, LR reduction, checkpoint resume |
| Data issues | Low | High | Validation scripts, backup raw data |
| Overfitting | Medium | Medium | Early stopping, validation monitoring, regularization |
| No scaling effect found | Medium | Medium | Still publishable as null result |
| Disk space exhaustion | Low | Medium | Monitor usage, prune old checkpoints |
| Hardware failure | Low | High | Cloud backups, checkpoint redundancy |

## 38. Contingency Plans

**If no scaling law observed:**
- Document as null result
- Analyze why (data quality? task difficulty? architecture limit?)
- Compare to published benchmarks
- Still publishable and valuable

**If training keeps failing:**
- Reduce model size temporarily
- Simplify task (binary classification only)
- Debug on smallest dataset first
- Consider cloud GPU for verification

**If results inconsistent:**
- Increase number of seeds per experiment
- Use ensemble of HPO trials
- Check for data leakage
- Verify train/val/test splits

---

# Part XII: Appendices

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| OHLCV | Open, High, Low, Close, Volume |
| PatchTST | Patch Time Series Transformer |
| HPO | Hyperparameter Optimization |
| MPS | Metal Performance Shaders (Apple GPU) |
| BCE | Binary Cross-Entropy loss |
| MSE | Mean Squared Error |
| Parquet | Columnar binary data format |
| Conformal Prediction | Method for uncertainty quantification |

## Appendix B: Key Formulas

**Scaling Law (expected form):**
```
Error âˆ N^(-Î±)
```
Where N = parameter count, Î± = scaling exponent

**Directional Accuracy:**
```
Acc = (Correct Up + Correct Down) / Total Predictions
```

**Brier Score:**
```
BS = (1/N) Ã— Î£(p_i - o_i)Â²
```
Where p_i = predicted probability, o_i = actual outcome (0 or 1)

## Appendix C: Reference Links

- PatchTST Paper: https://arxiv.org/abs/2211.14730
- Hugging Face PatchTST: https://huggingface.co/docs/transformers/model_doc/patchtst
- Optuna: https://optuna.org/
- Weights & Biases: https://wandb.ai/
- MLflow: https://mlflow.org/
- FRED API: https://fred.stlouisfed.org/docs/api/fred/
- SF Fed Sentiment: https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/