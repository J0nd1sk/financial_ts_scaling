"""Foundation models for time-series forecasting.

This module provides wrappers for pre-trained foundation models:
- Lag-Llama: Probabilistic forecasting with decoder-only architecture
- TimesFM (deferred): Google's time-series foundation model
"""

from src.models.foundation.base import FoundationModel
from src.models.foundation.lag_llama import LagLlamaWrapper, distribution_to_threshold_prob

__all__ = ["FoundationModel", "LagLlamaWrapper", "distribution_to_threshold_prob"]
