"""Foundation models for time-series forecasting.

This module provides wrappers for pre-trained foundation models:
- Lag-Llama: Probabilistic forecasting with decoder-only architecture
- TimesFM (deferred): Google's time-series foundation model
"""

from src.models.foundation.base import FoundationModel

__all__ = ["FoundationModel"]
