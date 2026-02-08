"""Feature curation modules for quality analysis and reduction.

This package provides tools for:
- Feature importance analysis (multiple methods)
- Feature quality assessment (pass/fail thresholds)
- Model performance validation during reduction
- Iterative feature reduction with category preservation
"""

from src.features.curation import importance
from src.features.curation import quality
from src.features.curation import validation
from src.features.curation import reduction

__all__ = ["importance", "quality", "validation", "reduction"]
