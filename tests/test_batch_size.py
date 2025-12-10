"""Tests for batch size discovery algorithm.

Tests the find_batch_size function which discovers the largest viable
batch size for a given model configuration via exponential search.
"""

from __future__ import annotations

import pytest


# Import will fail until implementation exists - this is expected in RED phase
try:
    from scripts.find_batch_size import find_batch_size
except ImportError:
    # Define stub for test collection to work
    def find_batch_size(*args, **kwargs):
        raise NotImplementedError("find_batch_size not yet implemented")


class TestFindBatchSizeReturnsPowerOfTwo:
    """Test that find_batch_size returns power-of-two values."""

    def test_result_is_power_of_two(self):
        """Result should always be a power of 2."""
        # Mock that always succeeds - should return max
        result = find_batch_size(
            try_batch_fn=lambda bs: True,
            min_batch=8,
            max_batch=64,
        )
        assert result == 64
        # Power of 2 check: n & (n-1) == 0 for powers of 2
        assert result & (result - 1) == 0, f"{result} is not a power of 2"

    def test_result_in_valid_range(self):
        """Result should be between min_batch and max_batch."""
        result = find_batch_size(
            try_batch_fn=lambda bs: True,
            min_batch=16,
            max_batch=128,
        )
        assert 16 <= result <= 128


class TestFindBatchSizeFindsLargestViable:
    """Test that algorithm finds the largest working batch size."""

    def test_finds_largest_before_oom(self):
        """Should return last successful size before OOM."""
        def try_batch(bs: int) -> bool:
            if bs >= 64:
                raise RuntimeError("MPS backend out of memory")
            return True

        result = find_batch_size(
            try_batch_fn=try_batch,
            min_batch=8,
            max_batch=512,
        )
        # 8 -> success, 16 -> success, 32 -> success, 64 -> OOM
        # Should return 32
        assert result == 32

    def test_finds_largest_at_higher_threshold(self):
        """Should work correctly with different OOM thresholds."""
        def try_batch(bs: int) -> bool:
            if bs >= 256:
                raise RuntimeError("MPS backend out of memory")
            return True

        result = find_batch_size(
            try_batch_fn=try_batch,
            min_batch=8,
            max_batch=512,
        )
        # 8, 16, 32, 64, 128 -> success, 256 -> OOM
        # Should return 128
        assert result == 128


class TestFindBatchSizeEdgeCases:
    """Test edge cases for batch size discovery."""

    def test_returns_minimum_on_immediate_oom(self):
        """If OOM at first size > min, return min."""
        def try_batch(bs: int) -> bool:
            if bs > 8:
                raise RuntimeError("MPS backend out of memory")
            return True

        result = find_batch_size(
            try_batch_fn=try_batch,
            min_batch=8,
            max_batch=512,
        )
        # 8 -> success, 16 -> OOM
        # Should return 8
        assert result == 8

    def test_returns_max_when_all_succeed(self):
        """If all sizes work, return max_batch."""
        result = find_batch_size(
            try_batch_fn=lambda bs: True,
            min_batch=8,
            max_batch=256,
        )
        assert result == 256

    def test_returns_minimum_when_minimum_fails(self):
        """If even minimum fails, still return minimum (best effort)."""
        def try_batch(bs: int) -> bool:
            raise RuntimeError("MPS backend out of memory")

        result = find_batch_size(
            try_batch_fn=try_batch,
            min_batch=8,
            max_batch=512,
        )
        # Even 8 fails, but we return 8 as minimum viable attempt
        assert result == 8


class TestFindBatchSizeAlgorithm:
    """Test the doubling algorithm behavior."""

    def test_doubles_batch_size_correctly(self):
        """Algorithm should double batch size each iteration."""
        tried_sizes = []

        def try_batch(bs: int) -> bool:
            tried_sizes.append(bs)
            if bs >= 64:
                raise RuntimeError("OOM")
            return True

        find_batch_size(
            try_batch_fn=try_batch,
            min_batch=8,
            max_batch=512,
        )

        # Should try: 8, 16, 32, 64 (then stop on OOM)
        assert tried_sizes == [8, 16, 32, 64]

    def test_stops_at_max_batch(self):
        """Should not try sizes beyond max_batch."""
        tried_sizes = []

        def try_batch(bs: int) -> bool:
            tried_sizes.append(bs)
            return True

        find_batch_size(
            try_batch_fn=try_batch,
            min_batch=8,
            max_batch=32,
        )

        # Should try: 8, 16, 32 (then stop at max)
        assert tried_sizes == [8, 16, 32]
        assert max(tried_sizes) == 32


class TestFindBatchSizeErrorHandling:
    """Test error handling in batch size discovery."""

    def test_catches_runtime_error(self):
        """Should catch RuntimeError as OOM signal."""
        def try_batch(bs: int) -> bool:
            if bs >= 32:
                raise RuntimeError("CUDA out of memory" if bs == 32 else "MPS error")
            return True

        # Should not raise, should return last successful
        result = find_batch_size(
            try_batch_fn=try_batch,
            min_batch=8,
            max_batch=512,
        )
        assert result == 16

    def test_propagates_non_oom_errors(self):
        """Non-RuntimeError exceptions should propagate."""
        def try_batch(bs: int) -> bool:
            raise ValueError("Invalid configuration")

        with pytest.raises(ValueError, match="Invalid configuration"):
            find_batch_size(
                try_batch_fn=try_batch,
                min_batch=8,
                max_batch=512,
            )
