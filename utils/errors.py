"""Custom exceptions used across the analytics pipeline."""

from __future__ import annotations


class AnalyticsError(Exception):
    """Base class for recoverable analytics errors."""


class CacheMissError(AnalyticsError):
    """Raised when an expected cache entry is unavailable."""


class DatabaseOperationError(AnalyticsError):
    """Raised when persistence or connection checks fail."""


class InsightGenerationError(AnalyticsError):
    """Raised when a cluster cannot be summarized into an insight."""
