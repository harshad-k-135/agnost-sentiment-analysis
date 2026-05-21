"""Pytest configuration for local test execution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db import create_schema


@pytest.fixture(autouse=True, scope="session")
def initialize_schema() -> None:
    """Create tables before tests run so integration tests have a live schema."""

    create_schema()
