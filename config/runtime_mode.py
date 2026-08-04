"""Explicit runtime deployment modes."""

from enum import Enum


class RuntimeMode(str, Enum):
    """Storage and isolation policy for one runtime instance."""

    TEST = "test"
    BENCHMARK = "benchmark"
    PRODUCTION = "production"
