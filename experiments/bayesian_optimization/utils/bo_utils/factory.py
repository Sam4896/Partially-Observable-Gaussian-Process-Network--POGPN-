"""Factories to provide concrete implementations of interfaces.

These factories centralize construction and keep orchestration code clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import naming as naming_impl
from . import tags as tags_impl
from . import metrics as metrics_impl

if TYPE_CHECKING:
    from .types import Namer, TagsBuilder, MetricsComputer


def get_namer() -> "Namer":
    """Return a Namer implementation."""
    return naming_impl  # module fulfills the Namer Protocol


def get_tags_builder() -> "TagsBuilder":
    """Return a TagsBuilder implementation."""
    return tags_impl  # module fulfills the TagsBuilder Protocol


def get_metrics_computer() -> "MetricsComputer":
    """Return a MetricsComputer implementation."""
    return metrics_impl  # module fulfills the MetricsComputer Protocol
