"""
Centralized Configuration

All configuration managed through pydantic-settings.
NO hardcoded values in code.

Created: Phase 1 - Foundation & Safety Net
"""

from .settings import settings, Settings

__all__ = ["settings", "Settings"]
