"""CueIt: open-source automatic cue-point detection for DJs."""

import os

__version__ = "0.2.0"

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")


def asset_path(name: str) -> str:
    """Absolute path to a bundled asset (icon, etc.)."""
    return os.path.join(_ASSETS, name)
