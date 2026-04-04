"""Configuration loader for Ari.

Reads ``config/default.yaml`` for base settings, then merges any overrides
found in ``config/local.yaml`` (git-ignored, machine-specific tweaks).

Usage::

    from ari.config import cfg
    print(cfg["tts"]["model"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, MutableMapping

import yaml

# Root of the ari-assistant repo / install directory.
_PROJECT_ROOT = Path("/home/arun/ari-assistant")
_DEFAULT_PATH = _PROJECT_ROOT / "config" / "default.yaml"
_LOCAL_PATH = _PROJECT_ROOT / "config" / "local.yaml"


def _expand_paths(obj: Any) -> Any:
    """Recursively expand ``~`` in any string value that looks like a path."""
    if isinstance(obj, str) and "~" in obj:
        return os.path.expanduser(obj)
    if isinstance(obj, dict):
        return {k: _expand_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_paths(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Return *base* dict recursively updated with *override* values.

    Lists and scalars in *override* replace those in *base*; nested dicts are
    merged so that only the keys present in *override* are changed.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class _Config(MutableMapping):
    """Thin dict-like wrapper around the merged YAML configuration.

    Supports ``cfg["key"]``, ``cfg.get("key", default)``, iteration, and
    ``"key" in cfg``.  The underlying data is a plain ``dict`` accessible
    via ``cfg.data`` if needed.
    """

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    # -- MutableMapping interface ------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def __repr__(self) -> str:
        return f"_Config({self.data!r})"

    # -- Loading -----------------------------------------------------------

    def load(
        self,
        default_path: Path | str = _DEFAULT_PATH,
        local_path: Path | str = _LOCAL_PATH,
    ) -> None:
        """(Re)load configuration from YAML files.

        Parameters
        ----------
        default_path:
            Path to the base config YAML (must exist).
        local_path:
            Path to an optional override YAML (silently skipped if missing).
        """
        default_path = Path(default_path)
        local_path = Path(local_path)

        if not default_path.exists():
            raise FileNotFoundError(
                f"Default config not found: {default_path}"
            )

        with open(default_path, "r") as fh:
            base: dict = yaml.safe_load(fh) or {}

        overrides: dict = {}
        if local_path.exists():
            with open(local_path, "r") as fh:
                overrides = yaml.safe_load(fh) or {}

        self.data = _expand_paths(_deep_merge(base, overrides))


# Global singleton -- import and use directly.
cfg = _Config()

# Auto-load on import so ``from ari.config import cfg`` is immediately usable.
# If default.yaml doesn't exist yet (e.g. during initial setup), we silently
# start with an empty config rather than crashing on import.
try:
    cfg.load()
except FileNotFoundError:
    pass
