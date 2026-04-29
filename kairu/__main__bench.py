"""Re-export shim so ``from kairu.__main__bench import main, build_parser``
works in tests.  All logic lives in ``kairu.bench``.
"""
from __future__ import annotations

from kairu.bench import build_parser, main, _collect_hardware  # noqa: F401

__all__ = ["build_parser", "main", "_collect_hardware"]
