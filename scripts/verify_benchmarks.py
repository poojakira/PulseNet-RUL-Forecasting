"""Compatibility wrapper for official NASA FD001 validation.

This script intentionally does not use generated benchmark fixtures. It delegates
to `scripts/run_validation.py`, which verifies the checked-in NASA C-MAPSS
archive and writes measured JSON/SVG evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.run_validation import main  # noqa: E402

if __name__ == "__main__":
    main()
