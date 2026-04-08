"""Ensure project root and src/ are on sys.path.

Import this module before importing from wgan_option, quantlib, or market_data::

    import scripts._path_setup  # noqa: F401
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
