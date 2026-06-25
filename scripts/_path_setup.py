"""Ensure project root and src/ are on sys.path.

Import this module before importing from wgan_option, quantlib, or market_data::

    import scripts._path_setup  # noqa: F401
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

for path in (str(ROOT_DIR), str(SRC_DIR)):
    while path in sys.path:
        sys.path.remove(path)

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))
