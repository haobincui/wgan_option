import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO, Union


DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_FMT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"


@dataclass(frozen=True)
class LoggingConfig:
    level: Union[int, str] = "INFO"
    fmt: str = DEFAULT_FMT
    datefmt: str = DEFAULT_DATEFMT
    use_utc: bool = False
    stream: Optional[TextIO] = sys.stdout
    log_file: Optional[Union[str, Path]] = None
    file_level: Optional[Union[int, str]] = None
    force: bool = False


def _coerce_level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    lvl = str(level).strip().upper()
    if not hasattr(logging, lvl):
        raise ValueError(f"Unknown log level: {level!r}")
    return int(getattr(logging, lvl))


def setup_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Configure the root logger for consistent runtime formatting.

    - Idempotent by default: if handlers already exist, no changes are applied unless force=True.
    - Supports both stream output (stdout by default) and optional file output.
    """
    cfg = config or LoggingConfig()
    root = logging.getLogger()

    if root.handlers and not cfg.force:
        # Respect existing logging setup (e.g., notebooks, other runners).
        return root

    if cfg.force:
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    root.setLevel(_coerce_level(cfg.level))

    formatter = logging.Formatter(fmt=cfg.fmt, datefmt=cfg.datefmt)
    if cfg.use_utc:
        formatter.converter = time.gmtime  # type: ignore[attr-defined]

    if cfg.stream is not None:
        sh = logging.StreamHandler(cfg.stream)
        sh.setLevel(_coerce_level(cfg.level))
        sh.setFormatter(formatter)
        root.addHandler(sh)

    if cfg.log_file is not None:
        log_path = Path(cfg.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh_level = cfg.file_level if cfg.file_level is not None else cfg.level
        fh.setLevel(_coerce_level(fh_level))
        fh.setFormatter(formatter)
        root.addHandler(fh)

    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Wrapper for logging.getLogger to keep imports consistent across the repo.
    """
    return logging.getLogger(name)
