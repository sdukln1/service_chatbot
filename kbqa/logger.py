from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "log"

_configured = False


def _setup() -> None:
    global _configured
    if _configured:
        return
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"{datetime.now():%Y-%m-%d}.log"

    root = logging.getLogger("kbqa")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"
    )
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _setup()
    return logging.getLogger(f"kbqa.{name}")
