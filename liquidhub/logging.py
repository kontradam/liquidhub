import logging
import sys

import colorlog

from liquidhub.config import settings

_configured = False


def setup_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return

    lvl = (level or settings.log_level).upper()

    handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s "
            "%(cyan)s%(name)s%(reset)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(lvl)

    for noisy in ("urllib3", "ccxt", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
