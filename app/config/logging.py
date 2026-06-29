import os
import logging
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler


# ── Log directory ─────────────────────────────────────────────────────────────

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "intelics.log")

os.makedirs(LOG_DIR, exist_ok=True)


# ── Format ────────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s  %(levelname)-8s [%(name)s]  %(stream_sid)s  %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ── Context variable for stream_sid ─────────────────────────────────────────

stream_sid_var: ContextVar[str] = ContextVar("stream_sid", default="-")


# ── Custom logging filter to add stream_sid to log records ─────────────────

class StreamSidFilter(logging.Filter):
    def filter(self, record):
        record.stream_sid = stream_sid_var.get()
        return True

# ── Root logger setup ─────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("intelics_bot")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Avoid adding duplicate handlers if module is reloaded
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console handler — INFO and above ──────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File handler — DEBUG and above, rotates at 5MB, keeps 7 backups ──────
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes    = 5 * 1024 * 1024,   # 5MB per file
        backupCount = 7,                  # keep last 7 rotated files
        encoding    = "utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler.addFilter(StreamSidFilter())
    file_handler.addFilter(StreamSidFilter())
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


_root = _setup_logging()


# ── Get logger by name ────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    return _root.getChild(name)
