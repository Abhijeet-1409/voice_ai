import logging
import os
from logging.handlers import RotatingFileHandler

# ── Log directory ─────────────────────────────────────────────────────────────

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "intelics.log")

os.makedirs(LOG_DIR, exist_ok=True)


# ── Format ────────────────────────────────────────────────────────────────────

LOG_FORMAT  = "%(asctime)s  %(levelname)-8s [%(name)s]  %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ── Root logger setup ─────────────────────────────────────────────────────────

def _build_root_logger() -> logging.Logger:
    logger = logging.getLogger("intelics")
    logger.setLevel(logging.DEBUG)

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

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


_root = _build_root_logger()


# ── Named child loggers — one per service/util ────────────────────────────────

audio_logger   = _root.getChild("audio")
stt_logger     = _root.getChild("stt")
tts_logger     = _root.getChild("tts")
rag_logger     = _root.getChild("rag")
llm_logger     = _root.getChild("llm")
ws_logger      = _root.getChild("ws")
session_logger = _root.getChild("session")
email_logger   = _root.getChild("email")
db_logger      = _root.getChild("db")