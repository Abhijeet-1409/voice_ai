# worker/rag/ingest.py
import pandas as pd
from pathlib import Path

from shared.logging_setup import get_logger

from config import get_worker_settings


_LOGGER = "worker.rag.extraction"
logger = get_logger(_LOGGER)


settings = get_worker_settings()
RATE_CARD_PATH = Path(settings.DATA_DIR)

SHEET_CONFIG = {
    "Compute - Linux"   : {"header": 2, "price_col": "OD/Hr"},
    "Compute - Windows" : {"header": 2, "price_col": "OD/Hr"},
    "Operating Systems" : {"header": 2, "price_col": "Vcpu/hr"},
    "Networking"        : {"header": 4, "price_col": "Price/hr"},
}


def _row_to_text(row: pd.Series, sheet_name: str) -> str:
    parts = [f"[{sheet_name}]"]
    for col, val in row.items():
        if str(val).strip():
            parts.append(f"{col}: {val}")
    return " | ".join(parts)


def _sheet_to_texts(df: pd.DataFrame, sheet_name: str) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        text_val = _row_to_text(row, sheet_name)
        if text_val.strip():
            texts.append(text_val)
    return texts


def _build_storage_texts() -> list[str]:
    texts = []
    try:
        df1 = pd.read_excel(RATE_CARD_PATH, sheet_name="Storage", header=6, nrows=12)
        if "Price/Hr" in df1.columns:
            df1 = df1[df1["Price/Hr"].notna() & (df1["Price/Hr"] != "")].fillna("")
            texts.extend(_sheet_to_texts(df1, "Storage - Block"))

        df2 = pd.read_excel(RATE_CARD_PATH, sheet_name="Storage", header=22)
        if "Price /Hr" in df2.columns:
            df2 = df2[df2["Price /Hr"].notna() & (df2["Price /Hr"] != "")].fillna("")
            texts.extend(_sheet_to_texts(df2, "Storage - Object"))
    except Exception as e:
        logger.error(f"Failed to load Storage sheet — {e}")
    return texts


def _build_backup_texts() -> list[str]:
    texts = []
    try:
        df1 = pd.read_excel(RATE_CARD_PATH, sheet_name="Backup", header=3, nrows=2)
        if "Monthly  INR per  GB" in df1.columns:
            df1 = df1[df1["Monthly  INR per  GB"].notna() & (df1["Monthly  INR per  GB"] != "")].fillna("")
            texts.extend(_sheet_to_texts(df1, "Backup - Capacity"))

        df2 = pd.read_excel(RATE_CARD_PATH, sheet_name="Backup", header=7)
        if "Monthly  for protected VM" in df2.columns:
            df2 = df2[df2["Monthly  for protected VM"].notna() & (df2["Monthly  for protected VM"] != "")].fillna("")
            texts.extend(_sheet_to_texts(df2, "Backup - Advanced"))
    except Exception as e:
        logger.error(f"Failed to load Backup sheet — {e}")
    return texts


def build_all_texts() -> list[str]:
    if not Path(RATE_CARD_PATH).exists():
        logger.warning(f"Rate card not found at {RATE_CARD_PATH}")
        return []

    texts: list[str] = []

    for sheet_name, config in SHEET_CONFIG.items():
        try:
            df = pd.read_excel(RATE_CARD_PATH, sheet_name=sheet_name, header=config["header"])
            price_col = config["price_col"]
            if price_col in df.columns:
                df = df[df[price_col].notna() & (df[price_col] != "")]
            df = df.fillna("")
            texts.extend(_sheet_to_texts(df, sheet_name))
        except Exception as e:
            logger.error(f"Failed to load sheet '{sheet_name}' — {e}")

    texts.extend(_build_backup_texts())
    texts.extend(_build_storage_texts())

    logger.info(f"Built {len(texts)} text chunks from rate card.")
    return texts