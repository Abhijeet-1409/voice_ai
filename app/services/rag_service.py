from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import rag_logger

# ── Paths ─────────────────────────────────────────────────────────────────────

RATE_CARD_PATH  = "/app/data/rate_card.xlsx"
RAG_MODEL_PATH  = "/app/rag_models/all-MiniLM-L6-v2"
TOP_N           = 5     # number of chunks to return per query


# ── Category keyword map ──────────────────────────────────────────────────────
# Maps category tag → list of trigger words found in customer transcript.
# Keyword filter checks chunk categories tag (not chunk text).

CATEGORY_MAP = {
    "linux"           : ["linux", "ubuntu", "centos", "rocky", "alma"],
    "windows"         : ["windows", "win", "microsoft"],
    "storage"         : ["storage", "disk", "ssd", "object"],
    "backup"          : ["backup", "acronis", "snapshot"],
    "networking"      : ["network", "vpc", "firewall", "ip", "dns", "ssl"],
    "memory_intensive": ["database", "db", "mysql", "postgres", "mongo", "cache"],
    "cpu_intensive"   : ["render", "encode", "ml", "ai", "processing"],
    "general_purpose" : ["website", "web", "app", "startup", "small", "cheap"],
}

SHEET_CONFIG = {
    "Compute - Linux"   : {"header": 2, "price_col": "OD/Hr"},
    "Compute - Windows" : {"header": 2, "price_col": "OD/Hr"},
    "Operating Systems" : {"header": 2, "price_col": "Vcpu/hr"},
    "Networking"        : {"header": 4, "price_col": "Price/hr"},
}


# ── Model and chunks load ONCE at import time ─────────────────────────────────

rag_logger.info("Loading embedding model...")

_model = SentenceTransformer(RAG_MODEL_PATH)

rag_logger.info("Reading rate card...")

_chunks: list[dict] = []

pd.set_option('future.no_silent_downcasting', True)

def _load_storage_sheet() -> None:
    """
    Storage sheet has multiple sub-tables each with their own header.
    We read the known sub-tables individually using header + nrows.

    From the raw Excel inspection:
      Block Storage  : header=6,  price_col='Price/Hr'
      Object Storage : header=22, price_col='Price /Hr'  (note the space)
    Adjust header row numbers if your Excel layout differs.
    """
    try:
        # Block Storage
        df1 = pd.read_excel(
            RATE_CARD_PATH,
            sheet_name = "Storage",
            header     = 6,
            nrows      = 12,
        )
        price_col_1 = "Price/Hr"
        if price_col_1 in df1.columns:
            df1 = df1[df1[price_col_1].notna() & (df1[price_col_1] != "")]
            df1 = df1.fillna("")
            chunks1 = _build_chunks(df1, "Storage - Block")
            _chunks.extend(chunks1)
            rag_logger.debug(f"Storage block — {len(chunks1)} chunks")

        # Object Storage
        df2 = pd.read_excel(
            RATE_CARD_PATH,
            sheet_name = "Storage",
            header     = 22,
        )
        price_col_2 = "Price /Hr"
        if price_col_2 in df2.columns:
            df2 = df2[df2[price_col_2].notna() & (df2[price_col_2] != "")]
            df2 = df2.fillna("")
            chunks2 = _build_chunks(df2, "Storage - Object")
            _chunks.extend(chunks2)
            rag_logger.debug(f"Storage object — {len(chunks2)} chunks")

    except Exception as e:
        rag_logger.error(f"Failed to load Storage sheet — {e}")


def _load_rate_card() -> None:
    if not Path(RATE_CARD_PATH).exists():
        rag_logger.warning(f"Rate card not found at {RATE_CARD_PATH}")
        return

    # ── Standard sheets ───────────────────────────────────────────────────────
    for sheet_name, config in SHEET_CONFIG.items():
        df = pd.read_excel(
            RATE_CARD_PATH,
            sheet_name = sheet_name,
            header     = config["header"],
        )

        # Drop rows where price column is empty — removes titles and section labels
        price_col = config["price_col"]
        if price_col in df.columns:
            df = df[df[price_col].notna() & (df[price_col] != "")]
        else:
            rag_logger.warning(f"Price column '{price_col}' not found in sheet '{sheet_name}'")

        df = df.fillna("")
        sheet_chunks = _build_chunks(df, sheet_name)
        _chunks.extend(sheet_chunks)
        rag_logger.debug(f"Sheet '{sheet_name}' — {len(sheet_chunks)} pricing chunks")

    # ── Backup sheet (two separate tables, different headers) ─────────────────
    _load_backup_sheet()
    _load_storage_sheet()

    rag_logger.info(f"Built {len(_chunks)} chunks from {len(SHEET_CONFIG) + 1} sheets.")

    # ── Embed all chunks in one batch ─────────────────────────────────────────
    texts   = [c["text"] for c in _chunks]
    vectors = _model.encode(texts, show_progress_bar=False)

    for i, chunk in enumerate(_chunks):
        chunk["vector"] = vectors[i]

    rag_logger.info("All chunks embedded and ready.")


def _load_backup_sheet() -> None:
    """
    Backup sheet has two separate tables each with their own header row.
    Table 1 — Capacity based: header row 3, price col 'Monthly  INR per  GB'
    Table 2 — Advanced WL:    header row 7, price col 'Monthly  for protected VM'
    We read them separately and chunk both.
    """
    try:
        # Table 1 — Acronis Standard Capacity based
        df1 = pd.read_excel(
            RATE_CARD_PATH,
            sheet_name = "Backup",
            header     = 3,
            nrows      = 2,
        )
        price_col_1 = "Monthly  INR per  GB"
        if price_col_1 in df1.columns:
            df1 = df1[df1[price_col_1].notna() & (df1[price_col_1] != "")]
            df1 = df1.fillna("")
            chunks1 = _build_chunks(df1, "Backup - Capacity")
            _chunks.extend(chunks1)
            rag_logger.debug(f"Backup table 1 — {len(chunks1)} chunks")

        # Table 2 — Acronis Advanced WL based
        df2 = pd.read_excel(
            RATE_CARD_PATH,
            sheet_name = "Backup",
            header     = 7,
        )
        price_col_2 = "Monthly  for protected VM"
        if price_col_2 in df2.columns:
            df2 = df2[df2[price_col_2].notna() & (df2[price_col_2] != "")]
            df2 = df2.fillna("")
            chunks2 = _build_chunks(df2, "Backup - Advanced")
            _chunks.extend(chunks2)
            rag_logger.debug(f"Backup table 2 — {len(chunks2)} chunks")

    except Exception as e:
        rag_logger.error(f"Failed to load Backup sheet — {e}")


def _build_chunks(df: pd.DataFrame, sheet_name: str) -> list[dict]:
    """
    Convert each row of a sheet into a plain text chunk with metadata.
    Categories are assigned based on sheet name and any section column.
    """
    chunks = []
    base_categories = _sheet_to_categories(sheet_name)

    for _, row in df.iterrows():
        text = _row_to_text(row, sheet_name)
        if not text.strip():
            continue

        # Detect sub-category from row content
        row_text_lower = text.lower()
        extra_categories = []

        # Detect from actual column values instead of row text
        ram = row.get("RAM (GB)", "")
        vcpu = row.get("vCPU", "")

        try:
            ram_val = float(ram)
            if ram_val >= 32:
                extra_categories.append("memory_intensive")
        except (ValueError, TypeError):
            rag_logger.debug(f"Could not parse RAM value: '{ram}' in sheet '{sheet_name}'")

        try:
            vcpu_val = float(vcpu)
            if vcpu_val >= 16:
                extra_categories.append("cpu_intensive")
        except (ValueError, TypeError):
            rag_logger.debug(f"Could not parse vCPU value: '{vcpu}' in sheet '{sheet_name}'")

        # general_purpose is everything else in compute sheets
        if not extra_categories and sheet_name in ("Compute - Linux", "Compute - Windows"):
            extra_categories.append("general_purpose")

        chunks.append({
            "text"      : text,
            "sheet"     : sheet_name,
            "categories": base_categories + extra_categories,
            "vector"    : None,   # filled by _load_rate_card after encode
        })

    return chunks


def _sheet_to_categories(sheet_name: str) -> list[str]:
    name = sheet_name.lower()
    if "linux" in name:
        return ["linux", "compute"]
    if "windows" in name:
        return ["windows", "compute"]
    if "storage" in name:
        return ["storage"]
    if "backup" in name:          # catches both "Backup - Capacity" and "Backup - Advanced"
        return ["backup"]
    if "network" in name:
        return ["networking"]
    if "operating" in name:
        return ["operating_systems"]
    return ["general"]


def _row_to_text(row: pd.Series, sheet_name: str) -> str:
    """Convert a DataFrame row into a readable text string for embedding."""
    parts = [f"[{sheet_name}]"]
    for col, val in row.items():
        if str(val).strip():
            parts.append(f"{col}: {val}")
    return " | ".join(parts)


# Load rate card at import
_load_rate_card()


# ── Keyword filter ─────────────────────────────────────────────────────────────

def _keyword_filter(query: str, session_id: str) -> list[str]:
    """
    Detect relevant categories from customer transcript words.
    Returns list of matched category strings.
    Returns empty list if no match (caller will use all chunks as fallback).
    """
    words    = query.lower().split()
    detected = []

    for category, triggers in CATEGORY_MAP.items():
        for word in words:
            if word in triggers:
                if category not in detected:
                    detected.append(category)
                break

    rag_logger.debug(f"[{session_id}] Keyword filter detected: {detected or 'none — using all chunks'}")
    return detected


def _filter_chunks(detected_categories: list[str], session_id: str) -> list[dict]:
    """
    Filter chunks to only those matching detected categories.
    Falls back to all chunks if no categories detected.
    """
    if not detected_categories:
        rag_logger.debug(f"[{session_id}] No category match — searching all {len(_chunks)} chunks")
        return _chunks

    filtered = [
        chunk for chunk in _chunks
        if any(cat in chunk["categories"] for cat in detected_categories)
    ]

    rag_logger.debug(f"[{session_id}] Filtered to {len(filtered)} chunks from {len(_chunks)}")
    return filtered


def _embedding_search(query: str, filtered: list[dict], session_id: str, top_n: int = TOP_N) -> list[dict]:
    """
    Encode query and compute cosine similarity against filtered chunks.
    Returns top_n highest scoring chunks.
    """
    if not filtered:
        rag_logger.warning(f"[{session_id}] No chunks to search")
        return []

    query_vector  = _model.encode([query])
    chunk_vectors = np.array([c["vector"] for c in filtered])

    scores  = cosine_similarity(query_vector, chunk_vectors)[0]
    top_idx = np.argsort(scores)[::-1][:top_n]

    top_chunks = [filtered[i] for i in top_idx]

    rag_logger.debug(
        f"[{session_id}] Top match: score={scores[top_idx[0]]:.3f} | "
        f"{top_chunks[0]['text'][:80]}"
    )

    return top_chunks


# ── Public interface ───────────────────────────────────────────────────────────

def retrieve(transcript: str, session_id: str, history: list | None = None) -> str:
    """
    Main RAG function. Called by llm_service before every Gemini call.

    Stage 1: keyword filter on transcript → detected categories
    Stage 2: embedding search on filtered chunks → top 3 chunks
    Returns formatted pricing context string to inject into Gemini prompt.

    Args:
        transcript: Current customer message
        session_id: Session ID for log tracing
        history:    Conversation history (reserved for future use)

    Returns:
        Formatted string with top matching pricing rows.
        Empty string if no chunks found.
    """
    rag_logger.debug(f"[{session_id}] RAG retrieve: {transcript[:80]}")

    detected   = _keyword_filter(transcript, session_id)
    filtered   = _filter_chunks(detected, session_id)
    top_chunks = _embedding_search(transcript, filtered, session_id)

    rag_logger.info(f"[{session_id}] RAG found {detected} as detected keywords")
    rag_logger.info(f"[{session_id}] RAG found {len(filtered)} filtered chunks out of {len(_chunks)} total")
    rag_logger.info(f"[{session_id}] RAG found {len(top_chunks)} top chunks out of {len(filtered)} filtered")

    if not top_chunks:
        rag_logger.info(f"[{session_id}] No relevant pricing chunks found")
        return ""

    lines = ["Relevant pricing from Intelics rate card:"]
    for chunk in top_chunks:
        lines.append(f"- {chunk['text']}")

    rag_logger.info(f"[{session_id}] Returning {len(top_chunks)} pricing chunks to LLM")
    return "\n".join(lines)