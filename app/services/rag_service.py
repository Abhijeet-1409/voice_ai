from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import rag_logger

# ── Paths ─────────────────────────────────────────────────────────────────────

RATE_CARD_PATH  = "/app/data/rate_card.xlsx"
RAG_MODEL_PATH  = "/app/rag_models/all-MiniLM-L6-v2"
TOP_N           = 3     # number of chunks to return per query


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


# ── Model and chunks load ONCE at import time ─────────────────────────────────

rag_logger.info("Loading embedding model...")

_model = SentenceTransformer(RAG_MODEL_PATH)

rag_logger.info("Reading rate card...")

_chunks: list[dict] = []


def _load_rate_card() -> None:
    """
    Read all 6 sheets from rate card Excel, build chunks, embed all at once.
    Called once at import time. Results stay in memory for server lifetime.
    """
    if not Path(RATE_CARD_PATH).exists():
        rag_logger.warning(f"Rate card not found at {RATE_CARD_PATH}")
        return

    sheets = pd.read_excel(RATE_CARD_PATH, sheet_name=None)

    for sheet_name, df in sheets.items():
        df = df.dropna(how="all")           # drop fully empty rows
        df = df.fillna("")                  # fill remaining NaN with empty string

        sheet_chunks = _build_chunks(df, sheet_name)
        _chunks.extend(sheet_chunks)

    rag_logger.info(f"Built {len(_chunks)} chunks from {len(sheets)} sheets.")

    # Embed all chunks in one batch — faster than one by one
    texts   = [c["text"] for c in _chunks]
    vectors = _model.encode(texts, show_progress_bar=False)

    for i, chunk in enumerate(_chunks):
        chunk["vector"] = vectors[i]

    rag_logger.info("All chunks embedded and ready.")


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

        if "general purpose" in row_text_lower:
            extra_categories.append("general_purpose")
        if "memory intensive" in row_text_lower:
            extra_categories.append("memory_intensive")
        if "cpu intensive" in row_text_lower:
            extra_categories.append("cpu_intensive")

        chunks.append({
            "text"      : text,
            "sheet"     : sheet_name,
            "categories": base_categories + extra_categories,
            "vector"    : None,   # filled by _load_rate_card after encode
        })

    return chunks


def _sheet_to_categories(sheet_name: str) -> list[str]:
    """Map sheet name to base category tags."""
    name = sheet_name.lower()
    if "linux" in name:
        return ["linux", "compute"]
    if "windows" in name:
        return ["windows", "compute"]
    if "storage" in name:
        return ["storage"]
    if "backup" in name:
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

    if not top_chunks:
        rag_logger.info(f"[{session_id}] No relevant pricing chunks found")
        return ""

    lines = ["Relevant pricing from Intelics rate card:"]
    for chunk in top_chunks:
        lines.append(f"- {chunk['text']}")

    rag_logger.info(f"[{session_id}] Returning {len(top_chunks)} pricing chunks to LLM")
    return "\n".join(lines)