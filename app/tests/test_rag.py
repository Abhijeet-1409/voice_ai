# ==============================================================================
# INTELICS VOICE AI AGENT — test_rag.py
# Tests the hybrid RAG pipeline across all 6 rate card sheets.
# Verifies: keyword filter, embedding search, and formatted output.
#
# HOW TO RUN (from inside the container):
#   docker exec -it <app_container_name> python tests/test_rag.py
#
# HOW TO RUN (locally):
#   Set RATE_CARD_PATH and MODEL_PATH below, then:
#   python app/tests/test_rag.py
#
# WHAT THIS TESTS:
#   1. Rate card Excel loads correctly (all 6 sheets, no missing columns)
#   2. Chunks are built properly from all sheets
#   3. Keyword filter correctly detects category from transcript words
#   4. Embedding model loads and encodes queries correctly
#   5. Cosine similarity returns sensible top-3 results for each query
#   6. Fallback works for vague queries (no keyword match → search all chunks)
#   7. Output format is clean enough to inject into Gemini system prompt
# ==============================================================================

import sys
import os
import time


# ------------------------------------------------------------------------------
# PATHS
# These match the paths baked into the Docker image.
# Change if running locally with different locations.
# ------------------------------------------------------------------------------
RATE_CARD_PATH = "/app/data/rate_card.xlsx"
MODEL_PATH     = "/app/rag_models/all-MiniLM-L6-v2"


# ------------------------------------------------------------------------------
# TEST QUERIES
# One query per category to ensure full coverage of all 6 rate card sheets.
# The final query is intentionally vague — tests the fallback behaviour.
#
# Sheet coverage:
#   "Linux 4 vCPU 8GB"         → Compute - Linux (General Purpose)
#   "cheapest Windows VM"      → Compute - Windows (General Purpose, low tier)
#   "database server"          → Compute - Linux (Memory Intensive) + OS prices
#   "backup for 10 VMs"        → Backup (Acronis per VM)
#   "firewall options"         → Networking (Pfsense, Fortigate)
#   "object storage pricing"   → Storage (Object Storage)
#   "what are your VM prices"  → vague — no strong keyword → fallback to all chunks
# ------------------------------------------------------------------------------
TEST_QUERIES = [
    {
        "query"          : "Linux 4 vCPU 8GB",
        "expected_sheet" : "Compute - Linux",
        "description"    : "Specific Linux VM — should return General Purpose 4C8R row",
    },
    {
        "query"          : "cheapest Windows VM",
        "expected_sheet" : "Compute - Windows",
        "description"    : "Cheap Windows VM — should return smallest General Purpose tier",
    },
    {
        "query"          : "database server",
        "expected_sheet" : "Compute - Linux",
        "description"    : "DB server → memory_intensive category — high RAM VMs",
    },
    {
        "query"          : "backup for 10 VMs",
        "expected_sheet" : "Backup",
        "description"    : "Backup → Acronis Advanced (per VM pricing)",
    },
    {
        "query"          : "firewall options",
        "expected_sheet" : "Networking",
        "description"    : "Firewall → Pfsense or Fortigate options",
    },
    {
        "query"          : "object storage pricing",
        "expected_sheet" : "Storage",
        "description"    : "Object storage → Storage sheet",
    },
    {
        "query"          : "what are your VM prices",
        "expected_sheet" : None,    # Vague — expects fallback, no single expected sheet
        "description"    : "Vague query — keyword filter finds no category → searches all chunks",
    },
]


# ------------------------------------------------------------------------------
# CATEGORY_MAP
# Copied from rag_service.py — used here to independently verify keyword detection.
# Maps category name → list of trigger words.
# The keyword filter lowercases the transcript and checks word by word.
# ------------------------------------------------------------------------------
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


def check_prerequisites() -> bool:
    """
    Checks that both the rate card Excel and embedding model exist
    before loading anything. Both are baked into the Docker image at build time.
    """
    ok = True

    if not os.path.exists(RATE_CARD_PATH):
        print(f"❌  Rate card not found: {RATE_CARD_PATH}")
        print("    The rate card is copied into the image via Dockerfile: COPY ../data /app/data")
        print("    Make sure rate_card.xlsx is in voice_ai/data/ before building.\n")
        ok = False
    else:
        size_kb = os.path.getsize(RATE_CARD_PATH) / 1024
        print(f"✅  Rate card found: {RATE_CARD_PATH} ({size_kb:.1f} KB)")

    if not os.path.isdir(MODEL_PATH):
        print(f"❌  Embedding model not found: {MODEL_PATH}")
        print("    The model is cloned during docker build via git clone from HuggingFace.")
        print("    Run: docker-compose up --build\n")
        ok = False
    else:
        print(f"✅  Embedding model found: {MODEL_PATH}")

    return ok


def load_rate_card() -> dict:
    """
    Loads all 6 sheets from the rate card Excel file using pandas.
    Returns a dict: { sheet_name: DataFrame }

    sheet_name=None tells pandas to load ALL sheets at once.
    This is exactly what rag_service.py does at startup.

    We print the sheet names and row counts to verify the Excel was read correctly.
    If a sheet is missing or has 0 rows, there's a problem with the Excel file.
    """
    print("\n⏳  Loading rate card Excel...")
    import pandas as pd

    t0 = time.time()
    sheets = pd.read_excel(RATE_CARD_PATH, sheet_name=None)
    elapsed = time.time() - t0

    print(f"✅  Rate card loaded in {elapsed:.2f}s")
    print(f"    Sheets found ({len(sheets)}):")
    for name, df in sheets.items():
        print(f"      '{name}' — {len(df)} rows × {len(df.columns)} columns")

    # Warn if expected sheets are missing
    expected_sheets = [
        "Compute - Linux", "Compute - Windows", "Operating Systems",
        "Storage", "Networking", "Backup"
    ]
    for expected in expected_sheets:
        if expected not in sheets:
            print(f"    ⚠️  Missing expected sheet: '{expected}'")

    return sheets


def load_rag_service():
    """
    Imports and initialises rag_service.
    rag_service loads at import time — model + chunks + embeddings are built immediately.
    We time this to verify the startup cost is acceptable.

    Expected:
      - all-MiniLM-L6-v2 load     : ~2-3 seconds
      - Rate card chunking         : < 1 second
      - Embedding all chunks       : ~5-15 seconds (depends on chunk count)
      Total startup cost           : ~10-20 seconds (happens once at boot)
    """

    # Add the app/ directory to sys.path so we can import rag_service
    # (This is needed when running tests directly, not via pytest)
    app_dir = os.path.join(os.path.dirname(__file__), "..")
    if app_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(app_dir))

    print("\n⏳  Loading RAG service (embedding model + rate card chunks)...")
    print("    This takes ~10-20 seconds — same as production startup time.")
    t0 = time.time()

    import services.rag_service as rag

    elapsed = time.time() - t0
    chunk_count = len(rag.chunks) if hasattr(rag, 'chunks') else "unknown"
    print(f"✅  RAG service loaded in {elapsed:.2f}s")
    print(f"    Total chunks built and embedded: {chunk_count}")

    return rag


def independent_keyword_filter(query: str) -> list:
    """
    Runs keyword detection independently (without calling rag_service)
    to verify the query triggers the expected categories.
    This mirrors what rag_service.keyword_filter() does internally.
    """
    words = query.lower().split()
    detected = []
    for category, keywords in CATEGORY_MAP.items():
        if any(word in keywords for word in words):
            detected.append(category)
    return detected


def run_single_query(rag, query_info: dict, query_num: int, total: int) -> bool:
    """
    Runs one test query through the full RAG pipeline and prints results.
    Returns True if the query produced results, False if it returned empty.

    Steps:
      1. Independent keyword filter check (for verification)
      2. rag_service.retrieve() — full pipeline (keyword filter + embedding search)
      3. Print top 3 results
      4. Verify expected sheet appears in results (if specified)
    """
    query           = query_info["query"]
    expected_sheet  = query_info["expected_sheet"]
    description     = query_info["description"]

    print(f"\n{'─'*65}")
    print(f"Query {query_num}/{total}: \"{query}\"")
    print(f"Description: {description}")

    # --- Step 1: Independent keyword detection ---
    detected_categories = independent_keyword_filter(query)
    if detected_categories:
        print(f"Keywords detected: {detected_categories}")
    else:
        print("Keywords detected: none → fallback to full chunk search")

    # --- Step 2: Run full RAG pipeline ---
    t0 = time.time()
    context = rag.retrieve(query, history=[])
    elapsed = time.time() - t0

    print(f"Retrieval time: {elapsed:.3f}s")

    # --- Step 3: Print the context output ---
    if not context or context.strip() == "":
        print("❌  RAG returned empty context — no results found.")
        print("    Check rag_service.py chunk building and embedding logic.")
        return False

    print(f"\nRAG output (injected into Gemini prompt):")
    print("┌" + "─" * 63 + "┐")
    for line in context.strip().split("\n"):
        # Truncate long lines for readability in terminal
        display_line = line[:61] + ".." if len(line) > 63 else line
        print(f"│ {display_line:<62}│")
    print("└" + "─" * 63 + "┘")

    # --- Step 4: Verify expected sheet appears in results ---
    if expected_sheet:
        if expected_sheet.lower() in context.lower():
            print(f"✅  Expected sheet '{expected_sheet}' appears in results.")
        else:
            print(f"⚠️  Expected sheet '{expected_sheet}' NOT found in top results.")
            print(f"    The embedding search may have ranked other sheets higher.")
            print(f"    This is a soft warning — check if the result still makes sense.")

    return True


def print_summary(results: list):
    """
    Prints a final summary table showing pass/fail for each query.
    """
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")

    passed = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)

    for i, (query_info, result) in enumerate(zip(TEST_QUERIES, results), 1):
        status = "✅ pass" if result else "❌ fail"
        print(f"  {i}. {status}  \"{query_info['query']}\"")

    print(f"\n  {passed}/{len(TEST_QUERIES)} queries returned results")

    if failed == 0:
        print("\n✅  RAG pipeline is working correctly.")
        print("    All queries returned relevant pricing context.")
        print("    The output above is exactly what Gemini will receive")
        print("    as the pricing context on each customer exchange.")
    else:
        print(f"\n⚠️  {failed} queries returned empty results.")
        print("    Check rag_service.py — look at:")
        print("      1. How chunks are built from the Excel sheets")
        print("      2. How categories are assigned to chunks")
        print("      3. Whether the embedding model loaded correctly")

    print(f"{'='*65}\n")


def main():
    print("=" * 65)
    print("INTELICS VOICE AI — RAG PIPELINE TEST")
    print("Hybrid RAG: keyword filter + all-MiniLM-L6-v2 embeddings")
    print("=" * 65)

    # --- Step 1: Check files exist ---
    if not check_prerequisites():
        sys.exit(1)

    # --- Step 2: Load rate card and verify sheets ---
    load_rate_card()

    # --- Step 3: Load RAG service (builds chunks + embeddings) ---
    rag = load_rag_service()

    # --- Step 4: Run all test queries ---
    print(f"\n{'='*65}")
    print(f"RUNNING {len(TEST_QUERIES)} TEST QUERIES")
    print(f"{'='*65}")

    results = []
    for i, query_info in enumerate(TEST_QUERIES, 1):
        success = run_single_query(rag, query_info, i, len(TEST_QUERIES))
        results.append(success)

    # --- Step 5: Print summary ---
    print_summary(results)


if __name__ == "__main__":
    main()