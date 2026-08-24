import asyncio

from shared.infra.vector_store import get_pgvectorstore
from shared.logging_setup import get_logger

from rag import get_embedding_model, build_all_texts as _build_all_texts


_LOGGER = "worker.rag.ingest"
logger = get_logger(_LOGGER)


async def run_ingestion() -> None:
    """
    Parses the rate card, embeds every chunk, and batch-inserts every
    (content, vector) pair into Postgres via PgVectorStore.insert.
    Run once, standalone — not on worker startup.
    """
    store = get_pgvectorstore()
    model = get_embedding_model()

    texts = _build_all_texts()
    if not texts:
        logger.warning("No chunks built — nothing to ingest")
        return

    try:
        vectors = model.encode(texts, show_progress_bar=False)

        items = [(text_val, vector.tolist()) for text_val, vector in zip(texts, vectors)]

        await store.insert(items)

        logger.info(f"Ingested {len(items)} chunks into knowledge_base.")

    except Exception as e:
        logger.error(f"Ingestion failed — {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_ingestion())