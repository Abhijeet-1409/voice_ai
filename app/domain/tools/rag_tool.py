import re

from livekit.agents import function_tool, RunContext
from sentence_transformers import SentenceTransformer

from infra.vector_store.pg_store import vector_store
from config.logging import get_logger

logger = get_logger("domain.tools.rag")

# Load once at module level — loading takes a few seconds
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

MAX_WORDS = 200


def _format_for_voice(raw: str) -> str:
    """
    Strip markdown and format raw text chunks into voice-friendly prose.
    Gemini speaks whatever is returned — markdown becomes literal asterisks.
    """
    # Remove markdown bold, italic, headers, bullets, numbered lists
    text = re.sub(r"\*\*|__|\*|_|#{1,6}\s?", "", raw)
    text = re.sub(r"^\s*[-•]\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", " ", text, flags=re.MULTILINE)

    # Collapse newlines and extra spaces into single spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to MAX_WORDS
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])

    return text


@function_tool
async def search_knowledge_base(ctx: RunContext, query: str) -> str:
    """
    Search the Intelics Cloud knowledge base for information about products,
    plans, pricing, features, policies, and FAQs. ALWAYS call this tool
    before answering any question about Intelics Cloud Services. Never
    answer product or pricing questions from memory.
    """
    logger.debug(f"search_knowledge_base called — query={query}")
    try:
        query_vector = model.encode(query).tolist()
        chunks = await vector_store.search(query_vector, top_k=2)

        if not chunks:
            logger.warning(f"No results found for query={query}")
            return "I don't have that specific information in our knowledge base. Let me connect you with our team for more details."

        raw = " ".join(chunks)
        result = _format_for_voice(raw)
        logger.debug(f"RAG result for query={query}: {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"search_knowledge_base failed — query={query} error={e}")
        return "I was unable to search our knowledge base at this time. Please allow me to connect you with our team."