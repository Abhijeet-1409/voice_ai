import asyncio

from livekit.agents import ToolError, function_tool, RunContext

from shared.logging_setup import get_logger
from shared.infra.vector_store import get_pgvectorstore, VectorStoreError

from config import get_worker_settings
from rag import get_embedding_model


_LOGGER = "worker.domain.tools.knowledge_base"
logger = get_logger(_LOGGER)


@function_tool
async def search_knowledge_base(ctx: RunContext, query: str) -> str:
    """
    Searches the vector knowledge base for information regarding Intelics services, 
    AWS partner programs, and internal cloud offerings.

    This tool MUST be called to retrieve verified context before answering factual 
    questions to avoid hallucination.

    Args:
        ctx (RunContext): The LiveKit agent execution context.
        query (str): The search query or specific topic to look up.

    Returns:
        str: A combined string of relevant text passages retrieved from the vector store, 
            or a generic string indicating that no information was found.

    Raises:
        ToolError: If the vector store is unreachable or if an unexpected exception occurs 
            during embedding generation or vector retrieval.
    """
    try:
        worker_settings = get_worker_settings()
        pg_store_client = get_pgvectorstore()
        embedding_model = get_embedding_model()

        logger.debug(f"Generating embeddings for knowledge base search | query='{query}'")
        query_vector = await asyncio.to_thread(embedding_model.encode, query)
        query_vector = query_vector.tolist()

        logger.debug(f"Executing vector search | top_k={worker_settings.RAG_TOP_K}")
        search_results = await pg_store_client.search(query_vector, worker_settings.RAG_TOP_K)

        if not search_results:
            logger.info(f"Knowledge base search yielded no results | query='{query}'")
            return "No relevant information found in the knowledge base."

        formatted_context = "\n\n".join(search_results)

        logger.info(f"Knowledge base search successful | chunks_retrieved={len(search_results)} | query='{query}'")
        return formatted_context

    except VectorStoreError as vec_err:
        logger.error(f"Vector store operation failed during search | query='{query}' | error={vec_err}")
        raise ToolError(
            "The knowledge base is temporarily unavailable. Let the caller know a "
            "specialist will follow up with the details instead of guessing."
        ) from vec_err
        
    except Exception as unexp_err:
        # Secondary, broader safety net — covers failures outside the
        # vector store itself (e.g. embedding_model.encode() raising due
        # to a bad input, device/tokenizer error). VectorStoreError is
        # caught first/specifically above; anything else lands here.
        logger.error(f"Unexpected error occurred during knowledge base search | query='{query}' | error={unexp_err}")
        raise ToolError(
            "Something went wrong while looking that up. Let the caller know a "
            "specialist will follow up with the details instead of guessing."
        ) from unexp_err
