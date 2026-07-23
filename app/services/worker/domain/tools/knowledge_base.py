from livekit.agents import function_tool, RunContext

from shared.config.logger import get_logger

from shared.infra.vector_store.pg_store import get_pgvectorstore

from config.worker_settings import get_worker_settings

from worker.embedding import get_embedding_model


_LOGGER = "worker.domain.tools.knowledge_base"
logger = get_logger(_LOGGER)


@function_tool
async def search_knowledge_base(ctx: RunContext, query: str) -> str:
    """
    Search the knowledge base for Intelics services, AWS partner program details, 
    and our own cloud offerings. ALWAYS call this before answering any factual 
    question — never answer from memory.

    Args:
        ctx (RunContext): The LiveKit agent execution context.
        query (str): The specific question or topic to search for.

    Returns:
        str: The retrieved context from the knowledge base, formatted as a string.
    """
    try:
        worker_settings = get_worker_settings()
        pg_store_client = get_pgvectorstore()
        embedding_model = get_embedding_model()
        
        query_vector = embedding_model.encode(query).tolist()
        
        search_results = await pg_store_client.search(query_vector, worker_settings.RAG_TOP_K)

        if not search_results:
            logger.debug(f"No results found in knowledge base — query='{query}'")
            return "No relevant information found in the knowledge base."

        formatted_context = "\n\n".join(search_results)

        logger.debug(f"Knowledge base search completed — query='{query}'")
        return formatted_context
        
    except Exception as e:
        logger.error(f"Failed to search knowledge base — query='{query}' error={e}")
        return "System error: unable to access the knowledge base at this time."