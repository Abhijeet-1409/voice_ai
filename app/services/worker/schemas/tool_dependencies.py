from typing import Optional

from pydantic import BaseModel, ConfigDict

from livekit.agents.llm import LLM


class ToolDependencies(BaseModel):
    """
    Define the runtime dependencies shared by supervisor function tools.

    This schema groups the task LLM and supervisor metadata required by
    tools that delegate bounded operations to specialized AgentTask instances.
    It keeps these dependencies separate from the function parameters exposed
    to the LLM for tool calling.

    Fields:
        task_llm (Optional[LLM]): The language model used by specialized tasks.
            Defaults to None, allowing tasks to use their configured fallback.
        agent_name (str): The name of the supervisor agent invoking the
            function tools.
        company_name (str): The name of the company the supervisor represents,
            used to provide business context to specialized tasks.
    """

    task_llm: Optional[LLM] = None
    agent_name: str
    company_name: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

