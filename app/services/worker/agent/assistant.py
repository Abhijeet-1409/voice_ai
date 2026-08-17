from livekit.agents import Agent, function_tool
from livekit.agents.llm import Tool


class Assistant(Agent):
    """
    """
    def __init__(self, instructions: str, tools: list[Tool]) -> None:
        super().__init__(
            instructions=instructions,
            tools=tools
        )

    @function_tool()
    async def schedule_meeting(self):
        """
        """
        pass 
