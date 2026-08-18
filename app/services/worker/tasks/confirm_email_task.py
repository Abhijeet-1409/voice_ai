from typing import Optional
 
from livekit.agents import AgentTask, ChatContext, function_tool


class ConfirmEmailTask(AgentTask[str]):
    def __init__(self, chat_ctx: Optional[ChatContext] = None):
        super().__init__(
            instructions="""
            Ask the user for their email address. Once they provide it,
            repeat it back to them clearly — spell out the local part
            (before the @) character by character if it contains
            uncommon words, numbers, or could be ambiguous — and ask
            them to confirm it's correct.

            Only call `submit_email` after the user has explicitly
            confirmed the spelling is correct. If they say it's wrong,
            ask them to repeat or spell it again, then read it back
            and confirm once more before submitting.
            """,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="""
            Ask the user for their email address so you can confirm it
            on file.
            """
        )

    @function_tool
    async def submit_email(self, email: str, read_back: bool) -> str:
        """Submit the confirmed email address.

        Args:
            email: The email address to submit.
            read_back: Set to True only after you have read the email
                address back to the user character-by-character (for the
                local part) and they have explicitly confirmed it is
                correct.
        """
        if not read_back:
            return "Read the email address back to the user and get explicit confirmation before calling this tool again."

        email = email.strip()
        self.complete(email)
        return f"Email confirmed: {email}"