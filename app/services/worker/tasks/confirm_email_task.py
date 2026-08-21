from typing import Optional

from livekit.agents import AgentTask, ChatContext, function_tool

from shared.logging_setup import get_logger


__LOGGER = "worker.tasks.confirm_email_task"
logger = get_logger(__LOGGER)


class ConfirmEmailTask(AgentTask[str]):
    """Resolves a confirmed email address over voice.

    Two entry paths, chosen automatically based on whether a candidate
    email is already known:
      - candidate_email given (e.g. from a returning caller's CRM
        record, or already set earlier this call via update_caller_info):
        reads it back and asks for a yes/no-or-correction, rather than
        asking the caller to state their email from scratch.
      - candidate_email is None (or the caller says it's wrong): asks
        for the email fresh, spelling out the local part character by
        character when it contains uncommon words, numbers, or is
        otherwise ambiguous.

    Either path requires explicit user confirmation before completing,
    enforced via a `read_back` self-report flag on `submit_email` rather
    than trusting the LLM's judgment alone.

    Result:
        The confirmed email address as a `str`.
    """
    def __init__(self, candidate_email: Optional[str] = None, chat_ctx: Optional[ChatContext] = None):
        logger.info("Initializing ConfirmEmailTask with candidate_email: %s", candidate_email)
        super().__init__(
            instructions="""
            If a candidate email is provided in your context, read it
            back to the user and ask them to confirm it's still correct,
            or provide a different one if not.

            If no candidate email is provided, or the user says the
            candidate is wrong, ask the user for their email address.
            Once they provide it, repeat it back to them clearly — spell
            out the local part (before the @) character by character if
            it contains uncommon words, numbers, or could be ambiguous —
            and ask them to confirm it's correct.

            Only call `submit_email` after the user has explicitly
            confirmed the email (whether the candidate or a freshly
            provided one) is correct. If they say it's wrong, ask them
            to repeat or spell it again, then read it back and confirm
            once more before submitting.
            """,
            chat_ctx=chat_ctx,
        )
        self.candidate_email = candidate_email

    async def on_enter(self) -> None:
        logger.info("Entering ConfirmEmailTask")
        if self.candidate_email:
            logger.info("Asking user to confirm existing candidate_email: %s", self.candidate_email)
            await self.session.generate_reply(
                instructions=f"""
                Read back the email address "{self.candidate_email}" to
                the user and ask them to confirm it's still correct, or
                let you know if it's changed.
                """
            )
        else:
            logger.info("No candidate email present; prompting user to provide email")
            await self.session.generate_reply(
                instructions="""
                Ask the user for their email address so you can confirm
                it on file.
                """
            )

    @function_tool
    async def submit_email(self, email: str, read_back: bool) -> str:
        """Submit the confirmed email address.

        Args:
            email: The email address to submit — either the confirmed
                candidate, or a freshly provided and confirmed one.
            read_back: Set to True only after you have read the email
                address back to the user character-by-character (for the
                local part) and they have explicitly confirmed it is
                correct.
        """
        logger.info("submit_email called with email: '%s', read_back: %s", email, read_back)

        if not read_back:
            logger.warning("submit_email failed because read_back was set to False")
            return "Read the email address back to the user and get explicit confirmation before calling this tool again."

        email = email.strip()
        logger.info("Email confirmed and task completing: %s", email)
        self.complete(email)
        return f"Email confirmed: {email}"