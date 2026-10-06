from typing import Optional

from livekit.agents import AgentTask, ChatContext, function_tool
from livekit.agents.llm import LLM

from shared.logging_setup import get_logger

from domain import CONFIRM_EMAIL_TASK_PROMPT


__LOGGER = "worker.tasks.confirm_email_task"
logger = get_logger(__LOGGER)


class ConfirmEmailTask(AgentTask[str]):
    """Task that verifies or collects an email address over a voice session.

    Handles two conversational paths based on initial input:
      1. Candidate email available: Reads the address back for confirmation or correction.
      2. No candidate email (or incorrect): Requests a new email address, spelling out
         the local part character-by-character when necessary.

    Enforces explicit user verification before completing via a `read_back` confirmation
    flag required by `submit_email`.

    Attributes:
        candidate_email (Optional[str]): Initial candidate email address passed into the task.
        task_llm (Optional[LLM]): Specialized LLM instance used for task execution.
    """

    def __init__(
        self,
        candidate_email: Optional[str] = None,
        chat_ctx: Optional[ChatContext] = None,
        task_llm: Optional[LLM] = None,
    ) -> None:
        """Initializes a new ConfirmEmailTask instance.

        Args:
            candidate_email (Optional[str]): Existing candidate email to verify, if known.
                Defaults to None.
            chat_ctx (Optional[ChatContext]): Pre-existing chat context. Defaults to None.
            task_llm (Optional[LLM]): LLM instance dedicated to task execution. Defaults to None.
        """
        self.task_llm = task_llm

        logger.info("Initializing ConfirmEmailTask with candidate_email: %s", candidate_email)
        super().__init__(
            instructions=CONFIRM_EMAIL_TASK_PROMPT,
            chat_ctx=chat_ctx,
            llm=task_llm,
        )
        self.candidate_email = candidate_email

    async def on_enter(self) -> None:
        """Lifecycle hook executed upon entering the email confirmation task.

        Prompts the caller to either confirm an existing candidate email or
        provide a new email address.
        """
        logger.info("Entering ConfirmEmailTask")
        if self.candidate_email:
            logger.info("Asking user to confirm existing candidate_email: %s", self.candidate_email)
            await self.session.generate_reply(
                instructions=f"""
                Read the email address "{self.candidate_email}" back clearly and ask the caller to confirm whether it is correct.
                """
            )
        else:
            logger.info("No candidate email present; prompting user to provide email")
            await self.session.generate_reply(
                instructions="""
                Ask the caller for their email address.
                """
            )

    @function_tool
    async def submit_email(self, email: str, read_back: bool) -> str:
        """Submits the caller's confirmed email address and completes the task.

        Args:
            email (str): The confirmed candidate or freshly collected email address.
            read_back (bool): Must be `True` to certify that the email (and its local part)
                was read back to the user and explicitly acknowledged as correct.

        Returns:
            str: A natural-language confirmation message or an instruction prompting
                the LLM to complete the read-back step.
        """
        logger.info("submit_email called with email: '%s', read_back: %s", email, read_back)

        if not read_back:
            logger.warning("submit_email failed because read_back was set to False")
            return "Read the email address back to the user and get explicit confirmation before calling this tool again."

        email = email.strip()
        logger.info("Email confirmed and task completing: %s", email)
        self.complete(email)
        return f"Email confirmed: {email}"