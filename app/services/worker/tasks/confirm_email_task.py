from typing import Optional

from livekit.agents import AgentTask, ChatContext, function_tool
from livekit.agents.llm import LLM

from shared.logging_setup import get_logger

from utils import build_confirm_email_task_prompt


_LOGGER = "worker.tasks.confirm_email_task"
logger = get_logger(_LOGGER)


class ConfirmEmailTask(AgentTask[str]):
    """Task for verifying or collecting a caller's email address."""

    def __init__(
        self,
        candidate_email: Optional[str] = None,
        chat_ctx: Optional[ChatContext] = None,
        task_llm: Optional[LLM] = None,
        *,
        supervisor_name: str,
        supervisor_role: str,
        invocation_reason: str,
        known_context: str,
        expected_result: str,
        already_completed: str,
        previous_step: str,
        current_step: str,
        next_step: str,
    ) -> None:
        self.candidate_email = candidate_email
        self.task_llm = task_llm
        self._email_submitted = False

        instructions = build_confirm_email_task_prompt(
            supervisor_name=supervisor_name,
            supervisor_role=supervisor_role,
            invocation_reason=invocation_reason,
            known_context=known_context,
            expected_result=expected_result,
            already_completed=already_completed,
            previous_step=previous_step,
            current_step=current_step,
            next_step=next_step,
        )

        logger.info(
            "Initializing ConfirmEmailTask with candidate_email: %s",
            candidate_email,
        )

        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            llm=task_llm,
        )

    async def on_enter(self) -> None:
        """Start the email-confirmation interaction."""
        logger.info("Entering ConfirmEmailTask")

        if self.candidate_email:
            logger.info(
                "Asking user to confirm candidate_email: %s",
                self.candidate_email,
            )

            await self.session.generate_reply(
                instructions=(
                    f'Read the email address "{self.candidate_email}" back '
                    "clearly and ask the caller to explicitly confirm whether "
                    "it is correct."
                )
            )
            return

        logger.info(
            "No candidate email present; prompting user to provide email"
        )

        await self.session.generate_reply(
            instructions="Ask the caller for their email address."
        )

    @function_tool
    async def submit_email(
        self,
        email: str,
        read_back: bool,
    ) -> str:
        """Submit the confirmed email and complete the task."""
        logger.info(
            "submit_email called with email: '%s', read_back: %s",
            email,
            read_back,
        )

        if self._email_submitted:
            logger.warning("submit_email called after task completion")
            return (
                "This email-confirmation task has already been completed. "
                "Return the result to the supervisor."
            )

        if not read_back:
            logger.warning(
                "submit_email called without explicit confirmation"
            )

            return (
                "Read the email address back to the user and get explicit "
                "confirmation before calling this tool again."
            )

        email = email.strip()

        logger.info(
            "Email confirmed and task completing: %s",
            email,
        )

        self._email_submitted = True
        self.complete(email)

        return f"Email confirmed: {email}"
