from livekit.agents import FunctionTool, function_tool, RunContext

from shared.logging_setup import get_logger

from tasks import ConfirmEmailTask
from schemas import ToolDependencies


_LOGGER = "worker.domain.tools.get_contact_info"
logger = get_logger(_LOGGER)


def create_get_contact_info_tool(deps: ToolDependencies) -> FunctionTool:
    """
    Create a function tool that confirms the caller's contact email.

    The factory captures runtime dependencies and returns a LiveKit
    FunctionTool. These dependencies remain internal to the Python
    implementation and are not exposed as parameters to the LLM.

    Args:
        deps: Runtime dependencies containing the task LLM, agent name,
            and company name used to configure ConfirmEmailTask.

    Returns:
        FunctionTool: A tool that confirms the caller's email, updates
        session user data, and returns the confirmation result.
    """

    @function_tool()
    async def get_contact_info(ctx: RunContext) -> str:
        """
        Collect and explicitly confirm the caller's email address.

        Use this tool when the caller's contact email must be collected
        or verified before proceeding with meeting scheduling. It delegates
        the confirmation process to ConfirmEmailTask and stores the
        confirmed email in the session's user data.

        This tool only handles contact-email confirmation. It does not
        schedule meetings or perform other business actions. The supervisor
        determines the next step after receiving the result.

        Args:
            ctx: LiveKit tool execution context containing the session,
                current agent, and caller's user data.

        Returns:
            str: A message indicating the result of email confirmation.
        """
        logger.info("Starting get_contact_info tool")
        logger.info("Executing ConfirmEmailTask")

        supervisor_chat_ctx = ctx.session.current_agent.chat_ctx.copy(
            exclude_function_call=True,
            exclude_instructions=True,
        )

        confirmed_email = await ConfirmEmailTask(
            candidate_email=ctx.userdata.email,
            chat_ctx=supervisor_chat_ctx,
            task_llm=deps.task_llm,
            supervisor_name=deps.agent_name,
            supervisor_role=(
                f"{deps.agent_name} is the primary voice assistant for "
                f"{deps.company_name}. The supervisor owns the overall caller "
                "conversation and business workflow, and decides what happens "
                "before and after each bounded subtask."
            ),
            invocation_reason=(
                "The supervisor needs to verify the caller's contact "
                "email before any meeting can be scheduled."
            ),
            known_context=(
                f"Caller name: {ctx.userdata.name or 'Unknown'}\n"
                f"Current email candidate: "
                f"{ctx.userdata.email or 'Not yet provided'}\n"
                f"Call type: {ctx.userdata.call_type}\n"
                f"Track: {ctx.userdata.track or 'Not yet determined'}"
            ),
            expected_result=(
                "Return one email address that the caller has explicitly "
                "confirmed."
            ),
            already_completed=(
                "Do not schedule a meeting, qualify callers, create tickets, "
                "or perform any other business action."
            ),
            previous_step=(
                "The supervisor has established that the caller's contact "
                "email needs to be confirmed."
            ),
            current_step="ConfirmEmailTask",
            next_step=(
                "Return the confirmed email to the supervisor, which will "
                "decide what happens next."
            ),
        )

        ctx.userdata.email = confirmed_email

        logger.info(f"Contact email confirmed: {confirmed_email}")

        return f"Contact information confirmed for {confirmed_email}."

    return get_contact_info
