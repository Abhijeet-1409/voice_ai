from schemas import UserData


def build_user_context_block(userdata: UserData) -> str:
    """
    Renders UserData into a fixed-shape, LLM-friendly text block for
    injection into the {user_context} placeholder in domain/system_prompt.py's
    prompt templates.

    Always includes every field (using "Unknown"/"Not yet determined" for
    unset values) rather than conditionally omitting empty fields — see
    design decision: a fixed, predictable shape every call is preferred
    over variable-length output, and explicit placeholders avoid raw
    "None" strings leaking into the prompt.

    Args:
        userdata: The current call's UserData.

    Returns:
        A formatted multi-line string ready to fill {user_context}.
    """
    return (
        f"Contact name: {userdata.name or 'Unknown'}\n"
        f"Phone number: {userdata.phone or 'Unknown'}\n"
        f"Email: {userdata.email or 'Unknown'}\n"
        f"Channel: {userdata.channel}\n"
        f"Call type: {userdata.call_type}\n"
        f"Track: {userdata.track or 'Not yet determined'}\n"
        f"Previously qualified: {'Yes' if userdata.qualified else 'No'}\n"
        f"Lifecycle stage: {userdata.lifecyclestage}"
    )