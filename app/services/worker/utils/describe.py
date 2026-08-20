import inspect
import enum


def describe(cls: type) -> str:
    """
    Formats a class's docstring and, if it's an Enum, its valid member
    values, into a compact block for injecting into an LLM prompt.

    Used to give the LLM an explicit, always-current list of valid
    values for fields like Track/TicketPriority/TicketStatus, rather
    than relying solely on tool-schema enum hints (which some model/
    framework combinations surface inconsistently in practice).

    Args:
        cls: The class to describe — typically a StrEnum from
            shared.config.constants.

    Returns:
        A formatted string, e.g.:
            "Track: The AWS partner qualification track...
             Valid values: billing_transfer, green_field_migration, vmware_workload_migration"
    """
    doc = inspect.getdoc(cls) or "No description provided."

    if issubclass(cls, enum.Enum):
        values = ", ".join(member.value for member in cls)
        return f"{cls.__name__}: {doc}\nValid values: {values}"

    return f"{cls.__name__}: {doc}"


def describe_all(*classes: type) -> str:
    """
    Formats multiple classes via describe(), joined into one block —
    the actual content injected into {enum_reference} in the prompt
    templates.

    Args:
        *classes: Classes to describe, typically Track, TicketPriority,
            TicketStatus.

    Returns:
        A newline-separated block, one describe() entry per class.
    """
    return "\n\n".join(describe(cls) for cls in classes)