import inspect


def describe(cls: type) -> str:
    """
    Returns a formatted string combining a class's name and docstring,
    for injecting class descriptions into LLM prompts.
    """
    doc = inspect.getdoc(cls) or "No description provided."
    return f"Class {cls.__name__}:\n\t{doc}"