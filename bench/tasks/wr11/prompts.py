"""The request identity is visual and never appears in the persistent prompt."""

from bench.tasks.contracts import PromptSpec

PROMPT = (
    "Observe the color of the reference carton, then fetch a carton of the same color "
    "and place it in the empty tray."
)


def resolve_prompt_spec(*, variant):
    if variant != "memory":
        raise ValueError("WR-11 currently supports only the memory variant")
    return PromptSpec(PROMPT, PROMPT, variant)
