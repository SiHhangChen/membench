"""One instruction independent of the hidden location."""

from bench.tasks.contracts import PromptSpec

PROMPT = (
    "Observe which box contains the can during the one-second viewing period. "
    "After both boxes close, move to the front, pick up the block, "
    "and place it in the tray in front of the remembered box."
)


def resolve_prompt_spec(*, variant):
    if variant != "memory":
        raise ValueError("WR-12 currently supports only the memory variant")
    return PromptSpec(PROMPT, PROMPT, variant)
