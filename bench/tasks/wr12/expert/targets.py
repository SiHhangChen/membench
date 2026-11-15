"""Private expert placement selection, never a policy observation."""

from ..domain.contracts import BLOCK_ID


def resolve_station_targets(episode, *, purpose=None):
    return {"pick_item": BLOCK_ID, "delivery_support": f"tray_{episode.facts['target_box']}"}
