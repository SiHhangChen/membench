"""Private expert selection; this mapping is never part of policy input."""


def resolve_station_targets(episode, *, purpose=None):
    del purpose
    return {"pick_item": episode.facts["matching_item"], "delivery_support": "delivery_tray"}


__all__ = ["resolve_station_targets"]
