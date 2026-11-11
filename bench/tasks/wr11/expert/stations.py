"""The two navigation workspaces are independent of the sample identity."""

from ..execution.navigation import station_specs


def resolve_station_targets(episode, *, purpose=None):
    del episode, purpose
    return {"observe": "sample_station", "pick": "stock_station", "deliver": "sample_station"}


__all__ = ["resolve_station_targets", "station_specs"]
