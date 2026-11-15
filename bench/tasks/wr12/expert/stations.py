"""Target-independent observation and manipulation workspaces."""

from ..execution.navigation import station_specs


def resolve_station_targets(episode, *, purpose=None):
    return {"observe": "observe_station", "pick": "operate_station", "place": "operate_station"}


__all__ = ["resolve_station_targets", "station_specs"]
