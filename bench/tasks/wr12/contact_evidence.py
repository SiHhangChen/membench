"""Read-only, per-state contact queries using MuJoCo geom IDs."""

from __future__ import annotations

from typing import Any


class ContactIndex:
    def __init__(self, sim: Any, geom_ids: dict[str, int]) -> None:
        self.sim = sim
        self.geom_ids = geom_ids
        self._neighbors: dict[int, set[int]] | None = None

    def _group(self, group: Any) -> set[int]:
        names = [group] if isinstance(group, str) else getattr(group, "contact_geoms", group)
        result = set()
        for name in names:
            if name not in self.geom_ids:
                self.geom_ids[name] = int(self.sim.model.geom_name2id(name))
            result.add(self.geom_ids[name])
        return result

    def check(self, group_1: Any, group_2: Any = None) -> bool:
        ids_1 = self._group(group_1)
        ids_2 = None if group_2 is None else self._group(group_2)
        if self._neighbors is None:
            self._scan()
        return any(
            bool(self._neighbors.get(geom, set()))
            if ids_2 is None
            else not self._neighbors.get(geom, set()).isdisjoint(ids_2)
            for geom in ids_1
        )

    def _scan(self) -> None:
        neighbors: dict[int, set[int]] = {}
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            first, second = int(contact.geom1), int(contact.geom2)
            neighbors.setdefault(first, set()).add(second)
            neighbors.setdefault(second, set()).add(first)
        self._neighbors = neighbors
