"""Preset locations.

The European presets match the six regions from the original April villa
analysis; each is represented by a single representative point. ERA5 grid
cells are ~31 km wide, so a point stands for its surrounding area.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    group: str

    @property
    def label(self) -> str:
        return f"{self.name} ({self.group})"


EUROPE = [
    Place("Algarve", 37.10, -8.25, "Holiday"),
    Place("South France (Provence)", 43.55, 5.45, "Holiday"),
    Place("Croatia (Split)", 43.51, 16.44, "Holiday"),
    Place("Bosnia (Mostar)", 43.34, 17.81, "Holiday"),
    Place("Barcelona Coast", 41.40, 2.15, "Holiday"),
    Place("Romania (Bucharest)", 44.43, 26.10, "Holiday"),
]

UK = [
    Place("Cotswolds", 51.83, -1.85, "UK"),
    Place("Lake District", 54.45, -3.05, "UK"),
    Place("Cornwall", 50.26, -5.05, "UK"),
    Place("Peak District", 53.35, -1.80, "UK"),
    Place("Norfolk", 52.75, 1.00, "UK"),
    Place("Kent", 51.25, 0.75, "UK"),
    Place("Edinburgh", 55.95, -3.19, "UK"),
    Place("Snowdonia", 53.00, -3.90, "UK"),
]

PRESETS = EUROPE + UK
