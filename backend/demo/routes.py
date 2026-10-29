"""Demo road route catalog and synthetic road geometry for Delhivery hackathon demo.

Provides validated road corridors for 2 vehicles and 4 delivery stops with GeoJSON
coordinates in [longitude, latitude] format for Leaflet mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RoadSegment:
    segment_id: str
    name: str
    base_duration_seconds: float
    length_metres: float
    coordinates: list[list[float]]  # [[lon, lat], ...]


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    corridor_id: str
    name: str
    vehicle_id: str
    stop_ids: list[str]
    segments: list[RoadSegment]
    is_default: bool = False

    @property
    def total_duration_seconds(self) -> float:
        return sum(s.base_duration_seconds for s in self.segments)

    @property
    def total_distance_metres(self) -> float:
        return sum(s.length_metres for s in self.segments)

    def to_geojson(self) -> dict[str, Any]:
        all_coords: list[list[float]] = []
        for s in self.segments:
            all_coords.extend(s.coordinates)
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": all_coords,
            },
            "properties": {
                "route_id": self.route_id,
                "corridor_id": self.corridor_id,
                "name": self.name,
                "vehicle_id": self.vehicle_id,
                "stop_ids": self.stop_ids,
                "duration_seconds": self.total_duration_seconds,
                "distance_metres": self.total_distance_metres,
            },
        }


# -----------------------------------------------------------------------------
# Demo Delhivery Corridor Coordinates (South Delhi / Okhla - Gurgaon Hub)
# -----------------------------------------------------------------------------

# Vehicle 1: Depot -> Stop 1 (Cyber City) -> Stop 2 (Udyog Vihar) -> Depot
# Corridor A: NH-48 Expressway Corridor (Faster base: 12 min, high congestion exposure)
SEGMENT_V1_EXPRESSWAY = RoadSegment(
    segment_id="seg_nh48_exp",
    name="NH-48 Main Expressway",
    base_duration_seconds=720.0,  # 12 minutes
    length_metres=8500.0,
    coordinates=[
        [77.1025, 28.5245],  # Okhla Hub Depot
        [77.0880, 28.5120],  # Mahipalpur
        [77.0650, 28.4980],  # Rajokri Flyover
        [77.0420, 28.4860],  # Cyber City Toll Gate
        [77.0310, 28.4790],  # Stop 1 (Cyber Hub)
    ],
)

SEGMENT_V1_LAST_MILE = RoadSegment(
    segment_id="seg_v1_lastmile",
    name="Udyog Vihar Loop",
    base_duration_seconds=480.0,  # 8 minutes
    length_metres=4200.0,
    coordinates=[
        [77.0310, 28.4790],  # Stop 1
        [77.0250, 28.4850],  # Stop 2 (Udyog Vihar Phase IV)
        [77.0380, 28.5020],  # Return loop
        [77.1025, 28.5245],  # Back to Depot
    ],
)

# Corridor B: MG Road / Mehrauli Bypass (Longer base: 16 min, unaffected bypass)
SEGMENT_V1_BYPASS = RoadSegment(
    segment_id="seg_mgroad_bypass",
    name="MG Road Mehrauli Bypass",
    base_duration_seconds=960.0,  # 16 minutes
    length_metres=11200.0,
    coordinates=[
        [77.1025, 28.5245],  # Okhla Hub Depot
        [77.1200, 28.5050],  # Mehrauli Ring
        [77.1110, 28.4820],  # Sultanpur
        [77.0950, 28.4680],  # Sikanderpur
        [77.0310, 28.4790],  # Stop 1 (Cyber Hub)
    ],
)

# Vehicle 2: Depot -> Stop 3 (Saket) -> Stop 4 (Nehru Place) -> Depot
SEGMENT_V2_SOUTH = RoadSegment(
    segment_id="seg_south_arterial",
    name="Outer Ring Road South",
    base_duration_seconds=840.0,  # 14 minutes
    length_metres=9800.0,
    coordinates=[
        [77.1025, 28.5245],  # Depot
        [77.1550, 28.5280],  # Saket District Centre (Stop 3)
        [77.1920, 28.5490],  # Nehru Place Market (Stop 4)
        [77.1025, 28.5245],  # Return Depot
    ],
)


# Prevalidated Candidate Catalog
ROUTE_CANDIDATES: dict[str, RouteCandidate] = {
    # Vehicle 1 primary (Corridor A)
    "route_v1_primary": RouteCandidate(
        route_id="route_v1_primary",
        corridor_id="corridor_a_expressway",
        name="Expressway Route (Fastest Base)",
        vehicle_id="vehicle_1",
        stop_ids=["depot", "stop_1", "stop_2", "depot"],
        segments=[SEGMENT_V1_EXPRESSWAY, SEGMENT_V1_LAST_MILE],
        is_default=True,
    ),
    # Vehicle 1 alternative (Corridor B bypass)
    "route_v1_bypass": RouteCandidate(
        route_id="route_v1_bypass",
        corridor_id="corridor_b_bypass",
        name="Mehrauli Bypass (Congestion Alternate)",
        vehicle_id="vehicle_1",
        stop_ids=["depot", "stop_1", "stop_2", "depot"],
        segments=[SEGMENT_V1_BYPASS, SEGMENT_V1_LAST_MILE],
        is_default=False,
    ),
    # Vehicle 2 primary (Corridor C)
    "route_v2_primary": RouteCandidate(
        route_id="route_v2_primary",
        corridor_id="corridor_c_south",
        name="South Arterial Loop",
        vehicle_id="vehicle_2",
        stop_ids=["depot", "stop_3", "stop_4", "depot"],
        segments=[SEGMENT_V2_SOUTH],
        is_default=True,
    ),
}


class RouteCatalog:
    """Access point for prevalidated road candidates and segment costs."""

    def __init__(self, candidates: Mapping[str, RouteCandidate] | None = None) -> None:
        self._candidates = dict(candidates or ROUTE_CANDIDATES)

    def get_candidate(self, route_id: str) -> RouteCandidate | None:
        return self._candidates.get(route_id)

    def get_default_route(self, vehicle_id: str) -> RouteCandidate | None:
        for c in self._candidates.values():
            if c.vehicle_id == vehicle_id and c.is_default:
                return c
        return None

    def get_alternatives(self, vehicle_id: str) -> list[RouteCandidate]:
        return [c for c in self._candidates.values() if c.vehicle_id == vehicle_id]

    def score_route(
        self,
        candidate: RouteCandidate,
        overlays: Mapping[str, float] | None = None,
    ) -> float:
        """Calculate total estimated seconds including active incident overlays."""
        base_seconds = candidate.total_duration_seconds
        if not overlays:
            return base_seconds
        # Add overlay delays matching segment IDs
        incident_delay = 0.0
        for seg in candidate.segments:
            if seg.segment_id in overlays:
                incident_delay += overlays[seg.segment_id]
        return base_seconds + incident_delay
