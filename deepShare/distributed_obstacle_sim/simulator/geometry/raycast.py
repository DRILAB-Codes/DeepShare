import math
from typing import Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon


def _nearest_point_from_intersection(origin: np.ndarray, geom) -> Optional[np.ndarray]:
    """Return the nearest point in a shapely intersection geometry."""
    if geom.is_empty:
        return None

    candidates = []

    if isinstance(geom, Point):
        candidates.append(geom)
    elif geom.geom_type == "MultiPoint":
        candidates.extend(list(geom.geoms))
    elif geom.geom_type == "LineString":
        coords = list(geom.coords)
        candidates.extend(Point(c) for c in coords)
    elif geom.geom_type == "MultiLineString":
        for line in geom.geoms:
            coords = list(line.coords)
            candidates.extend(Point(c) for c in coords)
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            p = _nearest_point_from_intersection(origin, g)
            if p is not None:
                candidates.append(Point(float(p[0]), float(p[1])))

    if not candidates:
        return None

    ox, oy = origin
    nearest = min(candidates, key=lambda p: (p.x - ox) ** 2 + (p.y - oy) ** 2)
    return np.array([nearest.x, nearest.y], dtype=np.float32)


def cast_ray(
    origin: np.ndarray,
    angle: float,
    max_range: float,
    obstacle: Polygon,
) -> Tuple[float, Optional[np.ndarray]]:
    """Cast a single ray and return distance and hit point."""
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    end = origin + max_range * direction
    ray = LineString([tuple(origin), tuple(end)])

    inter = ray.intersection(obstacle.boundary)
    hit = _nearest_point_from_intersection(origin, inter)

    if hit is None:
        return float(max_range), None

    dist = float(np.linalg.norm(hit - origin))
    if dist > max_range:
        return float(max_range), None

    return dist, hit
