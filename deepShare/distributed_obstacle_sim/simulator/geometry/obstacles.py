import math
from typing import Tuple

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


def regular_polygon(
    n_sides: int,
    radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0,
) -> Polygon:
    angles = np.linspace(0, 2 * math.pi, n_sides, endpoint=False) + rotation
    points = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
    poly = Polygon(points)
    return translate(poly, xoff=center[0], yoff=center[1])


def circle_polygon(
    radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    resolution: int = 64,
) -> Polygon:
    return Point(center).buffer(radius, resolution=resolution)


def star_polygon(
    radius_outer: float = 1.0,
    radius_inner: float = 0.45,
    n_points: int = 5,
    center: Tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0,
) -> Polygon:
    points = []

    for i in range(n_points * 2):
        r = radius_outer if i % 2 == 0 else radius_inner
        a = rotation + i * math.pi / n_points
        points.append((r * math.cos(a), r * math.sin(a)))

    poly = Polygon(points)
    return translate(poly, xoff=center[0], yoff=center[1])


def rectangle(
    width: float = 2.0,
    height: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> Polygon:
    w, h = width / 2, height / 2

    points = [
        (-w, -h),
        (w, -h),
        (w, h),
        (-w, h),
    ]

    poly = Polygon(points)
    return translate(poly, xoff=center[0], yoff=center[1])


def cross_shape(
    arm: float = 0.5,
    length: float = 2.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> Polygon:
    a = arm / 2
    l = length / 2

    points = [
        (-a, -l), (a, -l), (a, -a),
        (l, -a), (l, a), (a, a),
        (a, l), (-a, l), (-a, a),
        (-l, a), (-l, -a), (-a, -a),
    ]

    poly = Polygon(points)
    return translate(poly, xoff=center[0], yoff=center[1])


def u_shape(
    outer_w: float = 2.5,
    outer_h: float = 2.0,
    thickness: float = 0.45,
    center: Tuple[float, float] = (0.0, 0.0),
) -> Polygon:
    w = outer_w / 2
    h = outer_h / 2
    t = thickness

    points = [
        (-w, -h), (w, -h), (w, h),
        (w - t, h), (w - t, -h + t),
        (-w + t, -h + t), (-w + t, h),
        (-w, h),
    ]

    poly = Polygon(points)
    return translate(poly, xoff=center[0], yoff=center[1])


# ============================================================
# 추가 1) random polygon
# ============================================================
def random_simple_polygon(
    radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    n_vertices: int = 8,
    jitter: float = 0.5,
    seed: int = None,
) -> Polygon:
    rng = np.random.default_rng(seed)

    angles = np.sort(
        rng.uniform(0, 2 * np.pi, size=n_vertices)
    )

    radii = radius * (
        1.0 + rng.uniform(-jitter, jitter, size=n_vertices)
    )

    pts = np.stack([
        center[0] + radii * np.cos(angles),
        center[1] + radii * np.sin(angles),
    ], axis=1)

    poly = Polygon(pts).buffer(0)

    if poly.geom_type == "MultiPolygon":
        poly = max(list(poly.geoms), key=lambda g: g.area)

    return Polygon(poly.exterior)


# ============================================================
# 추가 2) rect + polygon composite
# ============================================================
def rect_with_two_polys(
    scale: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    seed: int = None,
) -> Polygon:
    rng = np.random.default_rng(seed)

    cx, cy = center

    rw = rng.uniform(0.8, 2.0) * scale
    rh = rng.uniform(1.0, 3.0) * scale

    rect = box(
        cx - rw / 2,
        cy - rh / 2,
        cx + rw / 2,
        cy + rh / 2,
    )

    parts = [rect]

    for _ in range(2):
        r = rng.uniform(0.3, 1.5) * scale
        nv = int(rng.integers(4, 9))

        side = int(rng.integers(0, 4))

        if side == 0:
            px = cx - rw / 2 + rng.uniform(-0.8 * r, 0.3 * r)
            py = rng.uniform(cy - rh / 2, cy + rh / 2)

        elif side == 1:
            px = cx + rw / 2 + rng.uniform(-0.3 * r, 0.8 * r)
            py = rng.uniform(cy - rh / 2, cy + rh / 2)

        elif side == 2:
            px = rng.uniform(cx - rw / 2, cx + rw / 2)
            py = cy - rh / 2 + rng.uniform(-0.8 * r, 0.3 * r)

        else:
            px = rng.uniform(cx - rw / 2, cx + rw / 2)
            py = cy + rh / 2 + rng.uniform(-0.3 * r, 0.8 * r)

        poly = random_simple_polygon(
            radius=r,
            center=(px, py),
            n_vertices=nv,
            jitter=0.4,
            seed=int(rng.integers(0, 1_000_000)),
        )

        parts.append(poly)

    obstacle = unary_union(parts).buffer(0)

    if obstacle.geom_type == "MultiPolygon":
        obstacle = max(list(obstacle.geoms), key=lambda g: g.area)

    return Polygon(obstacle.exterior)


# ============================================================
# factory
# ============================================================
def create_obstacle(
    shape_type: str = "circle",
    center=(0.0, 0.0),
    scale: float = 1.0,
    seed: int = None,
) -> Polygon:

    shape_type = shape_type.lower()

    if shape_type == "circle":
        return circle_polygon(
            radius=scale,
            center=center,
        )

    if shape_type == "triangle":
        return regular_polygon(
            3,
            radius=scale,
            center=center,
            rotation=math.pi / 2,
        )

    if shape_type == "rectangle":
        return rectangle(
            width=2.0 * scale,
            height=1.2 * scale,
            center=center,
        )

    if shape_type == "pentagon":
        return regular_polygon(
            5,
            radius=scale,
            center=center,
            rotation=math.pi / 2,
        )

    if shape_type == "star":
        return star_polygon(
            radius_outer=scale,
            radius_inner=0.45 * scale,
            center=center,
            rotation=math.pi / 2,
        )

    if shape_type == "cross":
        return cross_shape(
            arm=0.6 * scale,
            length=2.2 * scale,
            center=center,
        )

    if shape_type == "u":
        return u_shape(
            outer_w=2.5 * scale,
            outer_h=2.0 * scale,
            thickness=0.45 * scale,
            center=center,
        )

    # =========================
    # 추가된 shape
    # =========================
    if shape_type == "random_polygon":
        return random_simple_polygon(
            radius=scale,
            center=center,
            n_vertices=np.random.randint(5, 11),
            jitter=0.45,
            seed=seed,
        )

    if shape_type == "rect_with_two_polys":
        return rect_with_two_polys(
            scale=scale,
            center=center,
            seed=seed,
        )

    raise ValueError(f"Unknown obstacle shape_type: {shape_type}")