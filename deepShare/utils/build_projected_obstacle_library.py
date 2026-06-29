from pathlib import Path
import json
import random
import numpy as np
import trimesh

from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPoint


MODELNET10_ROOT = Path("../ae/data/ModelNet10")
OUT_DIR = Path("../dataset/obstacle_library/modelnet10_projected")

SPLITS = ["train", "test"]

N_SURFACE_POINTS = 20000
N_SAVE_SURFACE_POINTS = 2048
N_BOUNDARY_POINTS = 256

PROJECTION = "xy"

GRID_SIZE = 512
POINT_RADIUS = 2
CLOSING_RADIUS = 8
OPENING_RADIUS = 2


def project_points(points, mode="xy"):
    if mode == "xy":
        pts2d = points[:, [0, 1]]
    elif mode == "xz":
        pts2d = points[:, [0, 2]]
    elif mode == "yz":
        pts2d = points[:, [1, 2]]
    else:
        raise ValueError(mode)

    pts2d = pts2d - pts2d.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts2d, axis=1))
    pts2d = pts2d / (scale + 1e-8)
    return pts2d.astype(np.float32)


def points_to_occupancy(points, grid_size=512, point_radius=2):
    pad = 0.08
    pts = points.copy()

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)

    center = (min_xy + max_xy) / 2
    extent = (max_xy - min_xy).max() * (1 + pad)

    pts = (pts - center) / (extent + 1e-8) + 0.5
    pix = np.round(pts * (grid_size - 1)).astype(int)
    pix = np.clip(pix, 0, grid_size - 1)

    occ = np.zeros((grid_size, grid_size), dtype=bool)

    rr = point_radius
    for x, y in pix:
        x0, x1 = max(0, x - rr), min(grid_size, x + rr + 1)
        y0, y1 = max(0, y - rr), min(grid_size, y + rr + 1)
        occ[y0:y1, x0:x1] = True

    return occ, {
        "center": center,
        "extent": extent,
        "grid_size": grid_size,
    }


def occupancy_to_polygon(occ, transform):
    occ = binary_closing(occ, disk(CLOSING_RADIUS))
    occ = binary_fill_holes(occ)

    if OPENING_RADIUS > 0:
        occ = binary_opening(occ, disk(OPENING_RADIUS))

    contours = find_contours(occ.astype(float), level=0.5)

    if not contours:
        return MultiPoint([]).convex_hull

    contour = max(contours, key=len)

    y = contour[:, 0]
    x = contour[:, 1]

    grid_size = transform["grid_size"]
    center = transform["center"]
    extent = transform["extent"]

    pts01 = np.stack([x, y], axis=1) / (grid_size - 1)
    pts2d = (pts01 - 0.5) * extent + center

    poly = Polygon(pts2d)

    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly


def sample_boundary_points(poly, n_points=256):
    boundary = poly.boundary

    if boundary.geom_type == "MultiLineString":
        boundary = max(boundary.geoms, key=lambda g: g.length)

    distances = np.linspace(0, boundary.length, n_points, endpoint=False)

    return np.array([
        boundary.interpolate(d).coords[0]
        for d in distances
    ], dtype=np.float32)


def extract_projected_obstacle(off_path):
    mesh = trimesh.load(off_path, force="mesh")

    surface_points_3d, _ = trimesh.sample.sample_surface(
        mesh,
        N_SURFACE_POINTS,
    )

    surface_points_2d = project_points(surface_points_3d, PROJECTION)

    occ, transform = points_to_occupancy(
        surface_points_2d,
        grid_size=GRID_SIZE,
        point_radius=POINT_RADIUS,
    )

    poly = occupancy_to_polygon(occ, transform)

    if poly.is_empty or poly.area <= 1e-8:
        raise ValueError(f"Invalid polygon: {off_path}")

    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    boundary_points = sample_boundary_points(
        poly,
        n_points=N_BOUNDARY_POINTS,
    )

    polygon = np.asarray(poly.exterior.coords[:-1], dtype=np.float32)

    if len(surface_points_2d) > N_SAVE_SURFACE_POINTS:
        idx = np.random.choice(
            len(surface_points_2d),
            N_SAVE_SURFACE_POINTS,
            replace=False,
        )
        surface_points_2d = surface_points_2d[idx]

    return {
        "source_file": str(off_path),
        "category": off_path.parent.parent.name,
        "projection": PROJECTION,
        "surface_points_2d": [
            [float(x), float(y), 0.0]
            for x, y in surface_points_2d
        ],
        "polygon": [
            [float(x), float(y)]
            for x, y in polygon
        ],
        "boundary_points": [
            [float(x), float(y), 0.0]
            for x, y in boundary_points
        ],
    }


def main():
    random.seed(0)
    np.random.seed(0)

    for split in SPLITS:
        out_split_dir = OUT_DIR / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        categories = [p for p in MODELNET10_ROOT.iterdir() if p.is_dir()]

        idx = 0
        for category_dir in categories:
            off_files = sorted((category_dir / split).glob("*.off"))

            for off_path in off_files:
                try:
                    obstacle = extract_projected_obstacle(off_path)
                except Exception as e:
                    print(f"[Skip] {off_path}: {e}")
                    continue

                save_path = out_split_dir / f"{idx:06d}_{category_dir.name}.json"

                with open(save_path, "w") as f:
                    json.dump(obstacle, f)

                if idx % 100 == 0:
                    print(f"[{split}] {idx}: {save_path}")

                idx += 1

    print("Done.")


if __name__ == "__main__":
    main()