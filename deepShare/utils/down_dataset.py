from pathlib import Path
import json
import numpy as np
import trimesh

from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import find_contours
from shapely.geometry import Polygon


# =========================
# 설정
# =========================
MODELNET10_ROOT = Path("../ae/data/ModelNet10")
OUT_ROOT = Path("data/modelnet10_2d")

SPLITS = ["train", "test"]

N_SURFACE_POINTS = 20000
N_BOUNDARY_POINTS = 256

PROJECTION = "xy"  # "xy", "xz", "yz"

GRID_SIZE = 512
POINT_RADIUS = 2
CLOSING_RADIUS = 8
OPENING_RADIUS = 2


# =========================
# 기본 함수
# =========================
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

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)

    center = (min_xy + max_xy) / 2.0
    extent = (max_xy - min_xy).max() * (1.0 + pad)

    pts01 = (points - center) / (extent + 1e-8) + 0.5
    pix = np.round(pts01 * (grid_size - 1)).astype(np.int32)
    pix = np.clip(pix, 0, grid_size - 1)

    occ = np.zeros((grid_size, grid_size), dtype=bool)

    r = point_radius
    for x, y in pix:
        x0, x1 = max(0, x - r), min(grid_size, x + r + 1)
        y0, y1 = max(0, y - r), min(grid_size, y + r + 1)
        occ[y0:y1, x0:x1] = True

    transform = {
        "center": center.astype(np.float32),
        "extent": float(extent),
        "grid_size": int(grid_size),
    }

    return occ, transform


def occupancy_to_polygon(occ, transform):
    occ = binary_closing(occ, disk(CLOSING_RADIUS))
    occ = binary_fill_holes(occ)

    if OPENING_RADIUS > 0:
        occ = binary_opening(occ, disk(OPENING_RADIUS))

    contours = find_contours(occ.astype(float), level=0.5)

    if not contours:
        raise RuntimeError("No contour found")

    contour = max(contours, key=len)

    # contour: [row, col] = [y, x]
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

    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    if poly.is_empty:
        raise RuntimeError("Empty polygon")

    return poly, occ


def sample_boundary_points(poly, n_points=256):
    boundary = poly.boundary

    if boundary.geom_type == "MultiLineString":
        boundary = max(boundary.geoms, key=lambda g: g.length)

    distances = np.linspace(
        0.0,
        boundary.length,
        n_points,
        endpoint=False,
        dtype=np.float32,
    )

    pts = []
    for d in distances:
        p = boundary.interpolate(float(d))
        x, y = p.coords[0]
        pts.append([x, y])

    return np.asarray(pts, dtype=np.float32)


def polygon_to_array(poly):
    return np.asarray(poly.exterior.coords, dtype=np.float32)


def convert_single_off(off_path, category, split):
    mesh = trimesh.load(off_path, force="mesh")

    surface_points, _ = trimesh.sample.sample_surface(
        mesh,
        N_SURFACE_POINTS,
    )

    projected_points = project_points(surface_points, PROJECTION)

    occ, transform = points_to_occupancy(
        projected_points,
        grid_size=GRID_SIZE,
        point_radius=POINT_RADIUS,
    )

    poly, occ_processed = occupancy_to_polygon(occ, transform)

    boundary_points = sample_boundary_points(
        poly,
        n_points=N_BOUNDARY_POINTS,
    )

    boundary_points_3d = np.concatenate(
        [
            boundary_points,
            np.zeros((len(boundary_points), 1), dtype=np.float32),
        ],
        axis=1,
    )

    polygon = polygon_to_array(poly)

    return {
        "projected_points": projected_points,
        "boundary_points": boundary_points,
        "boundary_points_3d": boundary_points_3d,
        "polygon": polygon,
        "category": category,
        "split": split,
        "source_file": str(off_path),
        "projection": PROJECTION,
        "grid_size": GRID_SIZE,
        "point_radius": POINT_RADIUS,
        "closing_radius": CLOSING_RADIUS,
        "opening_radius": OPENING_RADIUS,
    }


def save_npz(data, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        projected_points=data["projected_points"],
        boundary_points=data["boundary_points"],
        boundary_points_3d=data["boundary_points_3d"],
        polygon=data["polygon"],
        category=np.array(data["category"]),
        split=np.array(data["split"]),
        source_file=np.array(data["source_file"]),
        projection=np.array(data["projection"]),
        grid_size=np.array(data["grid_size"]),
        point_radius=np.array(data["point_radius"]),
        closing_radius=np.array(data["closing_radius"]),
        opening_radius=np.array(data["opening_radius"]),
    )


def collect_off_files(root, split):
    items = []

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        split_dir = category_dir / split

        if not split_dir.exists():
            continue

        for off_path in sorted(split_dir.glob("*.off")):
            items.append((off_path, category, split))

    return items


def main():
    index = []
    fail_log = []

    for split in SPLITS:
        items = collect_off_files(MODELNET10_ROOT, split)
        print(f"[{split}] OFF files: {len(items)}")

        for idx, (off_path, category, split) in enumerate(items):
            try:
                data = convert_single_off(off_path, category, split)

                stem = off_path.stem
                save_path = OUT_ROOT / split / category / f"{stem}.npz"
                save_npz(data, save_path)

                index.append({
                    "split": split,
                    "category": category,
                    "source_file": str(off_path),
                    "npz_file": str(save_path),
                    "projection": PROJECTION,
                    "n_projected_points": int(len(data["projected_points"])),
                    "n_boundary_points": int(len(data["boundary_points"])),
                })

            except Exception as e:
                fail_log.append({
                    "split": split,
                    "category": category,
                    "source_file": str(off_path),
                    "error": str(e),
                })
                print(f"[FAIL] {off_path} | {e}")

            if idx % 100 == 0:
                print(f"[{split}] {idx}/{len(items)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with open(OUT_ROOT / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    with open(OUT_ROOT / "fail_log.json", "w") as f:
        json.dump(fail_log, f, indent=2)

    print("Done.")
    print("Saved:", OUT_ROOT)
    print("Success:", len(index))
    print("Failed :", len(fail_log))


if __name__ == "__main__":
    main()