from pathlib import Path
import random
import numpy as np
import trimesh
import matplotlib.pyplot as plt

from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPoint


# =========================
# 설정
# =========================
MODELNET10_ROOT = Path("../ae/data/ModelNet10")
CATEGORY = None
SPLIT = "train"

N_SURFACE_POINTS = 20000
N_BOUNDARY_POINTS = 512

PROJECTION = "xy"

GRID_SIZE = 512
POINT_RADIUS = 2
CLOSING_RADIUS = 8
OPENING_RADIUS = 2


# =========================
# 함수
# =========================
def load_random_off(root, category=None, split="train"):
    if category is None:
        categories = [p.name for p in root.iterdir() if p.is_dir()]
        category = random.choice(categories)

    files = list((root / category / split).glob("*.off"))
    if not files:
        raise FileNotFoundError(f"No OFF files in {root / category / split}")

    return random.choice(files), category


def project_points(points, mode="xz"):
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
    return pts2d


def points_to_occupancy(points, grid_size=512, point_radius=2):
    """
    normalized [-1, 1] 근처 2D points를 occupancy grid로 변환
    """
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

    transform = {
        "center": center,
        "extent": extent,
        "grid_size": grid_size,
    }

    return occ, transform


def occupancy_to_polygon(
    occ,
    transform,
    closing_radius=8,
    opening_radius=2,
):
    """
    occupancy grid에서 가장 큰 외곽 contour를 polygon으로 변환
    """
    occ = binary_closing(occ, disk(closing_radius))
    occ = binary_fill_holes(occ)

    if opening_radius > 0:
        occ = binary_opening(occ, disk(opening_radius))

    contours = find_contours(occ.astype(float), level=0.5)

    if not contours:
        return MultiPoint([]).convex_hull, occ

    contour = max(contours, key=len)

    # contour는 [row, col] = [y, x]
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

    return poly, occ


def sample_boundary_points(poly, n_points=512):
    boundary = poly.boundary

    if boundary.geom_type == "MultiLineString":
        boundary = max(boundary.geoms, key=lambda g: g.length)

    distances = np.linspace(0, boundary.length, n_points, endpoint=False)

    pts = np.array([
        boundary.interpolate(d).coords[0]
        for d in distances
    ])

    return pts


def extract_boundary_by_raster(points_2d):
    occ, transform = points_to_occupancy(
        points_2d,
        grid_size=GRID_SIZE,
        point_radius=POINT_RADIUS,
    )

    poly, occ_processed = occupancy_to_polygon(
        occ,
        transform,
        closing_radius=CLOSING_RADIUS,
        opening_radius=OPENING_RADIUS,
    )

    boundary_points = sample_boundary_points(
        poly,
        n_points=N_BOUNDARY_POINTS,
    )

    return poly, boundary_points, occ_processed


def visualize(surface_2d, boundary_pts, poly, title):
    plt.figure(figsize=(7, 7))

    plt.scatter(
        surface_2d[:, 0],
        surface_2d[:, 1],
        s=1,
        alpha=0.15,
        label="Projected surface points"
    )

    if poly.geom_type == "Polygon":
        x, y = poly.exterior.xy
        plt.plot(x, y, linewidth=2, label="Raster contour boundary")

    plt.scatter(
        boundary_pts[:, 0],
        boundary_pts[:, 1],
        s=8,
        label="Boundary point cloud"
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()


# =========================
# 실행
# =========================
for i in range(10):
    off_path, category = load_random_off(
        MODELNET10_ROOT,
        category=CATEGORY,
        split=SPLIT
    )

    print("Loaded:", off_path)

    mesh = trimesh.load(off_path, force="mesh")
    surface_points, _ = trimesh.sample.sample_surface(
        mesh,
        N_SURFACE_POINTS
    )

    points_2d = project_points(surface_points, PROJECTION)

    poly, boundary_points, occ_processed = extract_boundary_by_raster(points_2d)

    visualize(
        points_2d,
        boundary_points,
        poly,
        title=f"{category} | {off_path.name} | projection={PROJECTION}"
    )




