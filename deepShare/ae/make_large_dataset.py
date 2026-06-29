import os
import sys
import json
import random
import numpy as np
from pathlib import Path

# 시뮬레이션 루트
SIM_ROOT = "../distributed_obstacle_sim"
sys.path.append(SIM_ROOT)

from simulator.dataset.sample_generator import make_single_sample, save_sample_json


# =========================
# 설정
# =========================
TRAIN_DIR = "data/ae/train_mixed"
VAL_DIR = "data/ae/val_mixed"
TEST_DIR = "data/ae/test_mixed"

OBSTACLE_LIB_ROOT = Path("../dataset/obstacle_library/modelnet10_projected")

NUM_TRAIN = 5000
NUM_VAL = 500

BOUNDARY_POINTS = 256
TEST_SCALES = [0.6, 1.0, 1.4, 2.0]

BASE_SHAPES = [
    "star",
    "triangle",
    "circle",
    "cross",
    "u",
    "pentagon",
    "rectangle",
]

BASE_SHAPE_WEIGHTS = {
    "star": 1,
    "triangle": 1,
    "circle": 1,
    "cross": 1,
    "u": 1,
    "pentagon": 1,
    "rectangle": 1,
}

# 단순 / 데이터셋 비중 조절
SOURCE_WEIGHTS = {
    "base_shape": 1,
    "projected": 4,
}


def sample_polygon_boundary(polygon, num_points=256):
    """
    polygon: [[x, y], [x, y], ...]
    return: [[x, y, 0], ...]
    """
    pts = np.asarray(polygon, dtype=np.float32)

    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.concatenate([pts, pts[:1]], axis=0)

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())

    if total <= 1e-8:
        raise ValueError("zero-length polygon")

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    samples = []

    for d in np.linspace(0.0, total, num_points, endpoint=False, dtype=np.float32):
        i = np.searchsorted(cum, d, side="right") - 1
        i = min(i, len(seg_len) - 1)

        t = (d - cum[i]) / max(seg_len[i], 1e-8)
        p = pts[i] + t * seg[i]
        samples.append([float(p[0]), float(p[1]), 0.0])

    return samples

def choose_source():
    return random.choices(
        list(SOURCE_WEIGHTS.keys()),
        weights=list(SOURCE_WEIGHTS.values()),
        k=1,
    )[0]


def choose_base_shape():
    return random.choices(
        BASE_SHAPES,
        weights=[BASE_SHAPE_WEIGHTS[s] for s in BASE_SHAPES],
        k=1,
    )[0]

def transform_points_3d(points, scale=1.0, center=(5.0, 5.0)):
    out = []

    cx, cy = center

    for p in points:
        x = float(p[0]) * scale + cx
        y = float(p[1]) * scale + cy
        z = float(p[2]) if len(p) > 2 else 0.0
        out.append([x, y, z])

    return out

def load_projected_obstacle_files(split):
    root = OBSTACLE_LIB_ROOT / split
    files = sorted(root.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"No projected obstacle files in {root}")

    by_category = {}

    for path in files:
        category = path.stem.split("_", 1)[-1]
        by_category.setdefault(category, []).append(path)

    return files, by_category

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Make Train/Validation dataset
def generate_sample(i, save_dir, projected_files):
    source = choose_source()

    n_robots = random.randint(6, 16)
    sensor_range = random.uniform(3.0, 6.0)
    num_rays = random.choice([32, 64, 128])
    comm_range = random.uniform(2.0, 3.5)

    # 형태 학습 중심이면 scale 고정 추천
    obstacle_scale = 1.0

    if source == "base_shape":
        shape_type = choose_base_shape()

        _, _, _, _, sample = make_single_sample(
            shape_type=shape_type,
            obstacle_scale=obstacle_scale,
            n_robots=n_robots,
            sensor_range=sensor_range,
            num_rays=num_rays,
            comm_range=comm_range,
        )

        polygon = sample["obstacle"]["polygon"]
        boundary_points = sample_polygon_boundary(
            polygon,
            num_points=BOUNDARY_POINTS,
        )

        sample["obstacle"]["boundary_points"] = boundary_points
        sample["obstacle"]["obstacle_scale"] = float(obstacle_scale)
        sample["obstacle"]["shape_type"] = shape_type
        sample["obstacle"]["source_type"] = "base_shape"

    else:
        obstacle_path = random.choice(projected_files)
        obstacle_data = load_json(obstacle_path)

        shape_type = f"projected_{obstacle_data['category']}"

        _, _, _, _, sample = make_single_sample(
            shape_type=shape_type,
            external_polygon=obstacle_data["polygon"],
            external_meta={
                "source_type": "projected",
                "category": obstacle_data["category"],
                "source_file": obstacle_data["source_file"],
                "projection": obstacle_data["projection"],
                "library_path": str(obstacle_path),
            },
            obstacle_scale=obstacle_scale,
            n_robots=n_robots,
            sensor_range=sensor_range,
            num_rays=num_rays,
            comm_range=comm_range,
        )

        sample["obstacle"]["boundary_points"] = transform_points_3d(
            obstacle_data["boundary_points"],
            scale=obstacle_scale,
            center=(5.0, 5.0),
        )

        sample["obstacle"]["surface_points_2d"] = transform_points_3d(
            obstacle_data["surface_points_2d"],
            scale=obstacle_scale,
            center=(5.0, 5.0),
        )

        sample["obstacle"]["shape_type"] = shape_type
        sample["obstacle"]["source_type"] = "projected"
        sample["obstacle"]["category"] = obstacle_data["category"]

    save_path = os.path.join(save_dir, f"sample_{i:05d}.json")
    save_sample_json(sample, save_path)

# Make Test Dataset
def generate_fixed_base_sample(i, save_dir, shape_type):
    n_robots = 10
    sensor_range = 5.0
    num_rays = 64
    comm_range = 3.0
    obstacle_scale = 1.0

    _, _, _, _, sample = make_single_sample(
        shape_type=shape_type,
        obstacle_scale=obstacle_scale,
        n_robots=n_robots,
        sensor_range=sensor_range,
        num_rays=num_rays,
        comm_range=comm_range,
    )

    polygon = sample["obstacle"]["polygon"]
    boundary_points = sample_polygon_boundary(
        polygon,
        num_points=BOUNDARY_POINTS,
    )

    sample["obstacle"]["boundary_points"] = boundary_points
    sample["obstacle"]["obstacle_scale"] = float(obstacle_scale)
    sample["obstacle"]["shape_type"] = shape_type
    sample["obstacle"]["source_type"] = "base_shape"

    save_path = os.path.join(save_dir, f"test_{i:05d}_{shape_type}.json")
    save_sample_json(sample, save_path)

def generate_fixed_projected_sample(i, save_dir, obstacle_path):
    n_robots = 10
    sensor_range = 5.0
    num_rays = 64
    comm_range = 3.0
    obstacle_scale = 1.0

    obstacle_data = load_json(obstacle_path)
    shape_type = f"projected_{obstacle_data['category']}"

    _, _, _, _, sample = make_single_sample(
        shape_type=shape_type,
        external_polygon=obstacle_data["polygon"],
        external_meta={
            "source_type": "projected",
            "category": obstacle_data["category"],
            "source_file": obstacle_data["source_file"],
            "projection": obstacle_data["projection"],
            "library_path": str(obstacle_path),
        },
        obstacle_scale=obstacle_scale,
        n_robots=n_robots,
        sensor_range=sensor_range,
        num_rays=num_rays,
        comm_range=comm_range,
    )

    sample["obstacle"]["boundary_points"] = transform_points_3d(
        obstacle_data["boundary_points"],
        scale=obstacle_scale,
        center=(5.0, 5.0),
    )

    sample["obstacle"]["surface_points_2d"] = transform_points_3d(
        obstacle_data["surface_points_2d"],
        scale=obstacle_scale,
        center=(5.0, 5.0),
    )

    sample["obstacle"]["shape_type"] = shape_type
    sample["obstacle"]["source_type"] = "projected"
    sample["obstacle"]["category"] = obstacle_data["category"]

    save_path = os.path.join(
        save_dir,
        f"test_{i:05d}_{shape_type}_{obstacle_path.stem}.json",
    )

    save_sample_json(sample, save_path)


def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    train_projected_files, _ = load_projected_obstacle_files("train")
    test_projected_files, test_by_category = load_projected_obstacle_files("test")

    print("Generating TRAIN dataset...")
    for i in range(NUM_TRAIN):
        generate_sample(i, TRAIN_DIR, train_projected_files)
        if i % 100 == 0:
            print(f"[Train] {i}/{NUM_TRAIN}")

    print("Generating VAL dataset...")
    for i in range(NUM_VAL):
        generate_sample(i, VAL_DIR, train_projected_files)
        if i % 50 == 0:
            print(f"[Val] {i}/{NUM_VAL}")

    print("Generating TEST dataset...")
    idx = 0

    # 1. 기존 고정 도형 test
    for shape_type in BASE_SHAPES:
        generate_fixed_base_sample(idx, TEST_DIR, shape_type)
        print(f"[Test] {idx}: base_shape={shape_type}")
        idx += 1

    # 2. 투영 데이터 클래스별 test
    for category, files in sorted(test_by_category.items()):
        obstacle_path = random.choice(files)
        generate_fixed_projected_sample(idx, TEST_DIR, obstacle_path)
        print(f"[Test] {idx}: projected_category={category}, file={obstacle_path.name}")
        idx += 1

    print("Done.")


if __name__ == "__main__":
    main()