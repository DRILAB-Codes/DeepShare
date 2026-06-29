import os
import sys
import random
import numpy as np

# =========================
# 🔴 여기만 수정하면 됨
# =========================
SIM_ROOT = "../distributed_obstacle_sim"

sys.path.append(SIM_ROOT)

from simulator.dataset.sample_generator import make_single_sample, save_sample_json


# =========================
# 설정
# =========================
TRAIN_DIR = "data/ae/train"
VAL_DIR = "data/ae/val"
TEST_DIR = "data/ae/test"

NUM_TRAIN = 1000
NUM_VAL = 200

TEST_SCALES = [0.6, 1.0, 1.4, 2.0]

# simulator.geometry.obstacles.create_obstacle 이 지원하는 이름이어야 함
SHAPES = [
    "star",
    "triangle",
    "circle",
    "cross",
    "u",
    "random_polygon",
    "rect_with_two_polys",
    "pentagon",
    "rectangle",
]

BOUNDARY_POINTS = 256


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


def generate_sample(i, save_dir):
    shape_type = random.choice(SHAPES)

    # 장애물 크기 다양화
    obstacle_scale = random.uniform(0.6, 1.8)

    # 로봇 배치도 약간 다양화
    n_robots = random.randint(6, 16)
    sensor_range = random.uniform(3.0, 6.0)
    num_rays = random.choice([32, 64, 128])
    comm_range = random.uniform(2.0, 3.5)

    _, _, _, _, sample = make_single_sample(
        shape_type=shape_type,
        obstacle_scale=obstacle_scale,
        n_robots=n_robots,
        sensor_range=sensor_range,
        num_rays=num_rays,
        comm_range=comm_range,
    )

    # =========================
    # 핵심 추가:
    # 로봇 관측이 아니라 장애물 전체 외곽 포인트 클라우드 저장
    # =========================
    polygon = sample["obstacle"]["polygon"]
    boundary_points = sample_polygon_boundary(
        polygon,
        num_points=BOUNDARY_POINTS,
    )

    sample["obstacle"]["boundary_points"] = boundary_points
    sample["obstacle"]["obstacle_scale"] = float(obstacle_scale)
    sample["obstacle"]["shape_type"] = shape_type

    save_path = os.path.join(save_dir, f"sample_{i:05d}.json")
    save_sample_json(sample, save_path)

def generate_fixed_sample(i, save_dir, shape_type, obstacle_scale):
    n_robots = 10
    sensor_range = 5.0
    num_rays = 64
    comm_range = 3.0

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

    save_path = os.path.join(
        save_dir,
        f"test_{i:05d}_{shape_type}_scale_{obstacle_scale:.1f}.json"
    )
    save_sample_json(sample, save_path)

def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    print("Generating TRAIN dataset...")
    for i in range(NUM_TRAIN):
        generate_sample(i, TRAIN_DIR)
        if i % 100 == 0:
            print(f"[Train] {i}/{NUM_TRAIN}")

    print("Generating VAL dataset...")
    for i in range(NUM_VAL):
        generate_sample(i, VAL_DIR)
        if i % 50 == 0:
            print(f"[Val] {i}/{NUM_VAL}")

    print("Generating TEST dataset...")
    idx = 0
    for shape_type in SHAPES:
        for scale in TEST_SCALES:
            generate_fixed_sample(idx, TEST_DIR, shape_type, scale)
            print(f"[Test] {idx}: shape={shape_type}, scale={scale}")
            idx += 1

    print("Done.")


if __name__ == "__main__":
    main()