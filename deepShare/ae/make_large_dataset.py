# =========================
# 데이터셋 생성 코드
# 기본 도형 + ModelNet 투영 도형에 대해 주변에 장애물 배치하고 라이다 관측 결과와 통신 이웃 저장 (시뮬레이터)
# 추가로 AE용 장애물 외곽 점 256개
# 학습용 5000 검증용 500
# 각 형태, 크기, 로봇 수, 센서 범위 등에서 랜덤 생성
# =========================
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
# 장애물 반경 2.5 이내 (월드 반경 5 - 로봇 배치 2.5)
# 센서 범위 2.5(로봇 배치 마진 거리) ~ 2.5 * 장애물 반경 (최대 장애물 중심까지는 볼 수 있게)
# 통신 범위 2(반경 + 2.5)sin(pi/로봇 수 + 6) 기준 2배 이내 (끊어지지 않게 + 최대 2홉까지 연결 가능성)
# =========================
TRAIN_DIR = "data/ae/train"
VAL_DIR = "data/ae/val"
TEST_DIR = "data/ae/test"

OBSTACLE_LIB_ROOT = Path("../dataset/obstacle_library/modelnet10_projected")

NUM_TRAIN = 5000
NUM_VAL = 500

BOUNDARY_POINTS = 256

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

# 데이터셋 재시도 3회 이후 설정 바꿔서 5회 이상 실패 시 생성 조건 수정 필요
MAX_PLACEMENT_RETRIES = 3
MAX_SETTING_RETRIES = 5



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

    if source == "base_shape":
        shape_type = choose_base_shape()

        sample_kwargs = {
            "shape_type": shape_type,
        }

        obstacle_data = None

    else:
        obstacle_path = random.choice(projected_files)
        obstacle_data = load_json(obstacle_path)

        shape_type = f"projected_{obstacle_data['category']}"

        sample_kwargs = {
            "shape_type": shape_type,
            "external_polygon": obstacle_data["polygon"],
            "external_meta": {
                "source_type": "projected",
                "category": obstacle_data["category"],
                "source_file": obstacle_data["source_file"],
                "projection": obstacle_data["projection"],
                "library_path": str(obstacle_path),
            },
        }

    sample = None
    used_obstacle_scale = None

    # 전체 생성 시도 > 이거 실패 시 생성 조건에 문제 있음
    for setting_attempt in range(MAX_SETTING_RETRIES):

        obstacle_scale = random.uniform(0.8, 2.5)
        n_robots = random.randint(6, 12)
        sensor_range = random.uniform(2.5, obstacle_scale + 2.5)
        num_rays = random.choice([32, 64, 128])
        safe_comm_range = (2.0 * (obstacle_scale + 2.5) * np.sin(np.pi/n_robots + np.deg2rad(6.0)))
        comm_range = random.uniform(safe_comm_range, 2.0 * safe_comm_range)

        # 생성 조건에 대해 위치 바꿔가며 시도
        for placement_attempt in range(MAX_PLACEMENT_RETRIES):

            (_, _, _, _, candidate, success,) = make_single_sample(
                **sample_kwargs,
                obstacle_scale=obstacle_scale,
                n_robots=n_robots,
                sensor_range=sensor_range,
                num_rays=num_rays,
                comm_range=comm_range,
            )

            if success:
                sample = candidate
                used_obstacle_scale = obstacle_scale
                break

        # 현재 setting에서 성공했으면 종료
        if sample is not None:
            break

    # 모두 생성 실패했으면 실패 반환
    if sample is None:
        print(
            f"[Generation Failed] "
            f"sample={i}, "
            f"source={source}, "
            f"shape={shape_type}"
        )
        return False

    # =========================================================
    # 성공한 sample 후처리
    # =========================================================

    polygon = sample["obstacle"]["polygon"]

    boundary_points = sample_polygon_boundary(
        polygon,
        num_points=BOUNDARY_POINTS,
    )

    sample["obstacle"]["boundary_points"] = boundary_points
    sample["obstacle"]["obstacle_scale"] = float(
        used_obstacle_scale
    )
    sample["obstacle"]["shape_type"] = shape_type
    sample["obstacle"]["source_type"] = source

    if source == "projected":
        sample["obstacle"]["category"] = obstacle_data["category"]

    save_path = os.path.join(
        save_dir,
        f"sample_{i:05d}.json",
    )

    save_sample_json(sample, save_path)

    return True

# Make Test Dataset
def generate_fixed_base_sample(i, save_dir, shape_type):

    sample = None
    used_obstacle_scale = None

    # 설정을 바꿔가며 최대 MAX_SETTING_RETRIES회 시도
    for setting_attempt in range(MAX_SETTING_RETRIES):

        obstacle_scale = random.uniform(0.8, 2.5)
        n_robots = random.randint(6, 12)
        sensor_range = random.uniform(
            2.5,
            obstacle_scale + 2.5,
        )
        num_rays = random.choice([32, 64, 128])

        safe_comm_range = (
            2.0
            * (obstacle_scale + 2.5)
            * np.sin(
                np.pi / n_robots
                + np.deg2rad(6.0)
            )
        )

        comm_range = random.uniform(
            safe_comm_range,
            2.0 * safe_comm_range,
        )

        # 같은 설정에서 robot 위치 noise만 바꿔가며 재시도
        for placement_attempt in range(MAX_PLACEMENT_RETRIES):
            (
                _,
                _,
                _,
                _,
                candidate,
                success,
            ) = make_single_sample(
                shape_type=shape_type,
                obstacle_scale=obstacle_scale,
                n_robots=n_robots,
                sensor_range=sensor_range,
                num_rays=num_rays,
                comm_range=comm_range,
            )

            if success:
                sample = candidate
                used_obstacle_scale = obstacle_scale
                break

        # 현재 설정에서 성공했으면 setting retry 종료
        if sample is not None:
            break

    # 모든 설정에서 실패
    if sample is None:
        print(
            f"[Test Generation Failed] "
            f"base_shape={shape_type}"
        )
        return False

    # 실제 simulation polygon에서 AE target boundary 생성
    polygon = sample["obstacle"]["polygon"]

    boundary_points = sample_polygon_boundary(
        polygon,
        num_points=BOUNDARY_POINTS,
    )

    sample["obstacle"]["boundary_points"] = boundary_points
    sample["obstacle"]["obstacle_scale"] = float(
        used_obstacle_scale
    )
    sample["obstacle"]["shape_type"] = shape_type
    sample["obstacle"]["source_type"] = "base_shape"

    save_path = os.path.join(
        save_dir,
        f"test_{i:05d}_{shape_type}.json",
    )

    save_sample_json(sample, save_path)

    return True

def generate_fixed_projected_sample(
    i,
    save_dir,
    obstacle_path,
):

    obstacle_data = load_json(obstacle_path)

    shape_type = (
        f"projected_{obstacle_data['category']}"
    )

    sample = None
    used_obstacle_scale = None

    # 설정을 바꿔가며 최대 MAX_SETTING_RETRIES회 시도
    for setting_attempt in range(MAX_SETTING_RETRIES):

        obstacle_scale = random.uniform(0.8, 2.5)
        n_robots = random.randint(6, 12)
        sensor_range = random.uniform(
            2.5,
            obstacle_scale + 2.5,
        )
        num_rays = random.choice([32, 64, 128])

        safe_comm_range = (
            2.0
            * (obstacle_scale + 2.5)
            * np.sin(
                np.pi / n_robots
                + np.deg2rad(6.0)
            )
        )

        comm_range = random.uniform(
            safe_comm_range,
            2.0 * safe_comm_range,
        )

        # 같은 설정에서 robot 위치 noise만 바꿔가며 재시도
        for placement_attempt in range(MAX_PLACEMENT_RETRIES):
            (
                _,
                _,
                _,
                _,
                candidate,
                success,
            ) = make_single_sample(
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

            if success:
                sample = candidate
                used_obstacle_scale = obstacle_scale
                break

        # 현재 설정에서 성공했으면 setting retry 종료
        if sample is not None:
            break

    # 모든 설정에서 실패
    if sample is None:
        print(
            f"[Test Generation Failed] "
            f"projected_category={obstacle_data['category']}, "
            f"file={obstacle_path.name}"
        )
        return False

    # 실제 simulation polygon에서 AE target boundary 생성
    polygon = sample["obstacle"]["polygon"]

    boundary_points = sample_polygon_boundary(
        polygon,
        num_points=BOUNDARY_POINTS,
    )

    sample["obstacle"]["boundary_points"] = boundary_points
    sample["obstacle"]["obstacle_scale"] = float(
        used_obstacle_scale
    )
    sample["obstacle"]["shape_type"] = shape_type
    sample["obstacle"]["source_type"] = "projected"
    sample["obstacle"]["category"] = obstacle_data["category"]

    save_path = os.path.join(
        save_dir,
        f"test_{i:05d}_{shape_type}_{obstacle_path.stem}.json",
    )

    save_sample_json(sample, save_path)

    return True


def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    train_projected_files, _ = load_projected_obstacle_files("train")
    _, test_by_category = load_projected_obstacle_files("test")

    # Train
    print("Generating TRAIN dataset...")
    for i in range(NUM_TRAIN):
        success = generate_sample(
            i,
            TRAIN_DIR,
            train_projected_files,
        )

        if not success:
            print(
                f"[Train Generation Failed] "
                f"sample={i}. Stop dataset generation."
            )
            return

        if (i + 1) % 100 == 0:
            print(f"[Train] {i + 1}/{NUM_TRAIN}")

    # VALIDATION
    print("Generating VAL dataset...")
    for i in range(NUM_VAL):
        success = generate_sample(
            i,
            VAL_DIR,
            train_projected_files,
        )

        if not success:
            print(
                f"[Val Generation Failed] "
                f"sample={i}. Stop dataset generation."
            )
            return

        if (i + 1) % 50 == 0:
            print(f"[Val] {i + 1}/{NUM_VAL}")

    # TEST
    print("Generating TEST dataset...")
    idx = 0
    # 1. 기본 도형 test
    for shape_type in BASE_SHAPES:
        success = generate_fixed_base_sample(
            idx,
            TEST_DIR,
            shape_type,
        )

        if not success:
            print(
                f"[Test Generation Failed] "
                f"base_shape={shape_type}. "
                f"Stop dataset generation."
            )
            return

        print(
            f"[Test] {idx}: "
            f"base_shape={shape_type}"
        )

        idx += 1

    # 2. projected category별 test
    for category, files in sorted(test_by_category.items()):
        obstacle_path = random.choice(files)

        success = generate_fixed_projected_sample(
            idx,
            TEST_DIR,
            obstacle_path,
        )

        if not success:
            print(
                f"[Test Generation Failed] "
                f"category={category}, "
                f"file={obstacle_path.name}. "
                f"Stop dataset generation."
            )
            return

        print(
            f"[Test] {idx}: "
            f"projected_category={category}, "
            f"file={obstacle_path.name}"
        )
        idx += 1

    print("Done.")


if __name__ == "__main__":
    main()