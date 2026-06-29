import numpy as np

from simulator.geometry.raycast import cast_ray


def world_to_local(points_world: np.ndarray, robot_pos: np.ndarray, theta: float) -> np.ndarray:
    """Transform world coordinates into the robot local frame."""
    if len(points_world) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    c, s = np.cos(theta), np.sin(theta)
    rot_T = np.array([[c, s], [-s, c]], dtype=np.float32)
    return (points_world - robot_pos) @ rot_T.T


def simulate_lidar(robot, obstacle, timestamp=0.0):
    """Simulate simple 2D LiDAR against one polygon obstacle.

    Output format follows a ROS LaserScan-like structure:
    - scan: real sensor-compatible fields
    - debug: simulation-only fields
    """
    angle_min = -robot.fov / 2.0
    angle_max = robot.fov / 2.0

    if robot.num_rays > 1:
        angle_increment = (angle_max - angle_min) / (robot.num_rays - 1)
    else:
        angle_increment = 0.0

    # Local ray angles relative to robot heading
    local_angles = angle_min + np.arange(robot.num_rays, dtype=np.float32) * angle_increment

    # World ray angles used for actual ray-casting
    world_angles = robot.theta + local_angles

    ranges = []
    hit_mask = []
    hit_points_world = []

    for a in world_angles:
        dist, hit = cast_ray(robot.position, float(a), robot.sensor_range, obstacle)
        ranges.append(dist)

        if hit is None:
            hit_mask.append(0)
            hit_points_world.append([np.nan, np.nan])
        else:
            hit_mask.append(1)
            hit_points_world.append(hit.astype(float).tolist())

    ranges = np.asarray(ranges, dtype=np.float32)
    hit_mask = np.asarray(hit_mask, dtype=np.int64)
    hit_points_world_arr = np.asarray(hit_points_world, dtype=np.float32)

    # local hit points: keep same length as rays.
    # Missed rays remain [nan, nan].
    hit_points_local = np.full_like(hit_points_world_arr, np.nan, dtype=np.float32)
    valid = hit_mask == 1
    if np.any(valid):
        hit_points_local[valid] = world_to_local(
            hit_points_world_arr[valid],
            robot.position,
            robot.theta,
        )

    return {
        "scan": {
            "angle_min": float(angle_min),
            "angle_max": float(angle_max),
            "angle_increment": float(angle_increment),
            "range_min": 0.0,
            "range_max": float(robot.sensor_range),
            "ranges": ranges,
            "intensities": np.zeros(robot.num_rays, dtype=np.float32),
            "frame_id": f"robot_{robot.id}/laser",
            "stamp": float(timestamp),
        },
        "debug": {
            "hit_mask": hit_mask,
            "hit_points_local": hit_points_local,
            "hit_points_world": hit_points_world_arr,
        },
    }
