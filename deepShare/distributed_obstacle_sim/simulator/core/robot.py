from dataclasses import dataclass
import numpy as np


@dataclass
class Robot:
    id: int
    position: np.ndarray  # (x, y)
    theta: float = 0.0

    # sensor config
    sensor_range: float = 5.0
    fov: float = np.pi
    num_rays: int = 64

    # optional (real-world compatibility)
    frame_id: str = None

    def __post_init__(self):
        if self.frame_id is None:
            self.frame_id = f"robot_{self.id}/base_link"

    def as_dict(self):
        return {
            "robot_id": int(self.id),

            # pose (ROS스럽게 분리)
            "pose": {
                "x": float(self.position[0]),
                "y": float(self.position[1]),
                "theta": float(self.theta),
            },

            # sensor spec (고정 파라미터)
            "sensor": {
                "range_max": float(self.sensor_range),
                "fov": float(self.fov),
                "num_rays": int(self.num_rays),
            },

            "frame_id": self.frame_id,
        }
