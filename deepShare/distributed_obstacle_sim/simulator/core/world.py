import math
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon

from simulator.core.robot import Robot


class World:
    def __init__(self, width: float = 10.0, height: float = 10.0):
        self.width = width
        self.height = height
        self.obstacles: List[Polygon] = []
        self.robots: List[Robot] = []

    def add_obstacle(self, obstacle: Polygon):
        self.obstacles.append(obstacle)

    def add_robot(self, robot: Robot):
        self.robots.append(robot)

    def clear_robots(self):
        self.robots = []


def place_robots_on_circle(
    n_robots: int,
    center: Tuple[float, float] = (5.0, 5.0),
    radius: float = 3.0,
    sensor_range: float = 5.0,
    fov: float = math.pi,
    num_rays: int = 64,
    look_at_center: bool = True,
) -> List[Robot]:
    """Place robots on a circle around the obstacle."""
    robots = []
    cx, cy = center
    for i in range(n_robots):
        a = 2 * math.pi * i / n_robots
        pos = np.array([cx + radius * math.cos(a), cy + radius * math.sin(a)], dtype=np.float32)
        theta = math.atan2(cy - pos[1], cx - pos[0]) if look_at_center else a
        robots.append(
            Robot(
                id=i,
                position=pos,
                theta=theta,
                sensor_range=sensor_range,
                fov=fov,
                num_rays=num_rays,
            )
        )
    return robots
