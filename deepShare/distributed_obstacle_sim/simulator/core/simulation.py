import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from simulator.core.sensor import simulate_lidar
from simulator.core.world import World, place_robots_on_circle
from simulator.geometry.obstacles import create_obstacle
from simulator.graph.build_graph import build_comm_graph


@dataclass
class SimulationConfig:
    world_size: Tuple[float, float] = (10.0, 10.0)
    obstacle_shape: str = "star"
    obstacle_center: Tuple[float, float] = (5.0, 5.0)
    obstacle_scale: float = 1.0
    n_robots: int = 8
    robot_radius: float = 3.2
    sensor_range: float = 5.0
    fov: float = math.pi
    num_rays: int = 64
    comm_range: float = 2.6
    rotate_speed: float = 0.25
    orbit_speed: float = 0.20
    move_robots: bool = True


class Simulation:
    """State wrapper for live monitoring and future batch generation."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.time = 0.0
        self.world = None
        self.obstacle = None
        self.robots = []
        self.sensor_outputs = []
        self.edge_index = np.zeros((2, 0), dtype=np.int64)
        self.edge_attr = np.zeros((0, 1), dtype=np.float32)
        self.reset()

    def reset(self):
        cfg = self.config
        self.time = 0.0
        self.world = World(width=cfg.world_size[0], height=cfg.world_size[1])
        self.obstacle = create_obstacle(cfg.obstacle_shape, center=cfg.obstacle_center, scale=cfg.obstacle_scale)
        self.world.add_obstacle(self.obstacle)
        self.robots = place_robots_on_circle(
            n_robots=cfg.n_robots,
            center=cfg.obstacle_center,
            radius=cfg.robot_radius,
            sensor_range=cfg.sensor_range,
            fov=cfg.fov,
            num_rays=cfg.num_rays,
            look_at_center=True,
        )
        for robot in self.robots:
            self.world.add_robot(robot)
        self._update_outputs()

    def step(self, dt: float = 1.0 / 30.0):
        cfg = self.config
        self.time += dt
        if cfg.move_robots:
            cx, cy = cfg.obstacle_center
            for i, robot in enumerate(self.robots):
                base = 2 * math.pi * i / cfg.n_robots
                a = base + cfg.orbit_speed * self.time
                robot.position = np.array(
                    [cx + cfg.robot_radius * math.cos(a), cy + cfg.robot_radius * math.sin(a)],
                    dtype=np.float32,
                )
                look_theta = math.atan2(cy - robot.position[1], cx - robot.position[0])
                robot.theta = look_theta + 0.25 * math.sin(cfg.rotate_speed * self.time + i)
        self._update_outputs()

    def _update_outputs(self):
        self.sensor_outputs = [simulate_lidar(robot, self.obstacle) for robot in self.robots]
        self.edge_index, self.edge_attr = build_comm_graph(self.robots, comm_range=self.config.comm_range)

    def get_state(self) -> Dict:
        return {
            "time": self.time,
            "world": self.world,
            "obstacle": self.obstacle,
            "robots": self.robots,
            "sensor_outputs": self.sensor_outputs,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
        }
