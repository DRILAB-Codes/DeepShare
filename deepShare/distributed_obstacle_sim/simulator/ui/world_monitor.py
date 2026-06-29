import math

import numpy as np
import pygame

from simulator.ui import colors
from simulator.ui.utils import draw_arrow, draw_text, polygon_coords, world_to_screen

def get_robot_color(robot_id):
    rng = np.random.default_rng(robot_id)
    return tuple(int(c) for c in rng.integers(50, 255, size=3))

class WorldMonitor:
    """Left panel: global world view with obstacle, robots, lidar rays and comm edges."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.show_lidar = True
        self.show_graph = True

    def draw(self, surface, font, state, selected_robot_id: int):
        world = state["world"]
        obstacle = state["obstacle"]
        robots = state["robots"]
        sensor_outputs = state["sensor_outputs"]
        edge_index = state["edge_index"]

        pygame.draw.rect(surface, colors.PANEL_BG, self.rect)
        pygame.draw.rect(surface, colors.BORDER, self.rect, 1)
        self._draw_grid(surface, world)
        self._draw_obstacle(surface, world, obstacle)

        if self.show_graph:
            self._draw_graph(surface, world, robots, edge_index)
        if self.show_lidar:
            self._draw_lidar(surface, world, robots, sensor_outputs)
        self._draw_robots(surface, world, robots, selected_robot_id)

        draw_text(surface, font, "World Monitor", (self.rect.left + 10, self.rect.top + 8))
        draw_text(surface, font, "Space: pause | ←/→: robot | L: lidar | G: graph | R: reset | S: save", (self.rect.left + 10, self.rect.bottom - 24))

    def _draw_grid(self, surface, world):
        for i in range(int(world.width) + 1):
            p1 = world_to_screen((i, 0), self.rect, world.width, world.height)
            p2 = world_to_screen((i, world.height), self.rect, world.width, world.height)
            pygame.draw.line(surface, colors.GRID, p1, p2, 1)
        for j in range(int(world.height) + 1):
            p1 = world_to_screen((0, j), self.rect, world.width, world.height)
            p2 = world_to_screen((world.width, j), self.rect, world.width, world.height)
            pygame.draw.line(surface, colors.GRID, p1, p2, 1)

    def _draw_obstacle(self, surface, world, obstacle):
        pts = [world_to_screen(p, self.rect, world.width, world.height) for p in polygon_coords(obstacle)]
        if len(pts) >= 3:
            pygame.draw.polygon(surface, colors.OBSTACLE_FILL, pts)
            pygame.draw.lines(surface, colors.OBSTACLE_LINE, True, pts, 2)

    def _draw_graph(self, surface, world, robots, edge_index):
        if edge_index is None or edge_index.size == 0:
            return
        for src, dst in edge_index.T:
            if int(src) > int(dst):
                continue
            p1 = world_to_screen(robots[int(src)].position, self.rect, world.width, world.height)
            p2 = world_to_screen(robots[int(dst)].position, self.rect, world.width, world.height)
            pygame.draw.line(surface, colors.COMM_EDGE, p1, p2, 1)

    def _draw_lidar(self, surface, world, robots, sensor_outputs):
        for robot, out in zip(robots, sensor_outputs):
            origin = world_to_screen(robot.position, self.rect, world.width, world.height)

            scan = out["scan"]
            ranges = scan["ranges"]
            hit_mask = out["debug"]["hit_mask"]

            local_angles = (
                scan["angle_min"]
                + np.arange(len(ranges), dtype=np.float32) * scan["angle_increment"]
            )
            world_angles = robot.theta + local_angles

            for a, r, hit in zip(world_angles, ranges, hit_mask):
                end_world = robot.position + float(r) * np.array(
                    [math.cos(float(a)), math.sin(float(a))]
                )
                end = world_to_screen(end_world, self.rect, world.width, world.height)

                robot_color = get_robot_color(robot.robot_id)
                color = robot_color
                pygame.draw.line(surface, color, origin, end, 1)

                if int(hit):
                    pygame.draw.circle(surface, robot_color, end, 2)

    def _draw_robots(self, surface, world, robots, selected_robot_id: int):
        for robot in robots:
            pos = world_to_screen(robot.position, self.rect, world.width, world.height)
            is_selected = robot.id == selected_robot_id
            base_color = get_robot_color(robot.robot_id)
            color = colors.SELECTED_ROBOT if is_selected else base_color
            radius = 8 if is_selected else 6
            pygame.draw.circle(surface, color, pos, radius)
            draw_arrow(surface, pos, robot.theta, 22, colors.HEADING, 2)
