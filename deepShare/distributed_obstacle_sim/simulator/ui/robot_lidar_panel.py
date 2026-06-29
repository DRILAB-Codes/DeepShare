import math

import numpy as np
import pygame

from simulator.ui import colors
from simulator.ui.utils import draw_text, local_to_screen


class RobotLidarPanel:
    """Right panel: selected robot's local lidar data."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect

    def draw(self, surface, font, state, selected_robot_id: int):
        robots = state["robots"]
        sensor_outputs = state["sensor_outputs"]
        robot = robots[selected_robot_id]
        out = sensor_outputs[selected_robot_id]

        pygame.draw.rect(surface, colors.PANEL_BG, self.rect)
        pygame.draw.rect(surface, colors.BORDER, self.rect, 1)
        draw_text(surface, font, f"Robot R{selected_robot_id} Local LiDAR", (self.rect.left + 10, self.rect.top + 8))

        view_rect = pygame.Rect(self.rect.left + 16, self.rect.top + 42, self.rect.width - 32, int(self.rect.height * 0.58))
        range_rect = pygame.Rect(self.rect.left + 16, view_rect.bottom + 24, self.rect.width - 32, self.rect.bottom - view_rect.bottom - 40)

        self._draw_local_lidar(surface, view_rect, robot, out)
        self._draw_range_plot(surface, font, range_rect, out)

        hit_ratio = float(np.mean(out["hit_mask"])) if len(out["hit_mask"]) else 0.0
        min_range = float(np.min(out["ranges"])) if len(out["ranges"]) else 0.0
        draw_text(surface, font, f"hit ratio={hit_ratio:.2f} | min range={min_range:.2f} | rays={robot.num_rays}", (self.rect.left + 10, self.rect.bottom - 24))

    def _draw_local_lidar(self, surface, rect, robot, out):
        pygame.draw.rect(surface, (250, 250, 250), rect)
        pygame.draw.rect(surface, colors.BORDER, rect, 1)

        scale = min(rect.width, rect.height) / (2.25 * robot.sensor_range)
        origin = rect.center
        pygame.draw.line(surface, (210, 210, 210), (rect.left, origin[1]), (rect.right, origin[1]), 1)
        pygame.draw.line(surface, (210, 210, 210), (origin[0], rect.top), (origin[0], rect.bottom), 1)
        pygame.draw.circle(surface, colors.SELECTED_ROBOT, origin, 7)

        scan = out["scan"]
        local_angles = (scan["angle_min"]+ np.arange(len(ranges), dtype=np.float32) * scan["angle_increment"])
        for a_local, r, hit in zip(local_angles, out["ranges"], out["hit_mask"]):
            end_local = np.array([float(r) * math.cos(float(a_local)), float(r) * math.sin(float(a_local))])
            end = local_to_screen(end_local, rect, scale)
            color = colors.LIDAR_HIT if int(hit) else colors.LIDAR_MISS
            pygame.draw.line(surface, color, origin, end, 1)
            if int(hit):
                pygame.draw.circle(surface, colors.HIT_POINT, end, 3)
        pygame.draw.line(surface, colors.HEADING, origin, (origin[0] + 35, origin[1]), 3)

    def _draw_range_plot(self, surface, font, rect, out):
        pygame.draw.rect(surface, (250, 250, 250), rect)
        pygame.draw.rect(surface, colors.BORDER, rect, 1)
        draw_text(surface, font, "Range vector", (rect.left + 6, rect.top + 6))

        ranges = out["ranges"]
        hit_mask = out["hit_mask"]
        if len(ranges) == 0:
            return
        max_r = max(float(np.max(ranges)), 1e-6)
        bar_w = max(1, rect.width / len(ranges))
        base_y = rect.bottom - 8
        usable_h = rect.height - 34
        for i, (r, hit) in enumerate(zip(ranges, hit_mask)):
            h = int((float(r) / max_r) * usable_h)
            x = int(rect.left + i * bar_w)
            color = colors.LIDAR_HIT if int(hit) else colors.LIDAR_MISS
            pygame.draw.line(surface, color, (x, base_y), (x, base_y - h), 1)
