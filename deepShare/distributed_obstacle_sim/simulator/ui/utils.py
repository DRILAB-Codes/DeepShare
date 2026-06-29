import math
from typing import Tuple

import pygame


def world_to_screen(point, rect: pygame.Rect, world_width: float, world_height: float):
    x, y = float(point[0]), float(point[1])
    sx = rect.left + x / world_width * rect.width
    sy = rect.bottom - y / world_height * rect.height
    return int(round(sx)), int(round(sy))


def local_to_screen(point, rect: pygame.Rect, scale: float):
    x, y = float(point[0]), float(point[1])
    cx, cy = rect.center
    return int(round(cx + x * scale)), int(round(cy - y * scale))


def draw_text(surface, font, text: str, pos: Tuple[int, int], color=(25, 25, 25)):
    img = font.render(text, True, color)
    surface.blit(img, pos)


def polygon_coords(poly):
    return [(float(x), float(y)) for x, y in poly.exterior.coords]


def draw_arrow(surface, start, angle: float, length: float, color, width: int = 2):
    x, y = start
    end = (int(x + length * math.cos(angle)), int(y - length * math.sin(angle)))
    pygame.draw.line(surface, color, start, end, width)
    for da in (2.6, -2.6):
        hx = int(end[0] + 8 * math.cos(-angle + da))
        hy = int(end[1] + 8 * math.sin(-angle + da))
        pygame.draw.line(surface, color, end, (hx, hy), width)
