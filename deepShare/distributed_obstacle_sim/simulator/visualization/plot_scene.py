from pathlib import Path
import colorsys

import matplotlib.pyplot as plt
import numpy as np

def get_robot_color(robot_id):
    h = (robot_id * 0.61803398875) % 1.0  # golden ratio
    s = 0.75
    v = 0.95
    return colorsys.hsv_to_rgb(h, s, v)

def _plot_polygon(ax, poly):
    x, y = poly.exterior.xy
    ax.fill(x, y, alpha=0.35)
    ax.plot(x, y)


def plot_scene(
    world,
    obstacle,
    robots,
    sensor_outputs,
    edge_index=None,
    save_path=None,
    show=True,
):
    fig, ax = plt.subplots(figsize=(7, 7))

    _plot_polygon(ax, obstacle)

    if edge_index is not None and edge_index.size > 0:
        for src, dst in edge_index.T:
            p1 = robots[int(src)].position
            p2 = robots[int(dst)].position
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.8, alpha=0.25)

    for robot, out in zip(robots, sensor_outputs):
        x, y = robot.position
        robot_color = np.array(get_robot_color(robot.id))
        ax.scatter([x], [y], s=40, color=robot_color)
        ax.text(x + 0.05, y + 0.05, f"R{robot.id}", color=robot_color)
        
        # heading
        heading_end = robot.position + 0.35 * np.array(
            [np.cos(robot.theta), np.sin(robot.theta)]
        )
        ax.plot([x, heading_end[0]], [y, heading_end[1]], linewidth=2, color=robot_color)

        scan = out["scan"]
        debug = out["debug"]

        ranges = scan["ranges"]
        hit_mask = debug["hit_mask"]

        local_angles = (
            scan["angle_min"]
            + np.arange(len(ranges), dtype=np.float32) * scan["angle_increment"]
        )
        world_angles = robot.theta + local_angles

        for a, r, hit in zip(world_angles, ranges, hit_mask):
            end = robot.position + float(r) * np.array(
                [np.cos(float(a)), np.sin(float(a))]
            )
            ax.plot([x, end[0]], [y, end[1]], linewidth=0.3, alpha=0.3, color=robot_color)
            if int(hit):
                ax.scatter([end[0]], [end[1]], s=6, color=robot_color)

    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("2D Distributed Obstacle Sensing Demo")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
