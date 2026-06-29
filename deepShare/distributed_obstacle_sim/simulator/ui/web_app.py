"""Streamlit browser UI for the distributed obstacle simulator.

This module intentionally reads from the existing Simulation wrapper instead of
owning simulation logic. It is safe to run in a headless Docker/server
environment because rendering is done with matplotlib figures in the browser.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from simulator.core.simulation import Simulation, SimulationConfig


DEFAULT_SAVE_DIR = Path("outputs/web_ui_samples")


def _make_config(
    obstacle_shape: str,
    n_robots: int,
    num_rays: int,
    sensor_range: float,
    robot_radius: float,
    comm_range: float,
    move_robots: bool,
    obstacle_scale: float,
    fov_deg: float,
    orbit_speed: float,
    rotate_speed: float,
) -> SimulationConfig:
    return SimulationConfig(
        world_size=(10.0, 10.0),
        obstacle_shape=obstacle_shape,
        obstacle_center=(5.0, 5.0),
        obstacle_scale=obstacle_scale,
        n_robots=n_robots,
        robot_radius=robot_radius,
        sensor_range=sensor_range,
        fov=math.radians(fov_deg),
        num_rays=num_rays,
        comm_range=comm_range,
        move_robots=move_robots,
        orbit_speed=orbit_speed,
        rotate_speed=rotate_speed,
    )


def _config_key(cfg: SimulationConfig) -> Tuple[Any, ...]:
    return (
        cfg.world_size,
        cfg.obstacle_shape,
        cfg.obstacle_center,
        cfg.obstacle_scale,
        cfg.n_robots,
        cfg.robot_radius,
        cfg.sensor_range,
        cfg.fov,
        cfg.num_rays,
        cfg.comm_range,
        cfg.rotate_speed,
        cfg.orbit_speed,
        cfg.move_robots,
    )


def _polygon_coords(poly) -> np.ndarray:
    return np.asarray(poly.exterior.coords, dtype=np.float32)


def _safe_nan_to_none(x: float):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _array_to_jsonable(arr: np.ndarray):
    out = arr.astype(float).tolist()
    return json.loads(json.dumps(out, default=_safe_nan_to_none, allow_nan=False))


def _sensor_to_jsonable(sensor_output: Dict[str, np.ndarray]) -> Dict[str, Any]:
    # NaN hit points are converted to None so the JSON is standards-compliant.
    hit_world = sensor_output["hit_points_world"].astype(float).tolist()
    hit_world = [
        [None if (isinstance(v, float) and math.isnan(v)) else v for v in pt]
        for pt in hit_world
    ]
    return {
        "ranges": sensor_output["ranges"].astype(float).tolist(),
        "angles": sensor_output["angles"].astype(float).tolist(),
        "hit_mask": sensor_output["hit_mask"].astype(int).tolist(),
        "hit_points_world": hit_world,
        "hit_points_local": sensor_output["hit_points_local"].astype(float).tolist(),
    }


def state_to_sample(state: Dict[str, Any], cfg: SimulationConfig) -> Dict[str, Any]:
    return {
        "time": float(state["time"]),
        "world": {"width": float(cfg.world_size[0]), "height": float(cfg.world_size[1])},
        "obstacle": {
            "shape_type": cfg.obstacle_shape,
            "center": list(map(float, cfg.obstacle_center)),
            "scale": float(cfg.obstacle_scale),
            "polygon": _polygon_coords(state["obstacle"]).astype(float).tolist(),
        },
        "robots": [robot.as_dict() for robot in state["robots"]],
        "sensors": [_sensor_to_jsonable(out) for out in state["sensor_outputs"]],
        "graph": {
            "edge_index": state["edge_index"].astype(int).tolist(),
            "edge_attr": state["edge_attr"].astype(float).tolist(),
        },
    }


def save_current_sample(state: Dict[str, Any], cfg: SimulationConfig) -> Path:
    DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = DEFAULT_SAVE_DIR / f"sample_{ts}.json"
    sample = state_to_sample(state, cfg)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    return path


def draw_world_figure(
    state: Dict[str, Any],
    cfg: SimulationConfig,
    selected_robot_id: int,
    show_lidar: bool = True,
    show_graph: bool = True,
    show_hit_points: bool = True,
):
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.set_title("World monitor")
    ax.set_xlim(0.0, cfg.world_size[0])
    ax.set_ylim(0.0, cfg.world_size[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    # Obstacle polygon
    obs_xy = _polygon_coords(state["obstacle"])
    ax.fill(obs_xy[:, 0], obs_xy[:, 1], alpha=0.35, label="obstacle")
    ax.plot(obs_xy[:, 0], obs_xy[:, 1], linewidth=1.5)

    robots = state["robots"]
    sensor_outputs = state["sensor_outputs"]

    # Communication graph
    if show_graph and state["edge_index"].size > 0:
        for src, dst in state["edge_index"].T:
            p0 = robots[int(src)].position
            p1 = robots[int(dst)].position
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, alpha=0.28)

    # LiDAR rays and hit points
    if show_lidar:
        for robot, out in zip(robots, sensor_outputs):
            pos = robot.position
            for ray_idx, angle in enumerate(out["angles"]):
                r = float(out["ranges"][ray_idx])
                end = pos + r * np.array([math.cos(float(angle)), math.sin(float(angle))], dtype=np.float32)
                ray_alpha = 0.16 if robot.id != selected_robot_id else 0.35
                ax.plot([pos[0], end[0]], [pos[1], end[1]], linewidth=0.45, alpha=ray_alpha)

            if show_hit_points:
                mask = out["hit_mask"] == 1
                pts = out["hit_points_world"][mask]
                if len(pts) > 0:
                    size = 8 if robot.id != selected_robot_id else 18
                    ax.scatter(pts[:, 0], pts[:, 1], s=size, alpha=0.65)

    # Robots and heading arrows
    for robot in robots:
        x, y = robot.position
        selected = robot.id == selected_robot_id
        marker_size = 95 if selected else 55
        ax.scatter([x], [y], s=marker_size, marker="o", edgecolors="black", linewidths=1.2, zorder=4)
        ax.text(x + 0.07, y + 0.07, str(robot.id), fontsize=9, zorder=5)
        hx = 0.35 * math.cos(robot.theta)
        hy = 0.35 * math.sin(robot.theta)
        ax.arrow(x, y, hx, hy, head_width=0.09, length_includes_head=True, zorder=5)

    ax.text(
        0.02,
        0.98,
        f"t={state['time']:.2f}s | robots={cfg.n_robots} | rays={cfg.num_rays}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "alpha": 0.12},
    )
    return fig


def draw_robot_lidar_figure(state: Dict[str, Any], selected_robot_id: int):
    robots = state["robots"]
    out = state["sensor_outputs"][selected_robot_id]
    robot = robots[selected_robot_id]

    ranges = out["ranges"]
    rel_angles = out["angles"] - robot.theta
    hit_mask = out["hit_mask"]

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.set_title(f"Robot {selected_robot_id} local LiDAR")
    ax.set_aspect("equal", adjustable="box")
    lim = max(float(robot.sensor_range), 1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.25)
    ax.axhline(0.0, linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, linewidth=0.8, alpha=0.5)

    # Robot local frame: +x is forward.
    ax.scatter([0.0], [0.0], s=90, marker="o", edgecolors="black", linewidths=1.2, zorder=4)
    ax.arrow(0.0, 0.0, 0.45, 0.0, head_width=0.12, length_includes_head=True, zorder=5)
    ax.text(0.5, 0.05, "+x forward", fontsize=9)

    for a, r, hit in zip(rel_angles, ranges, hit_mask):
        x = float(r) * math.cos(float(a))
        y = float(r) * math.sin(float(a))
        alpha = 0.36 if hit else 0.12
        ax.plot([0.0, x], [0.0, y], linewidth=0.65, alpha=alpha)

    local_pts = out["hit_points_local"]
    if len(local_pts) > 0:
        ax.scatter(local_pts[:, 0], local_pts[:, 1], s=20, alpha=0.8, label="hit points")

    hit_ratio = float(hit_mask.mean()) if len(hit_mask) else 0.0
    ax.text(
        0.02,
        0.98,
        f"hit ratio={hit_ratio:.3f}\nmin range={float(np.min(ranges)):.3f}\nmean range={float(np.mean(ranges)):.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "alpha": 0.12},
    )
    return fig


def draw_range_figure(state: Dict[str, Any], selected_robot_id: int):
    out = state["sensor_outputs"][selected_robot_id]
    ranges = out["ranges"]
    hit_mask = out["hit_mask"]

    fig, ax = plt.subplots(figsize=(5.4, 2.2))
    ax.set_title(f"Robot {selected_robot_id} range vector")
    ax.plot(np.arange(len(ranges)), ranges, linewidth=1.2)
    if len(ranges) > 0:
        hit_idx = np.where(hit_mask == 1)[0]
        if len(hit_idx) > 0:
            ax.scatter(hit_idx, ranges[hit_idx], s=14, alpha=0.8)
    ax.set_xlabel("ray index")
    ax.set_ylabel("range")
    ax.grid(True, alpha=0.25)
    return fig


def _initialize_or_update_sim(cfg: SimulationConfig):
    key = _config_key(cfg)
    if "sim" not in st.session_state or st.session_state.get("config_key") != key:
        st.session_state.sim = Simulation(cfg)
        st.session_state.config_key = key
        st.session_state.selected_robot_id = 0


def main():
    st.set_page_config(page_title="Distributed Obstacle Simulator", layout="wide")
    st.title("Distributed Obstacle Simulator - Web UI")

    with st.sidebar:
        st.header("Simulation")
        obstacle_shape = st.selectbox(
            "Obstacle shape",
            ["star", "circle", "triangle", "rectangle", "pentagon", "cross", "u"],
            index=0,
        )
        n_robots = st.slider("Number of robots", 1, 32, 8, 1)
        selected_robot_id = st.slider("Selected robot", 0, max(n_robots - 1, 0), 0, 1)
        num_rays = st.slider("LiDAR rays", 8, 256, 64, 8)
        fov_deg = st.slider("FOV degrees", 30, 360, 180, 5)
        sensor_range = st.slider("Sensor range", 0.5, 8.0, 5.0, 0.1)
        robot_radius = st.slider("Robot orbit radius", 1.5, 5.0, 3.2, 0.1)
        comm_range = st.slider("Communication range", 0.5, 6.0, 2.6, 0.1)
        obstacle_scale = st.slider("Obstacle scale", 0.3, 2.0, 1.0, 0.05)
        move_robots = st.checkbox("Move robots", value=True)
        orbit_speed = st.slider("Orbit speed", 0.0, 1.5, 0.20, 0.01)
        rotate_speed = st.slider("Heading wobble speed", 0.0, 2.0, 0.25, 0.01)

        st.header("View")
        show_lidar = st.checkbox("Show world LiDAR rays", value=True)
        show_graph = st.checkbox("Show communication graph", value=True)
        show_hit_points = st.checkbox("Show hit points", value=True)
        auto_run = st.checkbox("Auto run", value=False)
        dt = st.slider("Step dt", 0.01, 0.25, 0.05, 0.01)
        refresh_sec = st.slider("Auto refresh seconds", 0.05, 1.0, 0.15, 0.05)

    cfg = _make_config(
        obstacle_shape=obstacle_shape,
        n_robots=n_robots,
        num_rays=num_rays,
        sensor_range=sensor_range,
        robot_radius=robot_radius,
        comm_range=comm_range,
        move_robots=move_robots,
        obstacle_scale=obstacle_scale,
        fov_deg=fov_deg,
        orbit_speed=orbit_speed,
        rotate_speed=rotate_speed,
    )
    _initialize_or_update_sim(cfg)
    st.session_state.selected_robot_id = min(selected_robot_id, cfg.n_robots - 1)

    sim: Simulation = st.session_state.sim

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if st.button("Step"):
            sim.step(dt=dt)
    with c2:
        if st.button("Reset"):
            sim.reset()
    with c3:
        if st.button("Save sample"):
            path = save_current_sample(sim.get_state(), cfg)
            st.success(f"Saved: {path}")
    with c4:
        st.caption("Docker/server usage: open the Streamlit port in your container, then view this page from a browser.")

    if auto_run:
        sim.step(dt=dt)

    state = sim.get_state()
    selected = st.session_state.selected_robot_id

    left, right = st.columns([1.45, 1.0])
    with left:
        fig_world = draw_world_figure(
            state,
            cfg,
            selected_robot_id=selected,
            show_lidar=show_lidar,
            show_graph=show_graph,
            show_hit_points=show_hit_points,
        )
        st.pyplot(fig_world, clear_figure=True)
    with right:
        fig_lidar = draw_robot_lidar_figure(state, selected)
        st.pyplot(fig_lidar, clear_figure=True)
        fig_range = draw_range_figure(state, selected)
        st.pyplot(fig_range, clear_figure=True)

    out = state["sensor_outputs"][selected]
    hit_ratio = float(out["hit_mask"].mean()) if len(out["hit_mask"]) else 0.0
    st.write(
        {
            "time": round(float(state["time"]), 4),
            "selected_robot": int(selected),
            "hit_ratio": round(hit_ratio, 4),
            "num_edges": int(state["edge_index"].shape[1]),
            "ranges_shape": list(out["ranges"].shape),
            "local_hit_points_shape": list(out["hit_points_local"].shape),
        }
    )

    with st.expander("Selected robot raw sensor preview"):
        st.json(
            {
                "ranges_first_10": out["ranges"][:10].astype(float).round(4).tolist(),
                "hit_mask_first_30": out["hit_mask"][:30].astype(int).tolist(),
                "hit_points_local_first_10": out["hit_points_local"][:10].astype(float).round(4).tolist(),
            }
        )

    if auto_run:
        time.sleep(refresh_sec)
        st.rerun()


if __name__ == "__main__":
    main()
