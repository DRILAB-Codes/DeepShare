# Distributed Obstacle Simulator

A minimal 2D simulator for distributed obstacle perception experiments.

This module simulates:

- 2D obstacles as polygons
- multiple robots around an obstacle
- simple 2D LiDAR ray-casting
- local sensor observations per robot
- robot communication graph
- matplotlib visualization

Recommended location:

```text
project_root/
├── simulator/
├── train/
├── models/
├── scripts/
└── ...
```

## Install

```bash
pip install numpy matplotlib shapely
```

## Run demo

```bash
python run_sim.py
```

## Output

The demo creates:

```text
outputs/sim_demo.png
outputs/sample.json
```

## Run live UI

```bash
pip install -r requirements.txt
python run_live_ui.py
```

Controls:

```text
Space      : pause / resume
Left/Right : select robot
L          : toggle world lidar rays
G          : toggle communication graph
R          : reset scene
S          : save current sample to outputs/live_ui_sample.json
Esc        : quit
```

Live UI layout:

```text
Left panel  : world monitor with obstacle, robots, lidar rays, and communication edges
Right panel : selected robot's local LiDAR rays and range vector
```
