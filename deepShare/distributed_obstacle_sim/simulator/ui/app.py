import json
from pathlib import Path

import pygame

from simulator.core.simulation import Simulation, SimulationConfig
from simulator.dataset.sample_generator import polygon_to_coords, sensor_to_serializable
from simulator.ui import colors
from simulator.ui.robot_lidar_panel import RobotLidarPanel
from simulator.ui.world_monitor import WorldMonitor


class LiveSimApp:
    def __init__(self, sim: Simulation, width: int = 1200, height: int = 720, fps: int = 30):
        self.sim = sim
        self.width = width
        self.height = height
        self.fps = fps
        self.paused = False
        self.selected_robot_id = 0
        self.running = True

        pygame.init()
        pygame.display.set_caption("Distributed Obstacle Simulator - Live UI")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        left_rect = pygame.Rect(12, 12, int(width * 0.65) - 18, height - 24)
        right_rect = pygame.Rect(left_rect.right + 12, 12, width - left_rect.right - 24, height - 24)
        self.world_monitor = WorldMonitor(left_rect)
        self.lidar_panel = RobotLidarPanel(right_rect)

    def run(self):
        while self.running:
            dt = self.clock.tick(self.fps) / 1000.0
            self._handle_events()
            if not self.paused:
                self.sim.step(dt)
            self._draw()
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT:
                    self.selected_robot_id = (self.selected_robot_id + 1) % len(self.sim.robots)
                elif event.key == pygame.K_LEFT:
                    self.selected_robot_id = (self.selected_robot_id - 1) % len(self.sim.robots)
                elif event.key == pygame.K_l:
                    self.world_monitor.show_lidar = not self.world_monitor.show_lidar
                elif event.key == pygame.K_g:
                    self.world_monitor.show_graph = not self.world_monitor.show_graph
                elif event.key == pygame.K_r:
                    self.sim.reset()
                    self.selected_robot_id = 0
                elif event.key == pygame.K_s:
                    self._save_current_sample()

    def _draw(self):
        self.screen.fill(colors.BACKGROUND)
        state = self.sim.get_state()
        self.world_monitor.draw(self.screen, self.font, state, self.selected_robot_id)
        self.lidar_panel.draw(self.screen, self.font, state, self.selected_robot_id)
        status = "PAUSED" if self.paused else "RUNNING"
        img = self.font.render(f"{status} | t={state['time']:.2f}s | selected=R{self.selected_robot_id}", True, colors.TEXT)
        self.screen.blit(img, (16, self.height - 22))
        pygame.display.flip()

    def _save_current_sample(self):
        state = self.sim.get_state()
        path = Path("outputs/live_ui_sample.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        sample = {
            "time": float(state["time"]),
            "world": {"width": state["world"].width, "height": state["world"].height},
            "obstacle": {"shape_type": self.sim.config.obstacle_shape, "polygon": polygon_to_coords(state["obstacle"])},
            "robots": [robot.as_dict() for robot in state["robots"]],
            "sensors": [sensor_to_serializable(out) for out in state["sensor_outputs"]],
            "graph": {
                "edge_index": state["edge_index"].astype(int).tolist(),
                "edge_attr": state["edge_attr"].astype(float).tolist(),
            },
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2)
        print(f"Saved {path}")


def main():
    config = SimulationConfig(
        obstacle_shape="star",
        n_robots=8,
        robot_radius=3.2,
        sensor_range=5.0,
        num_rays=64,
        comm_range=2.6,
        move_robots=True,
    )
    sim = Simulation(config)
    app = LiveSimApp(sim)
    app.run()


if __name__ == "__main__":
    main()
