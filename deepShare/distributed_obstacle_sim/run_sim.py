from simulator.dataset.sample_generator import make_single_sample, save_sample_json
from simulator.visualization.plot_scene import plot_scene


def main():
    world, obstacle, robots, sensor_outputs, sample = make_single_sample(
        shape_type="star",
        n_robots=12,
        robot_radius=3.2,
        sensor_range=3.0,
        num_rays=32,
        comm_range=2.6,
    )

    edge_index = sample["graph"]["edge_index"]

    # Convert list back to simple array for plotting.
    import numpy as np
    edge_index = np.asarray(edge_index, dtype=int)

    plot_scene(
        world=world,
        obstacle=obstacle,
        robots=robots,
        sensor_outputs=sensor_outputs,
        edge_index=edge_index,
        save_path="outputs/sim_demo.png",
        show=False,
    )

    save_sample_json(sample, "outputs/sample.json")
    print("Saved outputs/sim_demo.png")
    print("Saved outputs/sample.json")


if __name__ == "__main__":
    main()
