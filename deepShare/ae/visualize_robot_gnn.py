import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.lidar_pointcloud_dataset import LidarObstacleAEDataset
from models.autoencoder import PointNet2AutoEncoder
from models.robot_gnn import build_robot_gnn_model


def load_ae_decoder(cfg, device):
    ae_ckpt_path = cfg["ae"]["checkpoint"]
    ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_cfg = ckpt["cfg"]

    ae = PointNet2AutoEncoder(
        encoder_mode=ae_cfg["model"]["encoder_mode"],
        latent_dim=ae_cfg["model"]["latent_dim"],
        target_num_points=ae_cfg["data"]["target_num_points"],
        base_radius=ae_cfg["model"].get("base_radius", 1.0),
        npoint1=ae_cfg["model"].get("npoint1", 32),
        npoint2=ae_cfg["model"].get("npoint2", 16),
    ).to(device)

    ae.load_state_dict(ckpt["model"])
    ae.eval()

    decoder = ae.decoder
    decoder.eval()

    for p in decoder.parameters():
        p.requires_grad = False

    return decoder


def build_dataset(cfg, data_dir, seed=0):
    data_cfg = cfg["data"]

    return LidarObstacleAEDataset(
        data_dir=data_dir,
        input_num_points=data_cfg["input_num_points"],
        target_num_points=data_cfg["target_num_points"],
        include_miss=data_cfg.get("include_miss", False),
        normalize=data_cfg.get("normalize", True),
        seed=seed,
        use_world_frame=data_cfg.get("use_world_frame", True),
    )


def load_robot_gnn(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]

    decoder = load_ae_decoder(cfg, device)
    model = build_robot_gnn_model(cfg, decoder=decoder).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, cfg


def get_shape_type(sample):
    meta = sample.get("meta", {})
    return meta.get("shape_type", "unknown")


def get_robot_positions(sample):
    """
    dataset에 저장된 실제 로봇 위치만 사용.
    없으면 None 반환.
    """
    robot_pos = None

    if "robot_xy" in sample:
        robot_pos = sample["robot_xy"]
    elif "robot_pos" in sample:
        robot_pos = sample["robot_pos"]
    elif "robots" in sample:
        robot_pos = sample["robots"]

    if robot_pos is None:
        return None

    if torch.is_tensor(robot_pos):
        return robot_pos.detach().cpu().numpy()

    return torch.as_tensor(robot_pos, dtype=torch.float32).detach().cpu().numpy()


def collect_indices_by_shape(ds, per_shape=4, start_index=0):
    shape_to_indices = defaultdict(list)

    for idx in range(start_index, len(ds)):
        sample = ds[idx]
        shape_type = get_shape_type(sample)

        if len(shape_to_indices[shape_type]) < per_shape:
            shape_to_indices[shape_type].append(idx)

    selected_indices = []
    for shape_type in sorted(shape_to_indices.keys()):
        selected_indices.extend(shape_to_indices[shape_type])

    return selected_indices, shape_to_indices


def plot_one_sample(
    sample,
    pred_nodes,
    info,
    save_path,
    sample_idx,
    max_robots=None,
):
    x = sample["x"].cpu().numpy()                  # [N, P, 3]
    target = sample["target"].cpu().numpy()        # [Q, 3]
    pred_nodes = pred_nodes.detach().cpu().numpy() # [N, Q, 3]

    num_robots = x.shape[0]

    if max_robots is None:
        robot_indices = list(range(num_robots))
    else:
        robot_indices = list(range(min(max_robots, num_robots)))

    n_show = len(robot_indices)

    meta = sample.get("meta", {})
    shape_type = meta.get("shape_type", "unknown")
    obstacle_scale = meta.get("obstacle_scale", "unknown")

    robot_pos = get_robot_positions(sample)

    fig = plt.figure(figsize=(12, max(6, 2.3 * n_show)))
    gs = fig.add_gridspec(
        nrows=n_show,
        ncols=2,
        width_ratios=[1.45, 1.0],
        wspace=0.25,
        hspace=0.35,
    )

    # ========================================================
    # Left: 전체 환경
    # ========================================================
    env_ax = fig.add_subplot(gs[:, 0])

    env_ax.scatter(
        target[:, 0],
        target[:, 1],
        s=10,
        label="target obstacle",
        alpha=0.75,
    )

    for robot_idx in range(num_robots):
        partial = x[robot_idx]
        env_ax.scatter(
            partial[:, 0],
            partial[:, 1],
            s=16,
            marker="x",
            alpha=0.55,
        )

    if robot_pos is not None:
        env_ax.scatter(
            robot_pos[:, 0],
            robot_pos[:, 1],
            s=55,
            marker="o",
            facecolors="none",
            edgecolors="black",
            label="robots",
        )

        for robot_idx in range(min(num_robots, len(robot_pos))):
            env_ax.text(
                robot_pos[robot_idx, 0],
                robot_pos[robot_idx, 1],
                str(robot_idx),
                fontsize=8,
                ha="center",
                va="center",
            )
    else:
        print(
            f"[WARN] Sample {sample_idx}: robot position not found. "
            "Check LidarObstacleAEDataset returns 'robot_xy', 'robot_pos', or 'robots'."
        )

    env_ax.set_title("Environment: obstacle + robots + observed point clouds")
    env_ax.set_aspect("equal", adjustable="box")
    env_ax.grid(True, alpha=0.25)
    env_ax.legend(loc="best")

    # ========================================================
    # Right: 각 로봇별 reconstruction
    # ========================================================
    for row, robot_idx in enumerate(robot_indices):
        ax = fig.add_subplot(gs[row, 1])

        partial = x[robot_idx]
        pred = pred_nodes[robot_idx]

        ax.scatter(
            target[:, 0],
            target[:, 1],
            s=8,
            label="target",
            alpha=0.45,
        )
        ax.scatter(
            partial[:, 0],
            partial[:, 1],
            s=14,
            marker="x",
            label="partial",
            alpha=0.75,
        )
        ax.scatter(
            pred[:, 0],
            pred[:, 1],
            s=8,
            label="recon",
            alpha=0.85,
        )

        ax.set_title(f"Robot {robot_idx} reconstruction")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        if row == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Sample {sample_idx} | shape={shape_type} | scale={obstacle_scale} | "
        f"steps={info.get('used_steps')} | converged={info.get('converged')}",
        fontsize=13,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--start_index", type=int, default=0)

    # 기존 num_samples 대신 shape별 개수
    p.add_argument("--per_shape", type=int, default=4)

    p.add_argument("--save_dir", default="out/robot_gnn_vis_by_shape")
    p.add_argument("--max_robots", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_robot_gnn(args.checkpoint, device)
    ds = build_dataset(cfg, args.data_dir, seed=args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    selected_indices, shape_to_indices = collect_indices_by_shape(
        ds,
        per_shape=args.per_shape,
        start_index=args.start_index,
    )

    print("Selected samples by shape:")
    for shape_type in sorted(shape_to_indices.keys()):
        print(f"  {shape_type}: {shape_to_indices[shape_type]}")

    for i in selected_indices:
        sample = ds[i]

        x = sample["x"].to(device)
        edge_index = sample["edge_index"].to(device)

        with torch.no_grad():
            pred_nodes, final_h, info = model(x, edge_index)

        shape_type = get_shape_type(sample)
        shape_dir = save_dir / str(shape_type)
        shape_dir.mkdir(parents=True, exist_ok=True)

        save_path = shape_dir / f"robot_gnn_recon_{shape_type}_{i:04d}.png"

        plot_one_sample(
            sample=sample,
            pred_nodes=pred_nodes,
            info=info,
            save_path=save_path,
            sample_idx=i,
            max_robots=args.max_robots,
        )

        print(
            f"Saved: {save_path} | "
            f"steps={info.get('used_steps')} | "
            f"converged={info.get('converged')} | "
            f"gap_history={info.get('gap_history')}"
        )


if __name__ == "__main__":
    main()