import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.lidar_pointcloud_dataset import LidarObstacleAEDataset
from datasets.obstacle_dataset import ObstacleBoundaryAEDataset
from models.autoencoder import PointNet2AutoEncoder


def build_dataset(args, cfg):
    data_cfg = cfg["data"]

    if args.dataset_mode == "ae":
        return ObstacleBoundaryAEDataset(
            args.data_dir,
            input_num_points=data_cfg["input_num_points"],
            target_num_points=data_cfg["target_num_points"],
            normalize=data_cfg.get("normalize", True),
            seed=args.seed,
        )

    if args.dataset_mode == "partial":
        return LidarObstacleAEDataset(
            args.data_dir,
            input_num_points=data_cfg["input_num_points"],
            target_num_points=data_cfg["target_num_points"],
            include_miss=data_cfg.get("include_miss", False),
            normalize=data_cfg.get("normalize", True),
            seed=args.seed,
            use_world_frame=data_cfg.get("use_world_frame", True),
        )

    raise ValueError(f"Unknown dataset_mode: {args.dataset_mode}")

def get_model_input(sample, dataset_mode, robot_index=0):
    if dataset_mode == "ae":
        return sample["input"], "input"

    if dataset_mode == "partial":
        if "x" in sample:
            x = sample["x"]
            robot_index = min(robot_index, x.shape[0] - 1)
            return x[robot_index], f"robot {robot_index} partial"

        if "partial" in sample:
            return sample["partial"], "partial input"

        raise KeyError("partial mode requires sample['x'] or sample['partial'].")

    raise ValueError(f"Unknown dataset_mode: {dataset_mode}")


def get_shape_type(sample):
    meta = sample.get("meta", {})
    return meta.get("shape_type", "unknown")


def get_robot_positions(sample):
    """
    실제 로봇 위치를 dataset에서 읽어온다.
    우선순위:
      1. sample["robot_xy"]
      2. sample["robot_pos"]
      3. sample["robots"]

    없으면 None 반환.
    포인트클라우드 평균 fallback은 실제 로봇 위치가 아니므로 사용하지 않음.
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


def visualize_sample(model, ds, idx, dataset_mode, save_dir, robot_index=0):
    sample = ds[idx]

    model_input, input_label = get_model_input(
        sample,
        dataset_mode,
        robot_index=robot_index,
    )
    model_input = model_input.unsqueeze(0)

    with torch.no_grad():
        pred, _ = model(model_input)

    input_np = model_input.squeeze(0).detach().cpu().numpy()
    target_np = sample["target"].detach().cpu().numpy()
    pred_np = pred.squeeze(0).detach().cpu().numpy()

    robot_pos = get_robot_positions(sample)

    meta = sample.get("meta", {})
    shape_type = meta.get("shape_type", "unknown")
    obstacle_scale = meta.get("obstacle_scale", "unknown")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.scatter(target_np[:, 0], target_np[:, 1], s=12, label="target obstacle")
    ax.scatter(input_np[:, 0], input_np[:, 1], s=20, marker="x", label=input_label)
    ax.scatter(pred_np[:, 0], pred_np[:, 1], s=12, label="recon")

    if robot_pos is not None:
        ax.scatter(
            robot_pos[:, 0],
            robot_pos[:, 1],
            s=70,
            marker="o",
            facecolors="none",
            edgecolors="black",
            label="robots",
        )

        for ridx, (rx, ry) in enumerate(robot_pos):
            ax.text(
                rx,
                ry,
                str(ridx),
                fontsize=8,
                ha="center",
                va="center",
            )
    else:
        print(
            f"[WARN] Sample {idx}: robot position not found. "
            "Check LidarObstacleAEDataset returns 'robot_xy'."
        )

    ax.set_title(
        f"Sample {idx} | mode={dataset_mode} | "
        f"shape={shape_type} | scale={obstacle_scale}"
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    shape_dir = save_dir / str(shape_type)
    shape_dir.mkdir(parents=True, exist_ok=True)

    save_path = shape_dir / f"ae_recon_{dataset_mode}_{shape_type}_{idx:04d}.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)

    p.add_argument(
        "--dataset_mode",
        choices=["ae", "partial"],
        default="partial",
    )
    p.add_argument(
        "--model_type",
        choices=["pointnet2", "folding"],
        default="pointnet2",
    )

    p.add_argument("--per_shape", type=int, default=4)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--robot_index", type=int, default=0)
    p.add_argument("--save_dir", default="out/ae_recon_vis_by_shape")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["cfg"]

    ds = build_dataset(args, cfg)

    model = PointNet2AutoEncoder(
        encoder_mode=cfg["model"].get("encoder_mode", "ssg"),
        decoder_mode=cfg["model"].get("decoder_mode", "mlp"),
        latent_dim=cfg["model"]["latent_dim"],
        input_channels=cfg["model"].get("input_channels", 0),
        target_num_points=cfg["data"]["target_num_points"],
        output_dim=cfg["model"].get("output_dim", 3),
        base_radius=cfg["model"].get("base_radius", 1.0),
        npoint1=cfg["model"].get("npoint1", 32),
        npoint2=cfg["model"].get("npoint2", 16),
        hidden_dim=cfg["model"].get("hidden_dim", 128),
        k_cov=cfg["model"].get("k_cov", 32),
        k_agg=cfg["model"].get("k_agg", 16),
        use_attention=cfg["model"].get("use_attention", True),
        decoder_hidden_dim=cfg["model"].get("decoder_hidden_dim", 512),
        folding_grid_dim=cfg["model"].get("folding_grid_dim", 1),
        folding_num_folds=cfg["model"].get("folding_num_folds", 2),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

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

    for idx in selected_indices:
        visualize_sample(
            model=model,
            ds=ds,
            idx=idx,
            dataset_mode=args.dataset_mode,
            save_dir=save_dir,
            robot_index=args.robot_index,
        )


if __name__ == "__main__":
    main()