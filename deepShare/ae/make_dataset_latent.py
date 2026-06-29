# add_ae_latent_to_dataset.py

import argparse
import json
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from models.autoencoder import PointNet2AutoEncoder


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_ae(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]

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
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, cfg


@torch.no_grad()
def encode_boundary(ae, boundary_points, device):
    x = torch.tensor(boundary_points, dtype=torch.float32, device=device)
    x = x.unsqueeze(0)  # [1, Q, 3]

    # PointNet2AutoEncoder forward: pred, z
    _, z = ae(x)

    return z.squeeze(0).detach().cpu().tolist()


def process_one_file(src_path, dst_path, ae, device, overwrite=False):
    sample = load_json(src_path)

    if "obstacle" not in sample:
        raise KeyError(f"{src_path}: missing obstacle field")

    obstacle = sample["obstacle"]

    if "boundary_points" not in obstacle:
        raise KeyError(f"{src_path}: missing obstacle.boundary_points")

    if "ae_latent" in obstacle and not overwrite:
        save_json(sample, dst_path)
        return "skipped_existing"

    z = encode_boundary(ae, obstacle["boundary_points"], device)

    obstacle["ae_latent"] = z
    obstacle["ae_latent_dim"] = len(z)

    save_json(sample, dst_path)
    return "encoded"


def process_dir(src_dir, dst_dir, ae, device, overwrite=False):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    files = sorted(src_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"No json files found in {src_dir}")

    encoded = 0
    skipped = 0

    for src_path in tqdm(files, desc=f"{src_dir} -> {dst_dir}"):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel

        result = process_one_file(
            src_path=src_path,
            dst_path=dst_path,
            ae=ae,
            device=device,
            overwrite=overwrite,
        )

        if result == "encoded":
            encoded += 1
        else:
            skipped += 1

    print(f"Done: {src_dir}")
    print(f"  encoded: {encoded}")
    print(f"  skipped: {skipped}")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--ae_checkpoint", required=True)

    p.add_argument("--src_train", default="data/ae/train_mixed")
    p.add_argument("--src_val", default="data/ae/val_mixed")
    p.add_argument("--src_test", default="data/ae/test_mixed")

    p.add_argument("--dst_train", default="data/gnn_latent/train_mixed")
    p.add_argument("--dst_val", default="data/gnn_latent/val_mixed")
    p.add_argument("--dst_test", default="data/gnn_latent/test_mixed")

    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ae, ae_cfg = build_ae(args.ae_checkpoint, device)

    print(f"Loaded AE checkpoint: {args.ae_checkpoint}")
    print(f"AE latent_dim: {ae_cfg['model']['latent_dim']}")
    print(f"Device: {device}")

    process_dir(args.src_train, args.dst_train, ae, device, args.overwrite)
    process_dir(args.src_val, args.dst_val, ae, device, args.overwrite)
    process_dir(args.src_test, args.dst_test, ae, device, args.overwrite)


if __name__ == "__main__":
    main()