# 장애물 본래 스케일 사용
# 인코더 : 
#     포인트 좌표 + 2 hop 이웃의 분산
#     압축 3회
#     mlp 1회
# 디코더 : 
#     폴딩넷, 
# 로스 : 
#     챔퍼 + 형태

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.obstacle_dataset import ObstacleBoundaryAEDataset
from losses.chamfer import chamfer_distance
from losses.local_covariance import local_variance_loss
from models.autoencoder import PointNet2AutoEncoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(cfg, split, batch_size, workers, seed):
    ds = ObstacleBoundaryAEDataset(
        data_dir=cfg[f"{split}_dir"],
        input_num_points=cfg["input_num_points"],
        target_num_points=cfg["target_num_points"],
        normalize=cfg.get("normalize", True),
        seed=seed,
        use_saved_boundary=cfg.get("use_saved_boundary", True),
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        drop_last=(split == "train"),
    )


def evaluate(model, loader, device, cfg):
    model.eval()
    vals = []

    with torch.no_grad():
        for b in loader:
            x = b["input"].to(device)
            y = b["target"].to(device)

            pred, _ = model(x)

            cd_loss = chamfer_distance(pred, y)
            lv_loss = local_variance_loss(pred, y, k=16)

            loss = cd_loss + float(cfg["training"].get("local_variance_weight", 0.05)) * lv_loss

            vals.append(float(loss.item()))

    return float(np.mean(vals)) if vals else float("nan")


def build_model(cfg):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model = PointNet2AutoEncoder(
        encoder_mode=model_cfg.get("encoder_mode", "ssg"),
        decoder_mode=model_cfg.get("decoder_mode", "mlp"),

        latent_dim=model_cfg.get("latent_dim", 128),
        input_channels=model_cfg.get("input_channels", 0),

        target_num_points=data_cfg["target_num_points"],
        output_dim=model_cfg.get("output_dim", 3),

        # 기존 PointNet2 encoder 옵션
        base_radius=model_cfg.get("base_radius", 1.0),
        npoint1=model_cfg.get("npoint1", 32),
        npoint2=model_cfg.get("npoint2", 16),

        # 새 PointEmbedEncoder 옵션
        hidden_dim=model_cfg.get("hidden_dim", 128),
        k_cov=model_cfg.get("k_cov", 32),
        k_agg=model_cfg.get("k_agg", 16),
        use_attention=model_cfg.get("use_attention", True),

        # Folding decoder 옵션
        decoder_hidden_dim=model_cfg.get("decoder_hidden_dim", 512),
        folding_grid_dim=model_cfg.get("folding_grid_dim", 1),
        folding_num_folds=model_cfg.get("folding_num_folds", 2),
    )

    return model


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["training"].get("seed", 0))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["training"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader = make_loader(
        cfg["data"],
        "train",
        cfg["training"]["batch_size"],
        cfg["training"]["num_workers"],
        seed,
    )

    val_loader = make_loader(
        cfg["data"],
        "val",
        cfg["training"]["batch_size"],
        cfg["training"]["num_workers"],
        seed + 1234,
    )

    model = build_model(cfg).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    best = float("inf")
    hist = []

    patience = int(cfg["training"].get("patience", 30))
    min_delta = float(cfg["training"].get("min_delta", 1e-5))
    bad_epochs = 0

    epochs = int(cfg["training"]["epochs"])
    save_every = int(cfg["training"].get("save_every", 10))

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for b in train_loader:
            x = b["input"].to(device)
            y = b["target"].to(device)

            pred, z = model(x)

            cd_loss = chamfer_distance(pred, y)
            lv_loss = local_variance_loss(pred, y, k=16)

            loss = cd_loss + float(cfg["training"].get("local_variance_weight", 0.05)) * lv_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss = evaluate(model, val_loader, device, cfg)

        hist.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(f"[{epoch:04d}] train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best - min_delta:
            best = val_loss
            bad_epochs = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                out_dir / "model_best.pt",
            )
        else:
            bad_epochs += 1

        if epoch % save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                out_dir / f"model_epoch_{epoch:04d}.pt",
            )

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best:.6f}")
            break

    print(f"Best val loss: {best:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ae_config.yaml")
    args = p.parse_args()

    main(args.config)