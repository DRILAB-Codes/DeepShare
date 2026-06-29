#=======================================
# 오토인코더 학습 베이스
# 폴딩넷 + 챔퍼 로스만 사용
# 실행 옵션 적음
#========================================
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
from models.autoencoder import PointNet2AutoEncoder


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_loader(cfg, split, batch_size, workers, seed):
    ds = ObstacleBoundaryAEDataset(
        data_dir=cfg[f"{split}_dir"],
        input_num_points=cfg["input_num_points"],
        target_num_points=cfg["target_num_points"],
        normalize=cfg.get("normalize", True),
        seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=workers, drop_last=(split == "train"))


def evaluate(model, loader, device):
    model.eval(); vals = []
    with torch.no_grad():
        for b in loader:
            pred, _ = model(b["input"].to(device))
            loss = chamfer_distance(pred, b["target"].to(device))
            vals.append(float(loss.item()))
    return float(np.mean(vals)) if vals else float("nan")


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg["training"].get("seed", 0)); set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["training"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader = make_loader(cfg["data"], "train", cfg["training"]["batch_size"], cfg["training"]["num_workers"], seed)
    val_loader = make_loader(cfg["data"], "val", cfg["training"]["batch_size"], cfg["training"]["num_workers"], seed + 1234)
    model = PointNet2AutoEncoder(
        encoder_mode=cfg["model"]["encoder_mode"],
        latent_dim=cfg["model"]["latent_dim"],
        target_num_points=cfg["data"]["target_num_points"],
        base_radius=cfg["model"].get("base_radius", 1.0),
        npoint1=cfg["model"].get("npoint1", 32),
        npoint2=cfg["model"].get("npoint2", 16),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    best = float("inf")
    hist = []

    patience = int(cfg["training"].get("patience", 30))
    min_delta = float(cfg["training"].get("min_delta", 1e-5))
    bad_epochs = 0

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train(); losses = []
        for b in train_loader:
            pred, _ = model(b["input"].to(device))
            loss = chamfer_distance(pred, b["target"].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        tr = float(np.mean(losses)) if losses else float("nan")
        va = evaluate(model, val_loader, device)
        hist.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"[{epoch:04d}] train={tr:.6f} val={va:.6f}")
        if va < best - min_delta:
            best = va
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_loss": va},
                out_dir / "model_best.pt",
            )
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best:.6f}")
            break
            
        if epoch % int(cfg["training"].get("save_every", 10)) == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_loss": va}, out_dir / f"model_epoch_{epoch:04d}.pt")
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)
    print(f"Best val loss: {best:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/ae_config.yaml")
    main(p.parse_args().config)
