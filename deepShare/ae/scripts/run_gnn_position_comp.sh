#!/bin/bash
set -e

BASE_CONFIG=configs/gnn_global_config.yaml
TRAIN_SCRIPT=train_gnn_latent.py
BASE_OUT=out/gnn_position_compare

mkdir -p "${BASE_OUT}"
mkdir -p configs/generated

send_slack() {
  local MSG="$1"

  SLACK_MESSAGE="${MSG}" python - <<'PY'
import os
import socket
import requests

webhook = os.environ.get("SLACK_WEBHOOK_URL")
msg = os.environ.get("SLACK_MESSAGE", "")

if not webhook:
    print("[Slack skipped] SLACK_WEBHOOK_URL not set.")
    raise SystemExit(0)

payload = {
    "text": f"[{socket.gethostname()}] {msg}"
}

try:
    r = requests.post(webhook, json=payload, timeout=10)
    print(f"[Slack] {r.status_code}")
except Exception as e:
    print(f"[Slack error] {e}")
PY
}

send_slack "🚀 GNN robot position comparison started
Output: ${BASE_OUT}"

for POSITION in false true
do
  if [ "${POSITION}" = "true" ]; then
    RUN_NAME=with_position
  else
    RUN_NAME=without_position
  fi

  RUN_DIR="${BASE_OUT}/${RUN_NAME}"
  CFG="configs/generated/robot_gnn_${RUN_NAME}.yaml"

  echo "======================================"
  echo "Use robot XY : ${POSITION}"
  echo "Run name     : ${RUN_NAME}"
  echo "Run dir      : ${RUN_DIR}"
  echo "Config       : ${CFG}"
  echo "======================================"

  send_slack "▶️ Start run
Use robot XY: ${POSITION}
Run dir: ${RUN_DIR}"

  python - <<PY
import yaml
from pathlib import Path

base_config = "${BASE_CONFIG}"
out_config = "${CFG}"
run_dir = "${RUN_DIR}"

use_robot_xy = "${POSITION}".lower() == "true"

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["model"]["use_robot_xy"] = use_robot_xy
cfg["training"]["out_dir"] = run_dir

cfg["training"]["epochs"] = 200
cfg["training"]["lr"] = 0.0001
cfg["training"]["save_every"] = 30
cfg["training"]["patience"] = 30
cfg["training"]["min_delta"] = 0.0005

# 기본값이라도 실험 재현성을 위해 명시
cfg["training"]["latent_loss_type"] = cfg["training"].get(
    "latent_loss_type",
    "mse",
)
cfg["training"]["grad_clip"] = cfg["training"].get(
    "grad_clip",
    1.0,
)

Path(out_config).parent.mkdir(parents=True, exist_ok=True)

with open(out_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        cfg,
        f,
        sort_keys=False,
        allow_unicode=True,
    )

print(f"Generated config: {out_config}")
print(f"use_robot_xy: {use_robot_xy}")
print(f"out_dir: {run_dir}")
PY

  if python "${TRAIN_SCRIPT}" --config "${CFG}"; then
    BEST_INFO=$(python - <<PY
import json
from pathlib import Path

hist_path = Path("${RUN_DIR}") / "history.json"

if not hist_path.exists():
    print("history.json not found")
else:
    with open(hist_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    if not hist:
        print("history is empty")
    else:
        best = min(
            hist,
            key=lambda r: r.get("val_loss", float("inf")),
        )

        epoch = best.get("epoch", "unknown")
        val_loss = best.get("val_loss", "unknown")
        val_mse = best.get("val_latent_mse", "unknown")
        val_cos = best.get("val_latent_cosine", "unknown")
        val_steps = best.get("val_steps", "unknown")
        val_conv = best.get("val_conv", "unknown")

        if isinstance(val_loss, float):
            val_loss = f"{val_loss:.6f}"
        if isinstance(val_mse, float):
            val_mse = f"{val_mse:.6f}"
        if isinstance(val_cos, float):
            val_cos = f"{val_cos:.4f}"
        if isinstance(val_steps, float):
            val_steps = f"{val_steps:.2f}"
        if isinstance(val_conv, float):
            val_conv = f"{val_conv:.2%}"

        print(
            f"best_epoch={epoch}, "
            f"val_loss={val_loss}, "
            f"val_mse={val_mse}, "
            f"val_cos={val_cos}, "
            f"val_steps={val_steps}, "
            f"val_conv={val_conv}"
        )
PY
)

    send_slack "✅ Finished run
Use robot XY: ${POSITION}
Run dir: ${RUN_DIR}
${BEST_INFO}"
  else
    send_slack "❌ Failed run
Use robot XY: ${POSITION}
Run dir: ${RUN_DIR}"
    exit 1
  fi
done

send_slack "🎉 GNN robot position comparison finished
Results saved to ${BASE_OUT}"

echo "Done. Results saved to ${BASE_OUT}"