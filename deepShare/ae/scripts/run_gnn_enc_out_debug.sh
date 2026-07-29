#!/bin/bash
set -e

BASE_CONFIG=configs/gnn_latent_config.yaml
TRAIN_SCRIPT=train_gnn_latent.py
BASE_OUT=out/gnn_attention_compare

mkdir -p ${BASE_OUT}
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

send_slack "🚀 GNN attention compare started
Output: ${BASE_OUT}"

for AGG in attention dual_attention
do
  RUN_DIR=${BASE_OUT}/${AGG}
  CFG=configs/generated/robot_gnn_${AGG}.yaml

  echo "======================================"
  echo "Aggregator: ${AGG}"
  echo "Run dir   : ${RUN_DIR}"
  echo "Config    : ${CFG}"
  echo "======================================"

  send_slack "▶️ Start run
Aggregator: ${AGG}
Run dir: ${RUN_DIR}"

  python - <<PY
import yaml
from pathlib import Path

base_config = "${BASE_CONFIG}"
out_config = "${CFG}"
agg = "${AGG}"
run_dir = "${RUN_DIR}"

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["model"]["aggregator"] = agg
cfg["training"]["out_dir"] = run_dir

cfg["training"]["epochs"] = 200
cfg["training"]["lr"] = 0.0001
cfg["training"]["save_every"] = 30
cfg["training"]["patience"] = 30
cfg["training"]["min_delta"] = 0.0005

Path(out_config).parent.mkdir(parents=True, exist_ok=True)
with open(out_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  if python ${TRAIN_SCRIPT} --config ${CFG}; then
    BEST_INFO=$(python - <<PY
import json
from pathlib import Path

hist_path = Path("${RUN_DIR}") / "history.json"

if not hist_path.exists():
    print("history.json not found")
else:
    hist = json.load(open(hist_path, "r", encoding="utf-8"))
    if not hist:
        print("history is empty")
    else:
        best = min(hist, key=lambda r: r.get("val_loss", float("inf")))
        epoch = best.get("epoch", "unknown")
        val_loss = best.get("val_loss", "unknown")
        val_steps = best.get("val_steps", "unknown")
        val_conv = best.get("val_conv", "unknown")

        if isinstance(val_loss, float):
            val_loss = f"{val_loss:.6f}"
        if isinstance(val_steps, float):
            val_steps = f"{val_steps:.2f}"
        if isinstance(val_conv, float):
            val_conv = f"{val_conv:.2%}"

        print(f"best_epoch={epoch}, val_loss={val_loss}, val_steps={val_steps}, val_conv={val_conv}")
PY
)

    send_slack "✅ Finished run
Aggregator: ${AGG}
Run dir: ${RUN_DIR}
${BEST_INFO}"
  else
    send_slack "❌ Failed run
Aggregator: ${AGG}
Run dir: ${RUN_DIR}"
    exit 1
  fi
done

send_slack "🎉 GNN attention compare finished
Results saved to ${BASE_OUT}"

echo "Done. Results saved to ${BASE_OUT}"