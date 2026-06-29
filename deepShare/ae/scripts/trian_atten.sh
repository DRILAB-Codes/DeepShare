#!/bin/bash
set -e

BASE_CONFIG=configs/gnn_config.yaml
TRAIN_SCRIPT=train_gnn.py
BASE_OUT=out/gnn_attention_compare

mkdir -p ${BASE_OUT}
mkdir -p configs/generated

for AGG in attention dual_attention
do
  RUN_DIR=${BASE_OUT}/${AGG}
  CFG=configs/generated/robot_gnn_${AGG}.yaml

  echo "======================================"
  echo "Aggregator: ${AGG}"
  echo "Run dir   : ${RUN_DIR}"
  echo "Config    : ${CFG}"
  echo "======================================"

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

# 비교 실험용 공통 설정
cfg["training"]["epochs"] = 200
cfg["training"]["lr"] = 0.0001
cfg["training"]["save_every"] = 30
cfg["training"]["patience"] = 30
cfg["training"]["min_delta"] = 0.0005

Path(out_config).parent.mkdir(parents=True, exist_ok=True)
with open(out_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  python ${TRAIN_SCRIPT} --config ${CFG}
done

echo "Done. Results saved to ${BASE_OUT}"