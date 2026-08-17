#!/bin/bash
set -e

# ============================================================
# GNN Decoder Comparison
# Frozen Folding AE Decoder vs Frozen Snowflake AE Decoder
# ============================================================

BASE_CONFIG="configs/gnn_config.yaml"
TRAIN_SCRIPT="train_gnn.py"

BASE_OUT="out/gnn_decoder_compare"
GENERATED_CONFIG_DIR="configs/generated"


# Slack Incoming Webhook URL
# 주의: Git에 올라가지 않도록 할 것
# 공용 채널
# SLACK_WEBHOOK_URL="https://hooks.슬랙.com/services/T0940M7L40J/B0B4W692QAZ/fPibHLi1YN0G1nx6aDYk8xUW"
# 내 채널
SLACK_WEBHOOK_URL="https://hooks.슬랙.com/services/T0940M7L40J/B0B8L6UG91P/Bv395qR8nt2FtKqepzGliDfT" 


mkdir -p "${BASE_OUT}"
mkdir -p "${GENERATED_CONFIG_DIR}"


# ============================================================
# Slack
# ============================================================

send_slack() {
    local MSG="$1"

    SLACK_MESSAGE="${MSG}" \
    SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL}" \
    python - <<'PY'
import os
import socket
import requests

webhook = os.environ.get("SLACK_WEBHOOK_URL")
msg = os.environ.get("SLACK_MESSAGE", "")

if not webhook:
    print("[Slack skipped] webhook is empty.")
    raise SystemExit(0)

payload = {
    "text": f"[{socket.gethostname()}] {msg}"
}

try:
    r = requests.post(
        webhook,
        json=payload,
        timeout=10,
    )
    print(f"[Slack] status={r.status_code}")

except Exception as e:
    print(f"[Slack error] {e}")
PY
}


# ============================================================
# 전체 실험 시작
# ============================================================

send_slack "🚀 GNN decoder comparison started
Folding AE: out/ae_ssg/model_best.pt
Snow AE: out/ae_msg_snow/model_best.pt
Output: ${BASE_OUT}"


# ============================================================
# Folding / Snow 각각 GNN 학습
# ============================================================

for DECODER in snow folding
do

    # --------------------------------------------------------
    # 사용할 frozen AE decoder checkpoint 결정
    # --------------------------------------------------------

    if [ "${DECODER}" = "folding" ]; then
        AE_CHECKPOINT="out/ae_ssg/model_best.pt"

    elif [ "${DECODER}" = "snow" ]; then
        AE_CHECKPOINT="out/ae_msg_snow/model_best.pt"

    else
        echo "Unknown decoder: ${DECODER}"
        exit 1
    fi


    RUN_NAME="${DECODER}"
    RUN_DIR="${BASE_OUT}/${RUN_NAME}"

    CFG="${GENERATED_CONFIG_DIR}/gnn_${RUN_NAME}.yaml"


    echo
    echo "======================================"
    echo "GNN decoder comparison"
    echo "Decoder       : ${DECODER}"
    echo "AE checkpoint : ${AE_CHECKPOINT}"
    echo "Run dir       : ${RUN_DIR}"
    echo "Config        : ${CFG}"
    echo "======================================"

    mkdir -p "${RUN_DIR}"


    # ========================================================
    # Config 생성
    # ========================================================

    python - <<PY
import yaml
from pathlib import Path

base_config = "${BASE_CONFIG}"
out_config = "${CFG}"
run_dir = "${RUN_DIR}"
ae_checkpoint = "${AE_CHECKPOINT}"
decoder = "${DECODER}"

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# ------------------------------------------------------------
# 실험별 변경값
# ------------------------------------------------------------

# GNN 자체 decoder_mode를 바꾸는 것이 아니라
# 사용할 AE checkpoint를 바꾼다.
cfg["ae"]["checkpoint"] = ae_checkpoint
cfg["ae"]["freeze_decoder"] = True

# 결과 저장 위치 분리
cfg["training"]["out_dir"] = run_dir


# ------------------------------------------------------------
# 비교 실험 공통 조건
# ------------------------------------------------------------

cfg["training"]["epochs"] = 200
cfg["training"]["lr"] = 0.0001
cfg["training"]["weight_decay"] = 0.00001

cfg["training"]["batch_size"] = 1
cfg["training"]["seed"] = 42
cfg["training"]["save_every"] = 50


Path(out_config).parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(out_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        cfg,
        f,
        sort_keys=False,
        allow_unicode=True,
    )


print(f"Generated config : {out_config}")
print(f"Experiment       : {decoder}")
print(f"AE checkpoint    : {ae_checkpoint}")
print(f"GNN output       : {run_dir}")
PY


    # ========================================================
    # 각 run 시작
    # ========================================================

    send_slack "▶️ GNN training started
Decoder: ${DECODER}
AE checkpoint: ${AE_CHECKPOINT}
Output: ${RUN_DIR}"


    # ========================================================
    # GNN 학습
    # ========================================================

    if python "${TRAIN_SCRIPT}" --config "${CFG}"; then

        # ====================================================
        # best epoch / metrics 추출
        # ====================================================

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
            key=lambda r: r.get(
                "val_loss",
                float("inf"),
            ),
        )

        epoch = best.get(
            "epoch",
            "unknown",
        )

        train_loss = best.get(
            "train_loss",
            "unknown",
        )

        val_loss = best.get(
            "val_loss",
            "unknown",
        )

        node_cd = best.get(
            "val_node_chamfer_mean",
            "unknown",
        )

        consensus_gap = best.get(
            "val_consensus_gap",
            "unknown",
        )

        val_steps = best.get(
            "val_steps",
            "unknown",
        )

        val_conv = best.get(
            "val_conv",
            "unknown",
        )


        if isinstance(train_loss, float):
            train_loss = f"{train_loss:.6f}"

        if isinstance(val_loss, float):
            val_loss = f"{val_loss:.6f}"

        if isinstance(node_cd, float):
            node_cd = f"{node_cd:.6f}"

        if isinstance(consensus_gap, float):
            consensus_gap = f"{consensus_gap:.6f}"

        if isinstance(val_steps, float):
            val_steps = f"{val_steps:.2f}"

        if isinstance(val_conv, float):
            val_conv = f"{val_conv:.2%}"


        print(
            f"best_epoch={epoch}, "
            f"train_loss={train_loss}, "
            f"val_loss={val_loss}, "
            f"node_cd={node_cd}, "
            f"gap={consensus_gap}, "
            f"steps={val_steps}, "
            f"conv={val_conv}"
        )
PY
)

        send_slack "✅ GNN training finished
Decoder: ${DECODER}
AE checkpoint: ${AE_CHECKPOINT}
Output: ${RUN_DIR}
${BEST_INFO}"

    else

        send_slack "❌ GNN training failed
Decoder: ${DECODER}
AE checkpoint: ${AE_CHECKPOINT}
Output: ${RUN_DIR}"

        exit 1
    fi

done


# ============================================================
# 전체 완료
# ============================================================

send_slack "🎉 GNN decoder comparison finished
Folding: ${BASE_OUT}/folding
Snow: ${BASE_OUT}/snow"

echo
echo "======================================"
echo "GNN decoder comparison finished"
echo "Folding : ${BASE_OUT}/folding"
echo "Snow    : ${BASE_OUT}/snow"
echo "======================================"