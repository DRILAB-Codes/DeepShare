#!/bin/bash
set -e
set -o pipefail


# ============================================================
# Folding Decoder Training Comparison
#
# Experiment 1
#   AutoEncoder pretraining
#
# Experiment 2
#   GNN + pretrained frozen Folding decoder
#
# Experiment 3
#   GNN + random Folding decoder
#   end-to-end training
# ============================================================


# ============================================================
# 기본 설정
# ============================================================

AE_TRAIN_SCRIPT="train_ae.py"
GNN_TRAIN_SCRIPT="train_gnn.py"

BASE_AE_CONFIG="configs/ae_config.yaml"
BASE_GNN_CONFIG="configs/gnn_config.yaml"


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

TRAIN_DIR="data/ae/train"
VAL_DIR="data/ae/val"
TEST_DIR="data/ae/test"


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

BASE_OUT="out/folding_pretrained_vs_e2e"

AE_OUT="${BASE_OUT}/ae_pretrain"
GNN_FROZEN_OUT="${BASE_OUT}/gnn_frozen"
GNN_E2E_OUT="${BASE_OUT}/gnn_end_to_end"


# ------------------------------------------------------------
# Generated configs
# ------------------------------------------------------------

GENERATED_CONFIG_DIR="configs/generated/folding_pretrained_vs_e2e"

AE_CFG="${GENERATED_CONFIG_DIR}/ae_folding.yaml"
GNN_FROZEN_CFG="${GENERATED_CONFIG_DIR}/gnn_frozen.yaml"
GNN_E2E_CFG="${GENERATED_CONFIG_DIR}/gnn_end_to_end.yaml"


# ------------------------------------------------------------
# AE checkpoint
# ------------------------------------------------------------

AE_CHECKPOINT="${AE_OUT}/model_best.pt"


# ============================================================
# 학습 공통 설정
# ============================================================

SEED=42

AE_EPOCHS=200

GNN_EPOCHS=200
GNN_LR=0.0001
GNN_WEIGHT_DECAY=0.00001
GNN_BATCH_SIZE=1
GNN_SAVE_EVERY=50


# ============================================================
# Slack 설정
#
# 권장:
#
# export SLACK_WEBHOOK_URL="슬랙 URL"
# ./run_folding_pretrained_vs_e2e.sh
#
# ============================================================

SLACK_WEBHOOK_URL="https://hooks.슬랙.com/services/T0940M7L40J/B0B8L6UG91P/Bv395qR8nt2FtKqepzGliDfT"

# 몇 epoch마다 진행상황 전송할지
SLACK_EVERY=10


mkdir -p "${BASE_OUT}"
mkdir -p "${GENERATED_CONFIG_DIR}"


# ============================================================
# Slack 전송 함수
# ============================================================

send_slack() {

    local MSG="$1"

    SLACK_MESSAGE="${MSG}" \
    SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL}" \
    python - <<'PY'

import json
import os
import socket
import urllib.request


webhook = os.environ.get(
    "SLACK_WEBHOOK_URL",
    "",
)

msg = os.environ.get(
    "SLACK_MESSAGE",
    "",
)


if not webhook:
    print("[Slack skipped] webhook is empty.")
    raise SystemExit(0)


payload = {
    "text": f"[{socket.gethostname()}] {msg}"
}

data = json.dumps(
    payload,
).encode("utf-8")


req = urllib.request.Request(
    webhook,
    data=data,
    headers={
        "Content-Type": "application/json"
    },
    method="POST",
)


try:

    with urllib.request.urlopen(
        req,
        timeout=10,
    ) as response:

        print(
            f"[Slack] status="
            f"{response.status}"
        )

except Exception as e:

    # Slack 실패 때문에 학습 자체를 중단하지는 않음
    print(
        f"[Slack error] {e}"
    )

PY
}


# ============================================================
# history.json best 결과 추출
# ============================================================

get_best_info() {

    local RUN_DIR="$1"

    python - "${RUN_DIR}" <<'PY'

import json
import sys
from pathlib import Path


run_dir = Path(sys.argv[1])
hist_path = run_dir / "history.json"


if not hist_path.exists():

    print("history.json not found")
    raise SystemExit(0)


with open(
    hist_path,
    "r",
    encoding="utf-8",
) as f:

    hist = json.load(f)


if not hist:

    print("history is empty")
    raise SystemExit(0)


best = min(
    hist,
    key=lambda r: r.get(
        "val_loss",
        float("inf"),
    ),
)


parts = [
    f"best_epoch={best.get('epoch', 'unknown')}",
]


keys = [
    ("train_loss", "train"),
    ("val_loss", "val"),
    ("val_node_chamfer_mean", "node_cd"),
    ("val_consensus_gap", "gap"),
    ("val_steps", "steps"),
    ("val_conv", "conv"),
]


for key, name in keys:

    if key not in best:
        continue

    value = best[key]

    if isinstance(value, float):

        if key == "val_conv":
            value = f"{value:.2%}"

        elif key == "val_steps":
            value = f"{value:.2f}"

        else:
            value = f"{value:.6f}"

    parts.append(
        f"{name}={value}"
    )


print(
    ", ".join(parts)
)

PY
}


# ============================================================
# Training runner
#
# - stdout 저장
# - 터미널 출력
# - SLACK_EVERY epoch마다 Slack 전송
# - 실패하면 즉시 종료
# ============================================================

run_training() {

    local LABEL="$1"
    local RUN_DIR="$2"
    shift 2

    mkdir -p "${RUN_DIR}"

    local LOG_FILE="${RUN_DIR}/train.log"


    send_slack "▶️ ${LABEL} started
Output: ${RUN_DIR}"


    set +e


    "$@" 2>&1 \
    | tee "${LOG_FILE}" \
    | while IFS= read -r LINE
    do

        echo "${LINE}"


        # ----------------------------------------------------
        # 학습 코드의
        #
        # [0001] ...
        # [0010] ...
        #
        # 형식 로그 감지
        # ----------------------------------------------------

        if [[ "${LINE}" =~ ^\[([0-9]{4})\] ]]; then

            EPOCH=$((10#${BASH_REMATCH[1]}))


            if (( EPOCH == 1 || EPOCH % SLACK_EVERY == 0 )); then

                send_slack "📈 ${LABEL}
${LINE}"

            fi

        fi

    done


    STATUS=${PIPESTATUS[0]}


    set -e


    if [ "${STATUS}" -ne 0 ]; then

        send_slack "❌ ${LABEL} failed
Output: ${RUN_DIR}
Log: ${LOG_FILE}"

        echo
        echo "======================================"
        echo "FAILED: ${LABEL}"
        echo "======================================"

        exit "${STATUS}"

    fi


    BEST_INFO=$(get_best_info "${RUN_DIR}")


    send_slack "✅ ${LABEL} finished
Output: ${RUN_DIR}
${BEST_INFO}"


    echo
    echo "======================================"
    echo "FINISHED: ${LABEL}"
    echo "${BEST_INFO}"
    echo "======================================"
    echo
}


# ============================================================
# Config 생성
# ============================================================

python - <<PY

import copy
from pathlib import Path

import yaml


base_ae_config = "${BASE_AE_CONFIG}"
base_gnn_config = "${BASE_GNN_CONFIG}"

ae_config_path = "${AE_CFG}"
gnn_frozen_config_path = "${GNN_FROZEN_CFG}"
gnn_e2e_config_path = "${GNN_E2E_CFG}"


# ============================================================
# AE config
# ============================================================

with open(
    base_ae_config,
    "r",
    encoding="utf-8",
) as f:
    ae_cfg = yaml.safe_load(f)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

ae_cfg["data"]["train_dir"] = "${TRAIN_DIR}"
ae_cfg["data"]["val_dir"] = "${VAL_DIR}"
ae_cfg["data"]["test_dir"] = "${TEST_DIR}"

ae_cfg["data"]["input_num_points"] = 64
ae_cfg["data"]["target_num_points"] = 128
ae_cfg["data"]["normalize"] = True


# ------------------------------------------------------------
# AE model
#
# 최신 dataset point:
#   [x, y, z, extra_feature]
#
# PointNet2의 input_channels는 전체 차원이 아니라
# xyz 외의 추가 feature 개수이므로 1
# ------------------------------------------------------------

ae_cfg["model"]["encoder_mode"] = "msg"
ae_cfg["model"]["decoder_mode"] = "folding"

ae_cfg["model"]["latent_dim"] = 128

ae_cfg["model"]["input_channels"] = 1
ae_cfg["model"]["output_dim"] = 3

ae_cfg["model"]["base_radius"] = 1.0
ae_cfg["model"]["npoint1"] = 32
ae_cfg["model"]["npoint2"] = 16


# ------------------------------------------------------------
# AE training
# ------------------------------------------------------------

ae_cfg["training"]["out_dir"] = "${AE_OUT}"

ae_cfg["training"]["batch_size"] = 128
ae_cfg["training"]["epochs"] = ${AE_EPOCHS}
ae_cfg["training"]["lr"] = 0.001

ae_cfg["training"]["patience"] = 20
ae_cfg["training"]["min_delta"] = 0.00001

ae_cfg["training"]["num_workers"] = 4
ae_cfg["training"]["save_every"] = 10

ae_cfg["training"]["seed"] = ${SEED}


Path(
    ae_config_path
).parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    ae_config_path,
    "w",
    encoding="utf-8",
) as f:

    yaml.safe_dump(
        ae_cfg,
        f,
        sort_keys=False,
        allow_unicode=True,
    )


# ============================================================
# GNN base config
# ============================================================

with open(
    base_gnn_config,
    "r",
    encoding="utf-8",
) as f:
    gnn_base = yaml.safe_load(f)


# ------------------------------------------------------------
# Dataset
#
# 최신 통합 dataset을 AE / GNN 모두 동일하게 사용
#
# 완전 분산형 조건:
#   각 로봇은 자신의 local frame 관측만 사용
#   world frame 변환 사용하지 않음
# ------------------------------------------------------------

gnn_base["data"]["train_dir"] = "${TRAIN_DIR}"
gnn_base["data"]["val_dir"] = "${VAL_DIR}"
gnn_base["data"]["test_dir"] = "${TEST_DIR}"

gnn_base["data"]["input_num_points"] = 64
gnn_base["data"]["target_num_points"] = 128

gnn_base["data"]["include_miss"] = False
gnn_base["data"]["normalize"] = True

# 완전 분산형이므로 반드시 False
gnn_base["data"]["use_world_frame"] = False


# ------------------------------------------------------------
# GNN model
#
# 최신 dataset의 point 하나:
#   [x, y, z, extra_feature]
#
# GNN input_point_dim은 전체 point feature 차원
# ------------------------------------------------------------

gnn_base["model"]["input_point_dim"] = 3

# AE와 latent dimension 동일하게 강제
gnn_base["model"]["latent_dim"] = (
    ae_cfg["model"]["latent_dim"]
)


# ------------------------------------------------------------
# GNN 공통 training 조건
#
# Frozen / E2E 비교에서 아래 조건은 완전히 동일
# ------------------------------------------------------------

gnn_base["training"]["epochs"] = ${GNN_EPOCHS}
gnn_base["training"]["lr"] = ${GNN_LR}
gnn_base["training"]["weight_decay"] = ${GNN_WEIGHT_DECAY}

gnn_base["training"]["batch_size"] = ${GNN_BATCH_SIZE}
gnn_base["training"]["num_workers"] = 0

gnn_base["training"]["seed"] = ${SEED}
gnn_base["training"]["save_every"] = ${GNN_SAVE_EVERY}


# ============================================================
# Experiment 2
#
# GNN + pretrained Folding decoder
# decoder frozen
# ============================================================

gnn_frozen = copy.deepcopy(
    gnn_base
)

gnn_frozen.setdefault(
    "ae",
    {},
)

gnn_frozen["ae"]["mode"] = "pretrained"

gnn_frozen["ae"]["checkpoint"] = (
    "${AE_CHECKPOINT}"
)

gnn_frozen["ae"]["freeze_decoder"] = True

gnn_frozen["training"]["out_dir"] = (
    "${GNN_FROZEN_OUT}"
)


with open(
    gnn_frozen_config_path,
    "w",
    encoding="utf-8",
) as f:

    yaml.safe_dump(
        gnn_frozen,
        f,
        sort_keys=False,
        allow_unicode=True,
    )


# ============================================================
# Experiment 3
#
# GNN + random initialized Folding decoder
# End-to-End
#
# AE checkpoint weight는 사용하지 않음.
# AE config는 Folding decoder의 구조 정보만 사용.
# ============================================================

gnn_e2e = copy.deepcopy(
    gnn_base
)

gnn_e2e.setdefault(
    "ae",
    {},
)

gnn_e2e["ae"]["mode"] = "end_to_end"

gnn_e2e["ae"]["architecture_config"] = (
    "${AE_CFG}"
)

gnn_e2e["ae"]["freeze_decoder"] = False

# 기존 base config에 checkpoint가 있어도
# E2E 실험에서는 제거
gnn_e2e["ae"].pop(
    "checkpoint",
    None,
)

gnn_e2e["training"]["out_dir"] = (
    "${GNN_E2E_OUT}"
)


with open(
    gnn_e2e_config_path,
    "w",
    encoding="utf-8",
) as f:

    yaml.safe_dump(
        gnn_e2e,
        f,
        sort_keys=False,
        allow_unicode=True,
    )


# ============================================================
# 생성 결과 출력
# ============================================================

print()
print("Generated configs")
print("------------------------------")

print(f"AE         : {ae_config_path}")
print(f"GNN frozen : {gnn_frozen_config_path}")
print(f"GNN E2E    : {gnn_e2e_config_path}")

print()
print("Common settings")
print("------------------------------")

print(
    "Dataset      : "
    "${TRAIN_DIR}, ${VAL_DIR}, ${TEST_DIR}"
)

print(
    "AE input     : "
    "xyz(3) + extra(1), input_channels=1"
)

print(
    "GNN input    : "
    "input_point_dim=4"
)

print(
    "World frame  : False"
)

print(
    f"Latent dim   : "
    f"{ae_cfg['model']['latent_dim']}"
)

print(
    "Decoder      : folding"
)

print()

PY


# ============================================================
# 전체 실험 시작
# ============================================================

send_slack "🚀 Folding pretrained vs end-to-end comparison started

Dataset:
train=${TRAIN_DIR}
val=${VAL_DIR}
test=${TEST_DIR}

Output:
${BASE_OUT}"


# ============================================================
# 1. AutoEncoder pretraining
# ============================================================

run_training \
    "1/3 Folding AutoEncoder" \
    "${AE_OUT}" \
    python -u "${AE_TRAIN_SCRIPT}" \
        --config "${AE_CFG}"


# ============================================================
# AE checkpoint 생성 확인
# ============================================================

if [ ! -f "${AE_CHECKPOINT}" ]; then

    send_slack "❌ AE checkpoint not found
Expected:
${AE_CHECKPOINT}"

    echo "AE checkpoint not found:"
    echo "${AE_CHECKPOINT}"

    exit 1

fi


send_slack "💾 AE checkpoint ready
${AE_CHECKPOINT}"


# ============================================================
# 2. GNN + pretrained frozen decoder
# ============================================================

run_training \
    "2/3 GNN + Frozen Pretrained Folding Decoder" \
    "${GNN_FROZEN_OUT}" \
    python -u "${GNN_TRAIN_SCRIPT}" \
        --config "${GNN_FROZEN_CFG}"


# ============================================================
# 3. GNN + random decoder end-to-end
# ============================================================

run_training \
    "3/3 GNN + Folding Decoder End-to-End" \
    "${GNN_E2E_OUT}" \
    python -u "${GNN_TRAIN_SCRIPT}" \
        --config "${GNN_E2E_CFG}"


# ============================================================
# 최종 결과
# ============================================================

FROZEN_BEST=$(get_best_info "${GNN_FROZEN_OUT}")
E2E_BEST=$(get_best_info "${GNN_E2E_OUT}")


send_slack "🎉 Folding comparison finished

[Frozen pretrained]
${FROZEN_BEST}

[End-to-end]
${E2E_BEST}

Output:
${BASE_OUT}"


echo
echo "============================================================"
echo "All experiments finished"
echo "============================================================"
echo
echo "AE:"
echo "  ${AE_OUT}"
echo
echo "Frozen pretrained:"
echo "  ${GNN_FROZEN_OUT}"
echo "  ${FROZEN_BEST}"
echo
echo "End-to-end:"
echo "  ${GNN_E2E_OUT}"
echo "  ${E2E_BEST}"
echo
echo "============================================================"