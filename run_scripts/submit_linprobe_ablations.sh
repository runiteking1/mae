#!/bin/bash
# Submits linear probing for all pretrain ablations at the canonical checkpoints:
#   ViT-B → ckpt500,  ViT-L → ckpt400
# Optimizer is always LARS (hardcoded in main_linprobe.py), so one run per backbone.
#
# Usage:
#   ./submit_linprobe_ablations.sh [BASE_CKPT_EPOCH] [LARGE_CKPT_EPOCH]
# Defaults:
#   BASE_CKPT_EPOCH=500, LARGE_CKPT_EPOCH=400
# Dry run (prints sbatch commands without submitting):
#   DRY_RUN=1 ./submit_linprobe_ablations.sh
#
# Skips any run whose pretrain checkpoint is missing, or that already has a
# log.txt (so re-running only fills gaps and never clobbers finished runs).

set -euo pipefail

cd "$(dirname "$0")"

BASE_CKPT_EPOCH="${1:-500}"
LARGE_CKPT_EPOCH="${2:-400}"
DRY_RUN="${DRY_RUN:-0}"

MAE_DIR=/home/sjiang/Documents/mae
TEMPLATE=${MAE_DIR}/run_scripts/linprobe_template.sh

# name | model | ckpt_epoch_var | epochs
ABLATIONS=(
    "base_adamw_lr2.4e-3|vit_base_patch16|BASE|90"
    "base_muon_lr2.4e-3|vit_base_patch16|BASE|90"
    "base_muon_lr5.0e-3|vit_base_patch16|BASE|90"
    "large_adamw_lr2.4e-3|vit_large_patch16|LARGE|50"
    "large_muon_lr2.4e-3|vit_large_patch16|LARGE|50"
    "large_muon_lr5.0e-3|vit_large_patch16|LARGE|50"
    "base_muon_polar_lr2.4e-3|vit_base_patch16|BASE|90"
    "large_muon_polar_lr2.4e-3|vit_large_patch16|LARGE|50"
)

printf 'BASE_CKPT_EPOCH=%s  LARGE_CKPT_EPOCH=%s  DRY_RUN=%s\n' \
    "${BASE_CKPT_EPOCH}" "${LARGE_CKPT_EPOCH}" "${DRY_RUN}"
printf '%-50s  %-12s  %s\n' "RUN" "STATUS" "JOB_ID/REASON"

for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r PRETRAIN_NAME MODEL CKPT_VAR EPOCHS <<< "${entry}"

    if [[ "${CKPT_VAR}" == "BASE" ]]; then
        CKPT_EPOCH=${BASE_CKPT_EPOCH}
    else
        CKPT_EPOCH=${LARGE_CKPT_EPOCH}
    fi

    PRETRAIN_CKPT=${MAE_DIR}/output_dir/ablation/${PRETRAIN_NAME}/checkpoint-${CKPT_EPOCH}.pth
    RUN_NAME=${PRETRAIN_NAME}_ckpt${CKPT_EPOCH}
    OUTPUT_DIR=${MAE_DIR}/output_dir/linprobe/${RUN_NAME}

    if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
        printf '%-50s  %-12s  %s\n' "${RUN_NAME}" "SKIPPED" "missing ${PRETRAIN_CKPT}"
        continue
    fi

    if [[ -f "${OUTPUT_DIR}/log.txt" ]]; then
        printf '%-50s  %-12s  %s\n' "${RUN_NAME}" "SKIPPED" "already has log.txt"
        continue
    fi

    mkdir -p "${OUTPUT_DIR}"
    OUT_LOG=${OUTPUT_DIR}/${RUN_NAME}_%j.out
    ERR_LOG=${OUTPUT_DIR}/${RUN_NAME}_%j.err

    EXPORTS="ALL,PRETRAIN_NAME=${PRETRAIN_NAME},MODEL=${MODEL},CKPT_EPOCH=${CKPT_EPOCH},EPOCHS=${EPOCHS}"

    SBATCH_CMD=(
        sbatch
        --parsable
        --job-name="lp-${PRETRAIN_NAME}"
        --output="${OUT_LOG}"
        --error="${ERR_LOG}"
        --export="${EXPORTS}"
        "${TEMPLATE}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%-50s  %-12s  %s\n' "${RUN_NAME}" "DRY-RUN" "${SBATCH_CMD[*]}"
    else
        JOB_ID=$("${SBATCH_CMD[@]}")
        printf '%-50s  %-12s  %s\n' "${RUN_NAME}" "SUBMITTED" "${JOB_ID}"
    fi
done
