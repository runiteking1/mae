#!/bin/bash
# Resumes interrupted finetune jobs from checkpoint-latest.pth.
# Mirrors submit_finetune_ablations.sh: same ablation matrix, same template.
#
# Usage:
#   ./resume_finetune.sh [BASE_CKPT_EPOCH] [LARGE_CKPT_EPOCH] [EPOCHS]
# Defaults:
#   BASE_CKPT_EPOCH=500, LARGE_CKPT_EPOCH=400, EPOCHS=50
# Dry run (prints sbatch commands without submitting):
#   DRY_RUN=1 ./resume_finetune.sh 500 400 50
#
# Skips runs where checkpoint-latest.pth is absent (never started) or
# checkpoint-<EPOCHS-1>.pth already exists (already finished).

set -euo pipefail

cd "$(dirname "$0")"

BASE_CKPT_EPOCH="${1:-500}"
LARGE_CKPT_EPOCH="${2:-400}"
EPOCHS="${3:-50}"
DRY_RUN="${DRY_RUN:-0}"

MAE_DIR=/home/sjiang/Documents/mae
TEMPLATE=${MAE_DIR}/run_scripts/finetune_template.sh

ABLATIONS=(
    "base_adamw_lr2.4e-3|vit_base_patch16|5e-4|0.1|0.65"
    "base_muon_lr2.4e-3|vit_base_patch16|5e-4|0.1|0.65"
    "base_muon_lr5.0e-3|vit_base_patch16|5e-4|0.1|0.65"
    "large_adamw_lr2.4e-3|vit_large_patch16|1e-3|0.2|0.75"
    "large_muon_lr2.4e-3|vit_large_patch16|1e-3|0.2|0.75"
    "large_muon_lr5.0e-3|vit_large_patch16|1e-3|0.2|0.75"
)

OPTIMIZERS=(adamw muon)

LAST_EPOCH=$(( EPOCHS - 1 ))

printf 'BASE_CKPT_EPOCH=%s  LARGE_CKPT_EPOCH=%s  EPOCHS=%s  DRY_RUN=%s\n' \
    "${BASE_CKPT_EPOCH}" "${LARGE_CKPT_EPOCH}" "${EPOCHS}" "${DRY_RUN}"
printf '%-60s  %-12s  %s\n' "RUN" "STATUS" "JOB_ID/REASON"

for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r PRETRAIN_NAME MODEL BLR DROP_PATH LAYER_DECAY <<< "${entry}"

    if [[ "${MODEL}" == vit_large_* ]]; then
        CKPT_EPOCH=${LARGE_CKPT_EPOCH}
        BATCH_SIZE=128; ACCUM_ITER=8
    else
        CKPT_EPOCH=${BASE_CKPT_EPOCH}
        BATCH_SIZE=256; ACCUM_ITER=4
    fi

    for OPT in "${OPTIMIZERS[@]}"; do
        RUN_NAME=${PRETRAIN_NAME}_finetuneopt_${OPT}_ckpt${CKPT_EPOCH}
        JOB_NAME=ft-resume-${PRETRAIN_NAME}-${OPT}
        OUTPUT_DIR=${MAE_DIR}/output_dir/finetune_ablation/${RUN_NAME}
        RESUME_CKPT=${OUTPUT_DIR}/checkpoint-latest.pth

        if [[ ! -f "${RESUME_CKPT}" ]]; then
            printf '%-60s  %-12s  %s\n' "${RUN_NAME}" "SKIPPED" "no checkpoint-latest.pth"
            continue
        fi

        if [[ -f "${OUTPUT_DIR}/checkpoint-${LAST_EPOCH}.pth" ]]; then
            printf '%-60s  %-12s  %s\n' "${RUN_NAME}" "DONE" "checkpoint-${LAST_EPOCH}.pth exists"
            continue
        fi

        OUT_LOG=${OUTPUT_DIR}/${RUN_NAME}_resume_%j.out
        ERR_LOG=${OUTPUT_DIR}/${RUN_NAME}_resume_%j.err

        EXPORTS="ALL,PRETRAIN_NAME=${PRETRAIN_NAME},MODEL=${MODEL},FINETUNE_OPT=${OPT},BLR=${BLR},DROP_PATH=${DROP_PATH},LAYER_DECAY=${LAYER_DECAY},CKPT_EPOCH=${CKPT_EPOCH},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE},ACCUM_ITER=${ACCUM_ITER},RESUME_CKPT=${RESUME_CKPT}"

        SBATCH_CMD=(
            sbatch
            --parsable
            --job-name="${JOB_NAME}"
            --output="${OUT_LOG}"
            --error="${ERR_LOG}"
            --export="${EXPORTS}"
            "${TEMPLATE}"
        )

        if [[ "${DRY_RUN}" == "1" ]]; then
            printf '%-60s  %-12s  %s\n' "${RUN_NAME}" "DRY-RUN" "${SBATCH_CMD[*]}"
        else
            JOB_ID=$("${SBATCH_CMD[@]}")
            printf '%-60s  %-12s  %s\n' "${RUN_NAME}" "SUBMITTED" "${JOB_ID}"
        fi
    done
done
