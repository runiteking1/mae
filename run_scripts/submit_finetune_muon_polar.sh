#!/bin/bash
# Finetune the Muon+Polar Express pretrain ablations (base_muon_polar_lr2.4e-3 and
# large_muon_polar_lr2.4e-3) -- they aren't part of the 6-way matrix in
# submit_finetune_ablations.sh, so they get their own submitter here. Runs all
# three finetune optimizers (adamw, muon, muon_polar) via the shared
# finetune_template.sh.
#
# Checkpoint epochs and per-model hyperparameters match submit_finetune_ablations.sh:
# base finetunes from epoch 500, large from epoch 400.
#
# Usage:
#   ./submit_finetune_muon_polar.sh [BASE_CKPT_EPOCH] [LARGE_CKPT_EPOCH] [EPOCHS]
# Defaults:
#   BASE_CKPT_EPOCH=500, LARGE_CKPT_EPOCH=400, EPOCHS=50
# Dry run (prints sbatch commands without submitting):
#   DRY_RUN=1 ./submit_finetune_muon_polar.sh
#
# Skips any (pretrain × optimizer) where the pretrain checkpoint is missing,
# so the script is safe to re-run as more pretrain epochs land.

set -euo pipefail

cd "$(dirname "$0")"

BASE_CKPT_EPOCH="${1:-500}"
LARGE_CKPT_EPOCH="${2:-400}"
EPOCHS="${3:-50}"
DRY_RUN="${DRY_RUN:-0}"

MAE_DIR=/home/sjiang/Documents/mae
TEMPLATE=${MAE_DIR}/run_scripts/finetune_template.sh

# Pretrain ablation matrix: name | model | blr | drop_path | layer_decay
# Hyperparameters per FINETUNE.md (single set of values per model size).
ABLATIONS=(
    "base_muon_polar_lr2.4e-3|vit_base_patch16|5e-4|0.1|0.65"
    "large_muon_polar_lr2.4e-3|vit_large_patch16|1e-3|0.2|0.75"
)

OPTIMIZERS=(adamw muon muon_polar)

printf 'BASE_CKPT_EPOCH=%s  LARGE_CKPT_EPOCH=%s  EPOCHS=%s  DRY_RUN=%s\n' "${BASE_CKPT_EPOCH}" "${LARGE_CKPT_EPOCH}" "${EPOCHS}" "${DRY_RUN}"
printf '%-60s  %-12s  %s\n' "RUN" "STATUS" "JOB_ID/REASON"

for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r PRETRAIN_NAME MODEL BLR DROP_PATH LAYER_DECAY <<< "${entry}"

    # Match FINETUNE.md effective batch size of 1024 via accum_iter on 1 GPU.
    if [[ "${MODEL}" == vit_large_* ]]; then
        CKPT_EPOCH=${LARGE_CKPT_EPOCH}
        BATCH_SIZE=128; ACCUM_ITER=8
    else
        CKPT_EPOCH=${BASE_CKPT_EPOCH}
        BATCH_SIZE=256; ACCUM_ITER=4
    fi

    PRETRAIN_CKPT=${MAE_DIR}/output_dir/ablation/${PRETRAIN_NAME}/checkpoint-${CKPT_EPOCH}.pth

    for OPT in "${OPTIMIZERS[@]}"; do
        RUN_NAME=${PRETRAIN_NAME}_finetuneopt_${OPT}_ckpt${CKPT_EPOCH}
        JOB_NAME=ft-${PRETRAIN_NAME}-${OPT}
        OUTPUT_DIR=${MAE_DIR}/output_dir/finetune_ablation/${RUN_NAME}

        if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
            printf '%-60s  %-12s  %s\n' "${RUN_NAME}" "SKIPPED" "missing ${PRETRAIN_CKPT}"
            continue
        fi

        mkdir -p "${OUTPUT_DIR}"
        OUT_LOG=${OUTPUT_DIR}/${RUN_NAME}_%j.out
        ERR_LOG=${OUTPUT_DIR}/${RUN_NAME}_%j.err

        EXPORTS="ALL,PRETRAIN_NAME=${PRETRAIN_NAME},MODEL=${MODEL},FINETUNE_OPT=${OPT},BLR=${BLR},DROP_PATH=${DROP_PATH},LAYER_DECAY=${LAYER_DECAY},CKPT_EPOCH=${CKPT_EPOCH},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE},ACCUM_ITER=${ACCUM_ITER}"

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
