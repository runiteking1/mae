#!/bin/bash
# Resume the three large_muon_polar ViT-L finetunes that were cut off by the
# SLURM time limit at epoch 39/38/31 (of 50). Reuses finetune_template.sh, but
# sets RESUME_CKPT=checkpoint-latest.pth so main_finetune.py continues from the
# saved epoch/optimizer state instead of restarting from the pretrained encoder.
#
# Usage:
#   ./resume_finetune_muon_polar_large.sh
#   DRY_RUN=1 ./resume_finetune_muon_polar_large.sh   # print sbatch cmds only

set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN="${DRY_RUN:-0}"
MAE_DIR=/home/sjiang/Documents/mae
TEMPLATE=${MAE_DIR}/run_scripts/finetune_template.sh

# large_muon_polar hyperparameters (match submit_finetune_muon_polar.sh)
PRETRAIN_NAME=large_muon_polar_lr2.4e-3
MODEL=vit_large_patch16
BLR=1e-3
DROP_PATH=0.2
LAYER_DECAY=0.75
CKPT_EPOCH=400
EPOCHS=50
BATCH_SIZE=128
ACCUM_ITER=8

OPTIMIZERS=(adamw muon muon_polar)

printf '%-24s  %-10s  %s\n' "OPTIMIZER" "STATUS" "JOB_ID/REASON"

for OPT in "${OPTIMIZERS[@]}"; do
    RUN_NAME=${PRETRAIN_NAME}_finetuneopt_${OPT}_ckpt${CKPT_EPOCH}
    OUTPUT_DIR=${MAE_DIR}/output_dir/finetune_ablation/${RUN_NAME}
    RESUME_CKPT=${OUTPUT_DIR}/checkpoint-latest.pth

    if [[ ! -e "${RESUME_CKPT}" ]]; then
        printf '%-24s  %-10s  %s\n' "${OPT}" "SKIPPED" "missing ${RESUME_CKPT}"
        continue
    fi

    EXPORTS="ALL,PRETRAIN_NAME=${PRETRAIN_NAME},MODEL=${MODEL},FINETUNE_OPT=${OPT},BLR=${BLR},DROP_PATH=${DROP_PATH},LAYER_DECAY=${LAYER_DECAY},CKPT_EPOCH=${CKPT_EPOCH},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE},ACCUM_ITER=${ACCUM_ITER},RESUME_CKPT=${RESUME_CKPT}"

    SBATCH_CMD=(
        sbatch
        --parsable
        --job-name="ft-${PRETRAIN_NAME}-${OPT}-resume"
        --output="${OUTPUT_DIR}/${RUN_NAME}_resume_%j.out"
        --error="${OUTPUT_DIR}/${RUN_NAME}_resume_%j.err"
        --export="${EXPORTS}"
        "${TEMPLATE}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%-24s  %-10s  %s\n' "${OPT}" "DRY-RUN" "${SBATCH_CMD[*]}"
    else
        JOB_ID=$("${SBATCH_CMD[@]}")
        printf '%-24s  %-10s  %s\n' "${OPT}" "SUBMITTED" "${JOB_ID}"
    fi
done
