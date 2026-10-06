#!/bin/bash
# LR sweep for MuonPolar + no_mix_no_rand on PlantNet-300K.
# Submits one job per (backbone × lr) combination via sbatch slurm_plantnet_finetune.sh.
#
# Usage:
#   bash run_scripts/submit_plantnet_lr_search.sh
# Dry run:
#   DRY_RUN=1 bash run_scripts/submit_plantnet_lr_search.sh

set -euo pipefail

cd "$(dirname "$0")"

DRY_RUN="${DRY_RUN:-0}"

MAE_DIR=/home/sjiang/Documents/mae
TEMPLATE=${MAE_DIR}/run_scripts/slurm_plantnet_finetune.sh

OPT=muon_polar
RECIPE=no_mix_no_rand

# 5-point log-spaced sweep around the default 1e-3
LRS=(1e-4 3e-4 1e-3 3e-3 1e-2)

ABLATIONS=(
    "base_adamw_lr2.4e-3|vit_base_patch16|500|0.65|0.1|512|1"
    "base_muon_lr2.4e-3|vit_base_patch16|500|0.65|0.1|512|1"
    "large_adamw_lr2.4e-3|vit_large_patch16|400|0.75|0.2|128|4"
    "large_muon_lr2.4e-3|vit_large_patch16|400|0.75|0.2|128|4"
)

printf '%-75s  %-12s  %s\n' "RUN" "STATUS" "JOB_ID/REASON"

for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r PRETRAIN_NAME MODEL CKPT_EPOCH LAYER_DECAY DROP_PATH BATCH_SIZE GRAD_ACCUM <<< "${entry}"

    CKPT=${MAE_DIR}/output_dir/ablation/${PRETRAIN_NAME}/checkpoint-${CKPT_EPOCH}.pth

    for LR in "${LRS[@]}"; do
        [[ "${LR}" == "1e-3" ]] && LR_SUFFIX="" || LR_SUFFIX="_lr${LR}"
        RUN_NAME=${PRETRAIN_NAME}_ep${CKPT_EPOCH}_${RECIPE}_${OPT}${LR_SUFFIX}
        OUT=${MAE_DIR}/output_dir/plantnet/${PRETRAIN_NAME}/ep${CKPT_EPOCH}_${RECIPE}_${OPT}${LR_SUFFIX}

        if [[ ! -f "${CKPT}" ]]; then
            printf '%-75s  %-12s  %s\n' "${RUN_NAME}" "SKIPPED" "missing ${CKPT}"
            continue
        fi

        if [[ -f "${OUT}/log.txt" ]]; then
            printf '%-75s  %-12s  %s\n' "${RUN_NAME}" "DONE" "log.txt exists"
            continue
        fi

        mkdir -p "${OUT}"
        OUT_LOG=${OUT}/${RUN_NAME}_%j.out
        ERR_LOG=${OUT}/${RUN_NAME}_%j.err

        EXPORTS="ALL,PRETRAIN_NAME=${PRETRAIN_NAME},MODEL=${MODEL},CKPT_EPOCH=${CKPT_EPOCH},OPT=${OPT},RECIPE=${RECIPE},LAYER_DECAY=${LAYER_DECAY},DROP_PATH=${DROP_PATH},BATCH_SIZE=${BATCH_SIZE},GRAD_ACCUM=${GRAD_ACCUM},LR=${LR}"

        SBATCH_CMD=(
            sbatch
            --parsable
            --job-name="pn-lr-${LR}-${PRETRAIN_NAME: -8}-${OPT}"
            --output="${OUT_LOG}"
            --error="${ERR_LOG}"
            --export="${EXPORTS}"
            "${TEMPLATE}"
        )

        if [[ "${DRY_RUN}" == "1" ]]; then
            printf '%-75s  %-12s  %s\n' "${RUN_NAME}" "DRY-RUN" "${SBATCH_CMD[*]}"
        else
            JOB_ID=$("${SBATCH_CMD[@]}")
            printf '%-75s  %-12s  %s\n' "${RUN_NAME}" "SUBMITTED" "${JOB_ID}"
        fi
    done
done
