#!/bin/bash
# Submits 16 PlantNet finetune jobs: 2 backbones × 4 augmentation recipes × 2 optimizers.
# Each job is a single-GPU finetune launched via sbatch slurm_plantnet_finetune.sh
# with per-job env vars exported via --export.
#
# Usage:
#   bash run_scripts/submit_plantnet_finetune.sh
# Dry run (prints sbatch commands without submitting):
#   DRY_RUN=1 bash run_scripts/submit_plantnet_finetune.sh
#
# Skips any job where the pretrain checkpoint is missing, so the script is
# safe to re-run as more pretrain epochs land.

set -euo pipefail

cd "$(dirname "$0")"

DRY_RUN="${DRY_RUN:-0}"

MAE_DIR=/home/sjiang/Documents/mae
TEMPLATE=${MAE_DIR}/run_scripts/slurm_plantnet_finetune.sh

# Pretrain backbone matrix: name | model | ckpt_epoch | layer_decay | drop_path | batch_size | grad_accum
# ViT-L OOMs at batch 512 on 40 GB A100; use batch 128 x grad_accum 4 to keep effective batch 512.
ABLATIONS=(
    # "base_adamw_lr2.4e-3|vit_base_patch16|500|0.65|0.1|512|1"
    # "base_muon_lr2.4e-3|vit_base_patch16|500|0.65|0.1|512|1"
    "large_adamw_lr2.4e-3|vit_large_patch16|400|0.75|0.2|128|4"
    "large_muon_lr2.4e-3|vit_large_patch16|400|0.75|0.2|128|4"
)

RECIPES=(full no_rand no_mix no_mix_no_rand)
OPTIMIZERS=(adamw muon_polar)

# Per-optimizer learning rates (LR search showed 3e-4 best for muon_polar)
declare -A OPT_LR=([adamw]="1e-3" [muon_polar]="3e-4")

printf '%-75s  %-12s  %s\n' "RUN" "STATUS" "JOB_ID/REASON"

for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r PRETRAIN_NAME MODEL CKPT_EPOCH LAYER_DECAY DROP_PATH BATCH_SIZE GRAD_ACCUM <<< "${entry}"

    CKPT=${MAE_DIR}/output_dir/ablation/${PRETRAIN_NAME}/checkpoint-${CKPT_EPOCH}.pth

    for RECIPE in "${RECIPES[@]}"; do
        for OPT in "${OPTIMIZERS[@]}"; do
            LR="${OPT_LR[$OPT]}"
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
                --job-name="pn-${PRETRAIN_NAME: -12}-${RECIPE}-${OPT}"
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
done
