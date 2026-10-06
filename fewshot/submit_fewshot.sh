#!/bin/bash
# Few-shot sweep: data draws x inits (pretrained + scratch) x finetune optimizers x LRs.
# Each cell is one single-GPU sbatch of fewshot_template.sh.
#
# Usage:
#   DRY_RUN=1 bash fewshot/submit_fewshot.sh     # print sbatch commands only
#   bash fewshot/submit_fewshot.sh
#
# Edit the arrays below. Cells with an existing log.txt or a missing pretrain
# checkpoint are skipped, so the script is safe to re-run.

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
MAE_DIR=/home/sjiang/Documents/mae
TEMPLATE=${MAE_DIR}/fewshot/fewshot_template.sh
FEWSHOT_DATA=${FEWSHOT_DATA:-${HOME}/data/imagenet_fewshot}

# Few-shot datasets (dirs under FEWSHOT_DATA), built with make_fewshot_imagenet.py.
DRAWS=(
    c8_s8_cs0_ss0
)

# Pretrained inits: pretrain name under output_dir/ablation | model | layer_decay | drop_path
PRETRAINED=(
    "base_adamw_lr2.4e-3|vit_base_patch16|0.65|0.1"
    "base_muon_lr2.4e-3|vit_base_patch16|0.65|0.1"
)
PRETRAINED_LRS=(3e-5 1e-4 3e-4)

# From-scratch baseline (no layer decay: it makes no sense on a random init).
SCRATCH_MODELS=("vit_base_patch16|1.0|0.1")
SCRATCH_LRS=(1e-4 3e-4 1e-3)

OPTIMIZERS=(adamw muon)
SEEDS=(0)
CKPT_EPOCH=${CKPT_EPOCH:-latest}
EPOCHS=${EPOCHS:-40}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}

printf '%-75s  %-10s  %s\n' RUN STATUS JOB_ID/REASON

submit() {  # init model layer_decay drop_path opt lr draw seed
    local init=$1 model=$2 ld=$3 dp=$4 opt=$5 lr=$6 draw=$7 seed=$8
    local run="${draw}/${init}_ft-${opt}_lr${lr}_seed${seed}"
    local out=${MAE_DIR}/output_dir/fewshot/${run}

    if [[ -s ${out}/log.txt ]]; then
        printf '%-75s  %-10s  %s\n' "${run}" SKIP "log.txt exists"; return
    fi
    if [[ ${init} != scratch && ! -e ${MAE_DIR}/output_dir/ablation/${init}/checkpoint-${CKPT_EPOCH}.pth ]]; then
        printf '%-75s  %-10s  %s\n' "${run}" SKIP "no pretrain ckpt"; return
    fi

    local cmd=(sbatch --parsable
        --job-name "fs-${init}-${opt}-${lr}"
        --output "${out}/slurm-%j.out" --error "${out}/slurm-%j.err"
        --export "ALL,DATA_ROOT=${FEWSHOT_DATA}/${draw},INIT=${init},RUN_NAME=${run},CKPT_EPOCH=${CKPT_EPOCH},MODEL=${model},FINETUNE_OPT=${opt},LR=${lr},LAYER_DECAY=${ld},DROP_PATH=${dp},SEED=${seed},EPOCHS=${EPOCHS},WARMUP_EPOCHS=${WARMUP_EPOCHS},BATCH_SIZE=${BATCH_SIZE}"
        "${TEMPLATE}")

    if [[ ${DRY_RUN} == 1 ]]; then
        printf '%-75s  %-10s  %s\n' "${run}" DRY "${cmd[*]}"
    else
        mkdir -p "${out}"
        printf '%-75s  %-10s  %s\n' "${run}" SUBMIT "$("${cmd[@]}")"
    fi
}

for draw in "${DRAWS[@]}"; do
    [[ -d ${FEWSHOT_DATA}/${draw}/train ]] || { echo "missing dataset ${FEWSHOT_DATA}/${draw}" >&2; exit 1; }
    for seed in "${SEEDS[@]}"; do
        for opt in "${OPTIMIZERS[@]}"; do
            for entry in "${PRETRAINED[@]}"; do
                IFS='|' read -r init model ld dp <<< "${entry}"
                for lr in "${PRETRAINED_LRS[@]}"; do
                    submit "${init}" "${model}" "${ld}" "${dp}" "${opt}" "${lr}" "${draw}" "${seed}"
                done
            done
            for entry in "${SCRATCH_MODELS[@]}"; do
                IFS='|' read -r model ld dp <<< "${entry}"
                for lr in "${SCRATCH_LRS[@]}"; do
                    submit scratch "${model}" "${ld}" "${dp}" "${opt}" "${lr}" "${draw}" "${seed}"
                done
            done
        done
    done
done
