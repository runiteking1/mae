#!/bin/bash
# Few-shot sweep: ViT-B finetuned on a few-shot ImageFolder from several pretrain
# checkpoints of one MAE run, plus a from-scratch baseline, over an LR grid.
# Each cell is one single-GPU sbatch of fewshot_template.sh; only log.txt
# (val accuracy every EVAL_FREQ epochs) is kept, no finetuned weights.
#
# Usage:
#   DRY_RUN=1 bash fewshot/submit_fewshot.sh     # print sbatch commands only
#   bash fewshot/submit_fewshot.sh
#
# Edit the arrays below. Cells with an existing log.txt or a missing pretrain
# checkpoint are skipped, so the script is safe to re-run.

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
MAE_DIR=${MAE_DIR:-/home/sjiang/Documents/mae}
TEMPLATE=${MAE_DIR}/fewshot/fewshot_template.sh
FEWSHOT_DATA=${FEWSHOT_DATA:-${HOME}/data/imagenet_fewshot}

# Few-shot datasets (dirs under FEWSHOT_DATA), built with make_fewshot_imagenet.py.
DRAWS=(c8_s8_cs0_ss0)

# Pretrained init: output_dir/ablation/${PRETRAIN}/checkpoint-${epoch}.pth
PRETRAIN=base_adamw_lr2.4e-3
CKPT_EPOCHS=(200 500)
PRETRAINED_LRS=(3e-5 1e-4 3e-4)
PRETRAINED_LAYER_DECAY=0.65

# From scratch: no layer decay (it makes no sense on a random init).
SCRATCH_LRS=(1e-4 3e-4 1e-3)

MODEL=vit_base_patch16
DROP_PATH=0.1
OPTIMIZERS=(adamw)          # finetune optimizer(s): adamw | muon | muon_polar
SEEDS=(0)
EPOCHS=${EPOCHS:-5000}          # part of the run name, so a new budget = new runs
WARMUP_EPOCHS=${WARMUP_EPOCHS:-100}
EVAL_FREQ=${EVAL_FREQ:-50}
BATCH_SIZE=${BATCH_SIZE:-16}

printf '%-70s  %-7s  %s\n' RUN STATUS JOB_ID/REASON

submit() {  # init_tag init ckpt_epoch layer_decay opt lr draw seed
    local tag=$1 init=$2 ckpt=$3 ld=$4 opt=$5 lr=$6 draw=$7 seed=$8
    local run="${draw}/${tag}_ft-${opt}_lr${lr}_e${EPOCHS}_seed${seed}"
    local out=${MAE_DIR}/output_dir/fewshot/${run}

    if [[ -s ${out}/log.txt ]]; then
        printf '%-70s  %-7s  %s\n' "${run}" SKIP "log.txt exists"; return
    fi
    if [[ ${init} != scratch && ! -e ${MAE_DIR}/output_dir/ablation/${init}/checkpoint-${ckpt}.pth ]]; then
        printf '%-70s  %-7s  %s\n' "${run}" SKIP "no checkpoint-${ckpt}.pth"; return
    fi

    local cmd=(sbatch --parsable
        --job-name "fs-${tag}-${opt}-${lr}"
        --output "${out}/slurm-%j.out" --error "${out}/slurm-%j.err"
        --export "ALL,DATA_ROOT=${FEWSHOT_DATA}/${draw},INIT=${init},CKPT_EPOCH=${ckpt},RUN_NAME=${run},MODEL=${MODEL},FINETUNE_OPT=${opt},LR=${lr},LAYER_DECAY=${ld},DROP_PATH=${DROP_PATH},SEED=${seed},EPOCHS=${EPOCHS},WARMUP_EPOCHS=${WARMUP_EPOCHS},EVAL_FREQ=${EVAL_FREQ},SAVE_FREQ=0,BATCH_SIZE=${BATCH_SIZE}"
        "${TEMPLATE}")

    if [[ ${DRY_RUN} == 1 ]]; then
        printf '%-70s  %-7s  %s\n' "${run}" DRY "${cmd[*]}"
    else
        mkdir -p "${out}"
        printf '%-70s  %-7s  %s\n' "${run}" SUBMIT "$("${cmd[@]}")"
    fi
}

for draw in "${DRAWS[@]}"; do
    [[ -d ${FEWSHOT_DATA}/${draw}/train ]] || { echo "missing dataset ${FEWSHOT_DATA}/${draw}" >&2; exit 1; }
    for seed in "${SEEDS[@]}"; do
        for opt in "${OPTIMIZERS[@]}"; do
            for ckpt in "${CKPT_EPOCHS[@]}"; do
                for lr in "${PRETRAINED_LRS[@]}"; do
                    submit "${PRETRAIN}_ep${ckpt}" "${PRETRAIN}" "${ckpt}" "${PRETRAINED_LAYER_DECAY}" "${opt}" "${lr}" "${draw}" "${seed}"
                done
            done
            for lr in "${SCRATCH_LRS[@]}"; do
                submit scratch scratch none 1.0 "${opt}" "${lr}" "${draw}" "${seed}"
            done
        done
    done
done
