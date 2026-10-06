#!/bin/bash
#SBATCH --partition=blackwell,hopper,glinda
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=hopper2

# Linear probing template. Submitted by submit_linprobe_ablations.sh, which
# sets --job-name and --output/--error (into the run's output_dir) per-job via
# the sbatch command line and exports the env vars consumed below.
#
# Required env vars:
#   PRETRAIN_NAME  e.g. base_adamw_lr2.4e-3
#   MODEL          vit_base_patch16 | vit_large_patch16
#   CKPT_EPOCH     500 (base) | 400 (large)
#
# Optional env vars (with defaults):
#   EPOCHS         90 (base) | 50 (large)
#   BATCH_SIZE     512
#   ACCUM_ITER     32   (→ effective batch 16384, matching paper)

set -euo pipefail

: "${PRETRAIN_NAME:?PRETRAIN_NAME must be set}"
: "${MODEL:?MODEL must be set}"
: "${CKPT_EPOCH:?CKPT_EPOCH must be set}"
EPOCHS="${EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ACCUM_ITER="${ACCUM_ITER:-32}"

MAE_DIR=/home/sjiang/Documents/mae
cd ${MAE_DIR}

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
source ${MAE_DIR}/.venv/bin/activate

# Prevent util/misc.py:init_distributed_mode from entering the SLURM_PROCID
# branch (sbatch sets SLURM_PROCID=0 even without srun).
unset SLURM_PROCID

PRETRAIN_CKPT=${MAE_DIR}/output_dir/ablation/${PRETRAIN_NAME}/checkpoint-${CKPT_EPOCH}.pth
OUTPUT_DIR=${MAE_DIR}/output_dir/linprobe/${PRETRAIN_NAME}_ckpt${CKPT_EPOCH}

mkdir -p ${OUTPUT_DIR}

echo "PRETRAIN_NAME = ${PRETRAIN_NAME}"
echo "MODEL         = ${MODEL}"
echo "CKPT_EPOCH    = ${CKPT_EPOCH}"
echo "EPOCHS        = ${EPOCHS}"
echo "BATCH_SIZE    = ${BATCH_SIZE}"
echo "ACCUM_ITER    = ${ACCUM_ITER}"
echo "PRETRAIN_CKPT = ${PRETRAIN_CKPT}"
echo "OUTPUT_DIR    = ${OUTPUT_DIR}"

uv run python main_linprobe.py \
        --batch_size ${BATCH_SIZE} \
        --accum_iter ${ACCUM_ITER} \
        --epochs ${EPOCHS} \
        --model ${MODEL} \
        --cls_token \
        --finetune ${PRETRAIN_CKPT} \
        --blr 0.1 \
        --weight_decay 0.0 \
        --data_source huggingface \
        --data_path ${HOME}/.cache/huggingface/datasets/imagenet/imagenet/data \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}
