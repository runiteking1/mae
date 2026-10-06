#!/bin/bash
#SBATCH --partition=glinda
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=hopper2

# Generic finetune template. Submitted by submit_finetune_ablations.sh, which
# sets --job-name and --output/--error (into the run's output_dir) per-job via
# the sbatch command line and exports the env vars consumed below.
#
# Required env vars:
#   PRETRAIN_NAME  e.g. base_adamw_lr2.4e-3
#   MODEL          vit_base_patch16 | vit_large_patch16
#   FINETUNE_OPT   adamw | muon
#   BLR            5e-4 (base) | 1e-3 (large)
#   DROP_PATH      0.1 (base)  | 0.2  (large)
#   LAYER_DECAY    0.65 (base) | 0.75 (large)
#
# Optional env vars (with defaults):
#   CKPT_EPOCH     200
#   EPOCHS         50

set -euo pipefail

: "${PRETRAIN_NAME:?PRETRAIN_NAME must be set}"
: "${MODEL:?MODEL must be set}"
: "${FINETUNE_OPT:?FINETUNE_OPT must be set}"
: "${BLR:?BLR must be set}"
: "${DROP_PATH:?DROP_PATH must be set}"
: "${LAYER_DECAY:?LAYER_DECAY must be set}"
CKPT_EPOCH="${CKPT_EPOCH:-200}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ACCUM_ITER="${ACCUM_ITER:-4}"
RESUME_CKPT="${RESUME_CKPT:-}"

MAE_DIR=/home/sjiang/Documents/mae
cd ${MAE_DIR}

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
source ${MAE_DIR}/.venv/bin/activate

# Prevent util/misc.py:init_distributed_mode from entering the SLURM_PROCID
# branch (sbatch sets SLURM_PROCID=0 even without srun). Matches the pretrain
# scripts.
unset SLURM_PROCID

PRETRAIN_CKPT=${MAE_DIR}/output_dir/ablation/${PRETRAIN_NAME}/checkpoint-${CKPT_EPOCH}.pth
OUTPUT_DIR=${MAE_DIR}/output_dir/finetune_ablation/${PRETRAIN_NAME}_finetuneopt_${FINETUNE_OPT}_ckpt${CKPT_EPOCH}

mkdir -p ${OUTPUT_DIR}

echo "PRETRAIN_NAME = ${PRETRAIN_NAME}"
echo "MODEL         = ${MODEL}"
echo "FINETUNE_OPT  = ${FINETUNE_OPT}"
echo "BLR           = ${BLR}"
echo "DROP_PATH     = ${DROP_PATH}"
echo "LAYER_DECAY   = ${LAYER_DECAY}"
echo "CKPT_EPOCH    = ${CKPT_EPOCH}"
echo "EPOCHS        = ${EPOCHS}"
echo "PRETRAIN_CKPT = ${PRETRAIN_CKPT}"
echo "OUTPUT_DIR    = ${OUTPUT_DIR}"

uv run python main_finetune.py \
        --batch_size ${BATCH_SIZE} \
        --accum_iter ${ACCUM_ITER} \
        --epochs ${EPOCHS} \
        --model ${MODEL} \
        --finetune ${PRETRAIN_CKPT} \
        --optimizer ${FINETUNE_OPT} \
        --blr ${BLR} \
        --drop_path ${DROP_PATH} \
        --layer_decay ${LAYER_DECAY} \
        --weight_decay 0.05 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --reprob 0.25 \
        --data_source huggingface \
        --data_path ${HOME}/.cache/huggingface/datasets/imagenet/imagenet/data \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR} \
        ${RESUME_CKPT:+--resume "${RESUME_CKPT}"}
