#!/bin/bash
#SBATCH --partition=hopper,blackwell,glinda
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=hopper2
#
# PlantNet-300K full finetune template. Submitted by submit_plantnet_finetune.sh,
# which sets --job-name and --output/--error per-job via the sbatch command line
# and exports the env vars consumed below.
#
# Required env vars:
#   PRETRAIN_NAME   e.g. base_adamw_lr2.4e-3
#   MODEL           vit_base_patch16 | vit_large_patch16
#   OPT             adamw | muon | muon_polar
#   RECIPE          full | no_rand | no_mix | no_mix_no_rand
#   CKPT_EPOCH      e.g. 500
#
# Optional env vars (with defaults):
#   LAYER_DECAY     0.75
#   DROP_PATH       0.1
#   BATCH_SIZE      512
#   GRAD_ACCUM      1

set -euo pipefail

: "${PRETRAIN_NAME:?PRETRAIN_NAME must be set}"
: "${MODEL:?MODEL must be set}"
: "${OPT:?OPT must be set}"
: "${RECIPE:?RECIPE must be set}"
: "${CKPT_EPOCH:?CKPT_EPOCH must be set}"
LAYER_DECAY="${LAYER_DECAY:-0.75}"
DROP_PATH="${DROP_PATH:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-3}"

MAE_DIR=/home/sjiang/Documents/mae
cd ${MAE_DIR}

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
source ${MAE_DIR}/.venv/bin/activate

# Prevent util/misc.py:init_distributed_mode from entering the SLURM_PROCID
# branch (sbatch sets SLURM_PROCID=0 even without srun).
unset SLURM_PROCID

# Append _lr<value> only for non-default LR so existing output dirs are unchanged.
[[ "${LR}" == "1e-3" ]] && LR_SUFFIX="" || LR_SUFFIX="_lr${LR}"

CKPT=${MAE_DIR}/output_dir/ablation/${PRETRAIN_NAME}/checkpoint-${CKPT_EPOCH}.pth
OUT=${MAE_DIR}/output_dir/plantnet/${PRETRAIN_NAME}/ep${CKPT_EPOCH}_${RECIPE}_${OPT}${LR_SUFFIX}

mkdir -p "$OUT"

case "${RECIPE}" in
    no_rand)
        recipe_flags="--randaugment_ops 0 --random_erasing_prob 0"
        ;;
    no_mix)
        recipe_flags="--mixup_alpha 0 --cutmix_alpha 0 --mixup_prob 0 --label_smoothing 0"
        ;;
    no_mix_no_rand)
        recipe_flags="--randaugment_ops 0 --random_erasing_prob 0 --mixup_alpha 0 --cutmix_alpha 0 --mixup_prob 0 --label_smoothing 0"
        ;;
    *)
        recipe_flags=""
        ;;
esac

echo "PRETRAIN_NAME = ${PRETRAIN_NAME}"
echo "MODEL         = ${MODEL}"
echo "CKPT_EPOCH    = ${CKPT_EPOCH}"
echo "OPT           = ${OPT}"
echo "RECIPE        = ${RECIPE}  (${recipe_flags})"
echo "LAYER_DECAY   = ${LAYER_DECAY}"
echo "DROP_PATH     = ${DROP_PATH}"
echo "BATCH_SIZE    = ${BATCH_SIZE}"
echo "GRAD_ACCUM    = ${GRAD_ACCUM}"
echo "LR            = ${LR}"
echo "CKPT          = ${CKPT}"
echo "OUTPUT_DIR    = ${OUT}"

uv run python eval_finetune_plantnet.py \
    --model "${MODEL}" \
    --pretrained_weights "${CKPT}" \
    --checkpoint_key model \
    --optimizer "${OPT}" \
    --layer_decay "${LAYER_DECAY}" \
    --drop_path "${DROP_PATH}" \
    --lr "${LR}" \
    --weight_decay 0.05 \
    --warmup_steps 500 \
    --max_steps 10000 \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --num_workers 8 \
    --output_dir "${OUT}" \
    ${recipe_flags}
