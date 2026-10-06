#!/bin/bash
#SBATCH --partition=glinda
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=hopper2

# Few-shot finetune (pretrained MAE init or from scratch) on a tiny ImageFolder
# built by fewshot/make_fewshot_imagenet.py. Submitted by submit_fewshot.sh, or
# directly:
#   sbatch --export=ALL,DATA_ROOT=...,INIT=scratch,RUN_NAME=... fewshot/fewshot_template.sh
#
# Required env vars:
#   DATA_ROOT      few-shot ImageFolder root (has train/ val/ meta.json)
#   INIT           pretrain name under output_dir/ablation/ (e.g. base_adamw_lr2.4e-3),
#                  an absolute checkpoint path, or "scratch"
#   RUN_NAME       output subdir under output_dir/fewshot/
#
# Optional env vars (defaults follow finetune_template.sh, except batch/LR/epochs):
#   CKPT_EPOCH     latest   (pretrain checkpoint-<CKPT_EPOCH>.pth)
#   MODEL          vit_base_patch16
#   FINETUNE_OPT   adamw    (adamw | muon | muon_polar)
#   LR             1e-4     absolute LR (bypasses blr scaling)
#   BATCH_SIZE     16       must be <= #train images x repeat (drop_last=True)
#   EPOCHS         40       epochs over the *repeated* train set
#   WARMUP_EPOCHS  4
#   LAYER_DECAY    0.65     (use 1.0 for scratch)
#   DROP_PATH      0.1
#   WEIGHT_DECAY   0.05
#   MIXUP / CUTMIX 0.8 / 1.0
#   REPROB         0.25
#   SMOOTHING      0.1
#   SEED           0        torch seed (init of head / scratch weights, aug, order)
#   KEEP_CKPT      0        1 = keep the final checkpoint (~1 GB for ViT-B)

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${INIT:?INIT must be set (pretrain name, ckpt path, or scratch)}"
: "${RUN_NAME:?RUN_NAME must be set}"
CKPT_EPOCH="${CKPT_EPOCH:-latest}"
MODEL="${MODEL:-vit_base_patch16}"
FINETUNE_OPT="${FINETUNE_OPT:-adamw}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-40}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-4}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
DROP_PATH="${DROP_PATH:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MIXUP="${MIXUP:-0.8}"
CUTMIX="${CUTMIX:-1.0}"
REPROB="${REPROB:-0.25}"
SMOOTHING="${SMOOTHING:-0.1}"
SEED="${SEED:-0}"
KEEP_CKPT="${KEEP_CKPT:-0}"

MAE_DIR=/home/sjiang/Documents/mae
cd ${MAE_DIR}

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
source ${MAE_DIR}/.venv/bin/activate
unset SLURM_PROCID

NB_CLASSES=$(find "${DATA_ROOT}/train" -mindepth 1 -maxdepth 1 -type d | wc -l)

if [[ "${INIT}" == "scratch" ]]; then
    FINETUNE_ARG=()
elif [[ "${INIT}" == /* ]]; then
    FINETUNE_ARG=(--finetune "${INIT}")
else
    FINETUNE_ARG=(--finetune "${MAE_DIR}/output_dir/ablation/${INIT}/checkpoint-${CKPT_EPOCH}.pth")
fi
if [[ ${#FINETUNE_ARG[@]} -gt 0 && ! -e "${FINETUNE_ARG[1]}" ]]; then
    echo "pretrain checkpoint not found: ${FINETUNE_ARG[1]}" >&2
    exit 1
fi

OUTPUT_DIR=${MAE_DIR}/output_dir/fewshot/${RUN_NAME}
mkdir -p ${OUTPUT_DIR}
cp "${DATA_ROOT}/meta.json" "${OUTPUT_DIR}/data_meta.json"

echo "DATA_ROOT     = ${DATA_ROOT}  (${NB_CLASSES} classes)"
echo "INIT          = ${INIT}  ${FINETUNE_ARG[*]:-}"
echo "MODEL         = ${MODEL}"
echo "FINETUNE_OPT  = ${FINETUNE_OPT}"
echo "LR            = ${LR}"
echo "BATCH_SIZE    = ${BATCH_SIZE}"
echo "EPOCHS        = ${EPOCHS} (warmup ${WARMUP_EPOCHS})"
echo "LAYER_DECAY   = ${LAYER_DECAY}"
echo "DROP_PATH     = ${DROP_PATH}"
echo "MIXUP/CUTMIX  = ${MIXUP}/${CUTMIX}"
echo "SEED          = ${SEED}"
echo "OUTPUT_DIR    = ${OUTPUT_DIR}"

# main_finetune.py writes a full checkpoint every epoch. Delete every
# checkpoint older than the one checkpoint-latest.pth points at; the file being
# written is always newer than latest, so it is never touched.
prune_ckpts() {
    local latest n f
    latest=$(readlink "${OUTPUT_DIR}/checkpoint-latest.pth" 2>/dev/null || true)
    [[ "${latest}" =~ checkpoint-([0-9]+)\.pth ]] || return 0
    n=${BASH_REMATCH[1]}
    for f in "${OUTPUT_DIR}"/checkpoint-[0-9]*.pth; do
        [[ "${f}" =~ checkpoint-([0-9]+)\.pth$ ]] || continue
        (( BASH_REMATCH[1] < n )) && rm -f "${f}"
    done
    return 0
}
( while true; do prune_ckpts; sleep 15; done ) &
PRUNER=$!
trap 'kill ${PRUNER} 2>/dev/null || true' EXIT

uv run python main_finetune.py \
        --batch_size ${BATCH_SIZE} \
        --accum_iter 1 \
        --epochs ${EPOCHS} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --model ${MODEL} \
        "${FINETUNE_ARG[@]}" \
        --optimizer ${FINETUNE_OPT} \
        --lr ${LR} \
        --drop_path ${DROP_PATH} \
        --layer_decay ${LAYER_DECAY} \
        --weight_decay ${WEIGHT_DECAY} \
        --mixup ${MIXUP} \
        --cutmix ${CUTMIX} \
        --reprob ${REPROB} \
        --smoothing ${SMOOTHING} \
        --seed ${SEED} \
        --nb_classes ${NB_CLASSES} \
        --num_workers 8 \
        --data_source imagefolder \
        --data_path ${DATA_ROOT} \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}

kill ${PRUNER} 2>/dev/null || true
prune_ckpts
if [[ "${KEEP_CKPT}" != "1" ]]; then
    rm -f "${OUTPUT_DIR}"/checkpoint-*.pth
fi
