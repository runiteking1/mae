#!/bin/bash
#SBATCH --job-name=mae-large-muonpolar-2.4e-3
#SBATCH --partition=hopper
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/sjiang/Documents/mae/slurm_logs/large_muon_polar_2.4e-3_%j.out
#SBATCH --error=/home/sjiang/Documents/mae/slurm_logs/large_muon_polar_2.4e-3_%j.err
#SBATCH --exclude=hopper2

MAE_DIR=/home/sjiang/Documents/mae
cd ${MAE_DIR}

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
source ${MAE_DIR}/.venv/bin/activate

# Prevent util/misc.py:init_distributed_mode from entering the SLURM_PROCID
# branch (sbatch sets SLURM_PROCID=0 even without srun).
unset SLURM_PROCID

# Polar Express ablation: ViT-Large | Muon + Polar Express | effective lr=2.4e-3
# (matched to muon_large_2.4e-3.sh; higher lr was a no-op, so only 2.4e-3 here).
# blr=1.5e-4, eff_batch=256*16=4096 → actual lr = 1.5e-4 * (4096/256) = 2.4e-3
# Target ~440 epochs; --epochs sets the cosine-decay endpoint. Resumes from
# checkpoint-latest.pth, so chain several jobs (see submit_muon_polar.sh) to
# reach the target within the 4-day walltime.
OUTPUT_DIR=./output_dir/ablation/large_muon_polar_lr2.4e-3
uv run python main_pretrain.py \
        --batch_size 256 \
        --accum_iter 16 \
        --epochs 440 \
        --model mae_vit_large_patch16 \
        --norm_pix_loss \
        --optimizer muon_polar \
        --blr 1.5e-4 \
        --data_path ${HOME}/.cache/huggingface/datasets/imagenet/imagenet/data \
        --data_source huggingface \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR} \
        --resume ${OUTPUT_DIR}/checkpoint-latest.pth
