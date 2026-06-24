# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch re-implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., 2021), extended into an **optimizer ablation study: AdamW vs. Muon** for MAE pre-training and fine-tuning. The base repo supports three phases (MAE pre-training → supervised fine-tuning → linear probing); this fork adds Muon, MLflow tracking, a HuggingFace parquet data path, SLURM sweep scripts under `run_scripts/`, and a Pl@ntNet-300K transfer-learning workflow.

`PRETRAIN.md` and `FINETUNE.md` hold the original paper's hyperparameter tables and expected runtimes (note: their command examples predate the `uv` / `run_scripts/` workflow below).

## Environment & dependencies

Dependencies are managed by **`uv`** (`pyproject.toml` + `uv.lock`). Always run Python through uv — never bare `python`:

```bash
uv run python main_pretrain.py ...
```

Key versions (differ from the upstream README): `torch>=2.9`, `timm>=0.9.12` (NOT the old `0.3.2`; the version assert is commented out), `datasets`, `mlflow`, `tensorboard`. `submitit` is an optional extra (`[slurm]`) and only used by the legacy `submitit_*.py` launchers.

## Common commands

The actual workflow runs **single-GPU SLURM jobs** via `run_scripts/`, not `torch.distributed.launch` or `submitit_*.py`. Each script `source`s `.venv/bin/activate`, sets `OMP_NUM_THREADS=1`, and `unset SLURM_PROCID` (so `util/misc.py:init_distributed_mode` does NOT take the SLURM-distributed branch — sbatch sets `SLURM_PROCID=0` even without `srun`).

**Pre-training (one ablation cell):**
```bash
uv run python main_pretrain.py --model mae_vit_base_patch16 --norm_pix_loss \
    --optimizer muon --blr 1.5e-4 --batch_size 256 --accum_iter 16 --epochs 800 \
    --data_source huggingface --data_path ${HOME}/.cache/huggingface/datasets/imagenet/imagenet/data \
    --output_dir ./output_dir/ablation/base_muon_lr2.4e-3 --log_dir ./output_dir/ablation/base_muon_lr2.4e-3 \
    --resume ./output_dir/ablation/base_muon_lr2.4e-3/checkpoint-latest.pth
```

**Submit the full sweeps (SLURM):**
```bash
# 6 pretrain ablations (2 model sizes × adamw/muon × lrs), each chained twice via --dependency=afterany
bash run_scripts/submit_all.sh
# 12-run finetune sweep (6 pretrain ablations × {adamw, muon}); skips cells whose pretrain ckpt is missing
DRY_RUN=1 ./run_scripts/submit_finetune_ablations.sh    # preview sbatch commands without submitting
./run_scripts/submit_finetune_ablations.sh 500 400 50   # BASE_CKPT_EPOCH LARGE_CKPT_EPOCH EPOCHS
# 16-run Pl@ntNet finetune sweep (2 backbones × 4 aug recipes × 2 optimizers)
bash run_scripts/submit_plantnet_finetune.sh
```

`run_scripts/*_template.sh` and `slurm_plantnet_finetune.sh` are parameterized by env vars exported via `sbatch --export`; the `submit_*.sh` scripts build the matrix and submit them. Sweeps are safe to re-run — they skip cells with a missing pretrain checkpoint or an already-present `log.txt`.

**Evaluation only:**
```bash
uv run python main_finetune.py --eval --resume ${FINETUNE_CHKPT} --model vit_base_patch16 \
    --data_source huggingface --data_path ${IMAGENET_PARQUET_DIR}
```

**Aggregate & plot results:**
```bash
uv run python aggregate_finetune_results.py   # walks output_dir/finetune_ablation/*/log.txt → per_epoch.csv + summary.csv
uv run python plot_pretrain_losses.py         # reads output_dir/ablation/*/log.txt
uv run python plot_finetune_ablations.py
```

## Architecture

### Three-phase training

1. **Pre-training** (`main_pretrain.py` → `engine_pretrain.py`): trains `MaskedAutoencoderViT` via masked reconstruction. No labels.
2. **Fine-tuning** (`main_finetune.py` → `engine_finetune.py`): loads the pre-trained encoder, discards the decoder, attaches a classification head, trains end-to-end.
3. **Linear probing** (`main_linprobe.py`): freezes the encoder, trains only the head with LARS (`util/lars.py`).

### Optimizer ablation (the point of this fork)

`--optimizer {adamw,muon}` selects the optimizer in both `main_pretrain.py` and `main_finetune.py` (`main_linprobe.py` defaults to LARS). Muon lives in `util/muon.py` (Moonshot/Moonlight reference impl — Newton-Schulz orthogonalization of the update, `@torch.compile`d).

- **Muon is 2D-only.** Parameters are split: matrix weights (`ndim == 2`) go to Muon; everything else (conv `patch_embed`, 3D `pos_embed`/`cls_token`/`mask_token`, norms, biases) is routed to an internal AdamW. The split is done inline in `main_pretrain.py`; for fine-tuning it is combined with layer-wise LR decay via `util/lr_decay.py:param_groups_lrd_muon` (groups carry a `use_muon` flag). Both code paths print the Muon/AdamW tensor and parameter-count breakdown at startup.
- Moonlight's claim is that Muon reuses AdamW's lr/wd, so ablation cells reuse matched effective LRs (e.g. `blr 1.5e-4 × eff_batch 4096 / 256 = 2.4e-3`).

### Core models

**`models_mae.py` — `MaskedAutoencoderViT`**: `random_masking()` drops 75% of patches; `forward_encoder()` runs the visible 25% through a ViT encoder; `forward_decoder()` reinserts mask tokens and reconstructs via a lighter decoder; `forward_loss()` is MSE on masked patches only (`--norm_pix_loss` normalizes each patch first). Configs: `mae_vit_{base,large,huge}_*`.

**`models_vit.py` — `VisionTransformer`**: thin timm wrapper adding `global_pool`; used only in fine-tuning / linear probing.

### Key design details

- **Position embeddings**: fixed 2D sin-cos (`util/pos_embed.py`), interpolated when fine-tune resolution differs from pre-training.
- **LR scaling**: `lr = blr × effective_batch_size / 256` (effective = `batch_size × accum_iter × world_size`). Warmup + cosine decay in `util/lr_sched.py`.
- **Layer-wise LR decay** (fine-tuning): `--layer_decay` (0.65 base / 0.75 large), `util/lr_decay.py`.
- **Mixed precision**: `torch.cuda.amp` + `NativeScalerWithGradNormCount` (`util/misc.py`).
- **Checkpointing**: saved every 20 epochs (+ final). Scripts resume from `checkpoint-latest.pth`. `util/misc.py:load_model()` handles resuming.

### Data pipeline

`util/datasets.py` supports two sources via `--data_source`:
- `imagefolder` (default): torchvision `ImageFolder` on `train/`+`val/`.
- `huggingface`: loads parquet via `datasets.load_dataset(args.data_path)`, wrapped by `HuggingFaceImageNet`; `build_dataset_hf` (fine-tune transforms) and `build_dataset_hf_pretrain` (simple aug). This is the path the SLURM scripts use.

Pre-training uses `RandomResizedCrop` + flip only. Fine-tuning adds AutoAugment, RandErase, Mixup, CutMix (timm). `util/crop.py` provides a TF-compatible crop matching the paper.

### MLflow tracking

`util/mlflow_utils.py` logs on rank 0 only. Tracking URI is **hardcoded** to `file:///home/sjiang/Documents/mae/mlruns`; experiments are `mae-pretrain` / `mae-finetune` / `mae-linprobe`. All `args` are logged as params; run name is `{model}_opt-{optimizer}_lr-{lr}`. Per-epoch `log.txt` (JSONL) is still written to each `output_dir` independently of MLflow.

### Pl@ntNet-300K transfer

- `download_plantnet.py`: caches `mikehemberger/plantnet300K` via HF datasets (needs the Sandia proxy env vars in its docstring).
- `eval_finetune_plantnet.py`: fine-tunes/evaluates MAE backbones on Pl@ntNet (driven by `run_scripts/slurm_plantnet_finetune.sh` / `submit_plantnet_finetune.sh`). The sweep also exercises a `muon_polar` optimizer variant and 4 augmentation recipes (`full`, `no_rand`, `no_mix`, `no_mix_no_rand`).
- `bens_plantnet.py`: a large **standalone** ViT-B/16 Muon/MUD benchmark (its own optimizers, datasets, recipes) — not part of the `main_*.py` pipeline.

### Distributed training (legacy)

`submitit_*.py` (multi-node SLURM) and `torch.distributed.launch` still exist from upstream but are not the current workflow — prefer `run_scripts/`. `util/misc.py:init_distributed_mode()` handles setup from environment.
