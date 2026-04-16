# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch re-implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., 2021) from Meta. The repo supports three training phases: MAE pre-training, supervised fine-tuning, and linear probing on ImageNet.

## Dependencies

No requirements file exists. Install manually:
- PyTorch + CUDA
- torchvision
- `timm==0.3.2` (exact version required; needs a [one-line patch](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) for PyTorch ≥ 1.8.1)
- `submitit` (multi-node SLURM jobs only)
- `tensorboard`

## Common Commands

**Single-GPU pre-training:**
```bash
python main_pretrain.py --batch_size 64 --model mae_vit_large_patch16 --norm_pix_loss --data_path ${IMAGENET_DIR}
```

**Multi-GPU fine-tuning (8 GPUs):**
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    main_finetune.py --batch_size 32 --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} --data_path ${IMAGENET_DIR}
```

**Evaluation only:**
```bash
python main_finetune.py --eval --resume ${FINETUNE_CHKPT} \
    --model vit_base_patch16 --data_path ${IMAGENET_DIR}
```

**Linear probing (SLURM):**
```bash
python submitit_linprobe.py --nodes 4 --batch_size 512 \
    --model vit_base_patch16 --finetune ${PRETRAIN_CHKPT}
```

**Multi-node SLURM submission:**
```bash
python submitit_pretrain.py --nodes 8 --batch_size 64 --model mae_vit_large_patch16 --epochs 800
python submitit_finetune.py --nodes 4 --batch_size 32 --model vit_base_patch16 --epochs 100
```

See `PRETRAIN.md` and `FINETUNE.md` for full hyperparameter tables and expected runtimes.

## Architecture

### Three-phase training

1. **Pre-training** (`main_pretrain.py` → `engine_pretrain.py`): Trains `MaskedAutoencoderViT` on raw images via masked reconstruction. No labels used.
2. **Fine-tuning** (`main_finetune.py` → `engine_finetune.py`): Loads pre-trained encoder, discards decoder, attaches a classification head, trains end-to-end on ImageNet.
3. **Linear probing** (`main_linprobe.py`): Freezes encoder, trains only the classification head.

### Core models

**`models_mae.py` — `MaskedAutoencoderViT`**
- `random_masking()`: Randomly drops 75% of patches (per-sample, via noise-based shuffling)
- `forward_encoder()`: Runs the unmasked 25% of patches through a standard ViT encoder
- `forward_decoder()`: Reinserts learnable mask tokens, runs through a lighter decoder (8 blocks vs. 12–32 in encoder), reconstructs all patches
- `forward_loss()`: MSE on masked patch pixels only; with `--norm_pix_loss`, normalizes each patch by its mean/std before computing loss

Pre-defined configs: `mae_vit_base_patch16`, `mae_vit_large_patch16`, `mae_vit_huge_patch14`.

**`models_vit.py` — `VisionTransformer`**
- Thin wrapper around timm's ViT adding `global_pool` support (CLS token vs. average pooling)
- Used only during fine-tuning and linear probing — not part of MAE pre-training

### Key design details

- **Position embeddings**: Fixed 2D sine-cosine (`util/pos_embed.py`). At fine-tune time, embeddings are interpolated if input resolution differs from pre-training.
- **Learning rate scaling**: `lr = base_lr × effective_batch_size / 256`. Warmup + cosine decay (`util/lr_sched.py`).
- **Layer-wise LR decay** (fine-tuning only): Deeper layers get higher LR. Controlled by `--layer_decay` (0.65–0.75 depending on model). Logic in `util/lr_decay.py`.
- **Optimizer**: AdamW (β=0.9, 0.95) for pre-training; AdamW with layer decay for fine-tuning; LARS for linear probing (`util/lars.py`).
- **Mixed precision**: All training uses `torch.cuda.amp` with gradient scaling via `NativeScalerWithGradNormCount` in `util/misc.py`.
- **Checkpointing**: Saved every 20 epochs. `util/misc.py:load_model()` handles resuming.

### Data pipeline

`util/datasets.py` builds ImageNet train/val loaders. Pre-training uses only `RandomResizedCrop` + `RandomHorizontalFlip`. Fine-tuning adds AutoAugment, RandErase, Mixup, and CutMix (via timm). `util/crop.py` provides a TF-compatible crop to match the original paper's exact augmentation.

### Distributed training

- Single-node multi-GPU: `torch.distributed.launch`
- Multi-node: `submitit_*.py` submits SLURM jobs; each job sets `MASTER_ADDR`/`MASTER_PORT` via a shared init file. `util/misc.py:init_distributed_mode()` handles setup from environment.
