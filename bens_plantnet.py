#!/usr/bin/env python3
"""vit_mud_bench_plantnet.py

ViT-B/16 adaptation of the user's vision Muon / MUD benchmark for
Pl@ntNet-300K-style classification experiments.

Highlights
----------
- Models: ViT-B/16 (torchvision) with optional ImageNet init, plus a ~20M Xception option for same-dataset CNN comparisons.
- Datasets:
  * local ImageFolder layout (recommended for Pl@ntNet-300K)
  * optional Hugging Face dataset loading via `datasets.load_dataset`
- Optimizers:
  * adamw   : AdamW on all params
  * sgd     : SGD on all params
  * rmsprop : RMSprop on all params
  * muon    : Muon on selected matrix-like params + AdamW on others
  * mud     : MUD on selected matrix-like params + AdamW on others
  * halley  : Halley polar iteration on selected matrix-like params + AdamW on others
- Matrix optimizer extras:
  * choose AdamW- or Nesterov-based matrix updates for Muon / MUD
  * optional cosine schedule that blends orthogonalized matrix updates with AdamW
  * optional damped MUD off-diagonal weighting via eta
- Modern ViT training recipe knobs:
  * RandAugment
  * Mixup / CutMix
  * label smoothing
  * Random Erasing
  * optional ImageNet initialization
- Keeps diagnostics / CSV logging / cosine warmup-decay from the prior script.
- Adds macro top-1 / top-5 metrics for both training windows and validation, useful on the long-tailed PlantNet problem because species are weighted equally rather than by sample count.
- Removes AdaptivePreNSController plumbing per user request.

- New script adds strong modern augmentation. This is supposed to help particularly
with ViTs, but adds significant expense. To turn off, can use command line flags
    
    --randaugment_ops 0 --random_erasing_prob 0 --mixup_alpha 0 --cutmix_alpha 0 --mixup_prob 0 --label_smoothing 0

"""

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

_TRITON_TRIL_AVAILABLE = False
_TRITON_TRIL_FN = None
try:
    from optimized_qqt_tril import tril_gram_triton as _triton_tril_gram
    _TRITON_TRIL_AVAILABLE = True
    _TRITON_TRIL_FN = _triton_tril_gram
except Exception:
    _TRITON_TRIL_AVAILABLE = False
    _TRITON_TRIL_FN = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_and_amp_dtype(prefer_bf16: bool = True) -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return dev, torch.bfloat16
        return dev, torch.float16
    return torch.device("cpu"), torch.float32


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def _resolve_imagefolder_split(root: str, candidates: List[str]) -> str:
    for cand in candidates:
        path = os.path.join(root, cand)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Could not find any of {candidates} under {root}")


class InfiniteLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.it = iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


class CSVLogger:
    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        self._w.writeheader()
        self._f.flush()

    def write(self, row: Dict):
        out = {k: row.get(k, "") for k in self.fieldnames}
        self._w.writerow(out)
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LocalImageFolder(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        classes = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        if not classes:
            raise RuntimeError(f"No class directories found under {root}")
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: List[Tuple[str, int]] = []
        for c in classes:
            cdir = os.path.join(root, c)
            for dp, _, fns in os.walk(cdir):
                for fn in sorted(fns):
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in self.IMG_EXTS:
                        self.samples.append((os.path.join(dp, fn), self.class_to_idx[c]))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


class HFDatasetWrapper(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            raise TypeError("Expected HF dataset image column to decode to PIL.Image")
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = int(ex["label"])
        return img, y


def build_transforms(args):
    try:
        import torchvision.transforms as T
        from torchvision.transforms import InterpolationMode
    except Exception as e:
        raise RuntimeError(
            "ViT training script requires torchvision transforms. "
            "Please install a torchvision build compatible with your PyTorch version."
        ) from e

    train_ops: List[nn.Module] = [
        T.RandomResizedCrop(
            args.image_size,
            scale=(args.train_crop_min_scale, 1.0),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.RandomHorizontalFlip(p=args.hflip_prob),
    ]
    if args.randaugment_ops > 0:
        train_ops.append(
            T.RandAugment(
                num_ops=args.randaugment_ops,
                magnitude=args.randaugment_magnitude,
                interpolation=InterpolationMode.BICUBIC,
            )
        )
    train_ops.extend([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if args.random_erasing_prob > 0:
        train_ops.append(T.RandomErasing(p=args.random_erasing_prob, value="random"))
    train_tfm = T.Compose(train_ops)

    resize_size = int(round(args.image_size / max(args.eval_crop_ratio, 1e-6)))
    val_tfm = T.Compose([
        T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfm, val_tfm


def build_dataloaders(args, device: torch.device):
    train_tfm, val_tfm = build_transforms(args)

    if args.dataset_source == "imagefolder":
        root = args.data_dir
        train_dir = _resolve_imagefolder_split(root, [args.train_split, "train"])
        val_dir = _resolve_imagefolder_split(root, [args.val_split, "val", "validation"])

        train_ds = LocalImageFolder(train_dir, transform=train_tfm)
        val_ds = LocalImageFolder(val_dir, transform=val_tfm)
        class_names = train_ds.classes
        num_classes = len(class_names)

    elif args.dataset_source == "hf":
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError(
                "HF dataset loading requires `datasets`. Install it with `pip install datasets[vision]`."
            ) from e

        train_raw = load_dataset(args.hf_dataset, split=args.train_split, cache_dir=args.hf_cache_dir)
        val_raw = load_dataset(args.hf_dataset, split=args.val_split, cache_dir=args.hf_cache_dir)
        train_ds = HFDatasetWrapper(train_raw, transform=train_tfm)
        val_ds = HFDatasetWrapper(val_raw, transform=val_tfm)

        feat = train_raw.features.get("label", None)
        if feat is not None and hasattr(feat, "names") and feat.names is not None:
            class_names = list(feat.names)
            num_classes = len(class_names)
        else:
            num_classes = max(int(train_raw[i]["label"]) for i in range(min(len(train_raw), 2048))) + 1
            class_names = [str(i) for i in range(num_classes)]
    else:
        raise ValueError(f"Unknown dataset_source: {args.dataset_source}")

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )
    return train_loader, val_loader, num_classes, class_names


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    def __init__(self, in_filters: int, out_filters: int, reps: int, stride: int = 1,
                 start_with_relu: bool = True, grow_first: bool = True):
        super().__init__()
        self.skip = None
        if out_filters != in_filters or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_filters),
            )

        rep: List[nn.Module] = []
        filters = in_filters
        if grow_first:
            if start_with_relu:
                rep.append(nn.ReLU(inplace=True))
            rep.extend([
                SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_filters),
            ])
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.extend([
                SeparableConv2d(filters, filters, 3, 1, 1, bias=False),
                nn.BatchNorm2d(filters),
            ])

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.extend([
                SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_filters),
            ])

        if not start_with_relu and rep and isinstance(rep[0], nn.ReLU):
            rep = rep[1:]
        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.rep(inp)
        skip = inp if self.skip is None else self.skip(inp)
        return x + skip


class Xception(nn.Module):
    def __init__(self, num_classes: int = 100, in_chans: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = XceptionBlock(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = XceptionBlock(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = XceptionBlock(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = XceptionBlock(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(2048)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(2048, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.global_pool(x).flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x


def build_model(args, num_classes: int) -> nn.Module:
    if args.model == "xception":
        return Xception(num_classes=num_classes, dropout=args.dropout)

    try:
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    except Exception as e:
        raise RuntimeError(
            "ViT model construction requires torchvision.models. "
            "Please install a torchvision build compatible with your PyTorch version."
        ) from e

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if args.pretrained_imagenet else None
    if weights is None:
        model = vit_b_16(
            weights=None,
            image_size=args.image_size,
            num_classes=num_classes,
            dropout=args.dropout,
            attention_dropout=args.attn_dropout,
        )
    else:
        model = vit_b_16(
            weights=weights,
            image_size=args.image_size,
            dropout=args.dropout,
            attention_dropout=args.attn_dropout,
        )
        head_in = model.heads.head.in_features
        model.heads.head = nn.Linear(head_in, num_classes)
        nn.init.zeros_(model.heads.head.bias)
        nn.init.trunc_normal_(model.heads.head.weight, std=0.02)
    return model


# -----------------------------------------------------------------------------
# Mixup / CutMix / soft targets
# -----------------------------------------------------------------------------


def one_hot(target: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    off = smoothing / max(1, num_classes)
    on = 1.0 - smoothing + off
    y = torch.full((target.shape[0], num_classes), off, device=target.device, dtype=torch.float32)
    y.scatter_(1, target.unsqueeze(1), on)
    return y


class MixupCutmix:
    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float,
        cutmix_alpha: float,
        prob: float,
        switch_prob: float,
        label_smoothing: float,
    ):
        self.num_classes = int(num_classes)
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.prob = float(prob)
        self.switch_prob = float(switch_prob)
        self.label_smoothing = float(label_smoothing)

    def _sample_lambda(self, alpha: float) -> float:
        if alpha <= 0.0:
            return 1.0
        lam = np.random.beta(alpha, alpha)
        return float(lam)

    def _rand_bbox(self, x: torch.Tensor, lam: float) -> Tuple[int, int, int, int]:
        _, _, h, w = x.shape
        cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
        cut_w = max(1, int(w * cut_ratio))
        cut_h = max(1, int(h * cut_ratio))
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, x1 + cut_w)
        y2 = min(h, y1 + cut_h)
        return x1, y1, x2, y2

    def __call__(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str, float]:
        if (self.mixup_alpha <= 0.0 and self.cutmix_alpha <= 0.0) or random.random() > self.prob:
            return x, one_hot(target, self.num_classes, self.label_smoothing), "none", 1.0

        use_cutmix = False
        if self.cutmix_alpha > 0.0 and self.mixup_alpha > 0.0:
            use_cutmix = (random.random() < self.switch_prob)
        elif self.cutmix_alpha > 0.0:
            use_cutmix = True

        perm = torch.randperm(x.size(0), device=x.device)
        target1 = one_hot(target, self.num_classes, self.label_smoothing)
        target2 = target1[perm]

        if use_cutmix:
            lam = self._sample_lambda(self.cutmix_alpha)
            x1, y1, x2, y2 = self._rand_bbox(x, lam)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            area = max(1, (x2 - x1) * (y2 - y1))
            lam = 1.0 - (area / float(x.shape[-1] * x.shape[-2]))
            target_mix = lam * target1 + (1.0 - lam) * target2
            return x, target_mix, "cutmix", float(lam)

        lam = self._sample_lambda(self.mixup_alpha)
        x = lam * x + (1.0 - lam) * x[perm]
        target_mix = lam * target1 + (1.0 - lam) * target2
        return x, target_mix, "mixup", float(lam)


def soft_target_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


# -----------------------------------------------------------------------------
# Matrix views / diagnostics helpers
# -----------------------------------------------------------------------------


def tensor_to_matrix(X: torch.Tensor) -> torch.Tensor:
    if X.ndim == 2:
        return X
    if X.ndim == 4:
        return X.reshape(X.shape[0], -1)
    raise ValueError(f"Unsupported tensor rank for matrix optimizer: ndim={X.ndim}")


def matrix_to_tensor(M: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if ref.ndim in (2, 4):
        return M.reshape_as(ref)
    raise ValueError(f"Unsupported ref rank for matrix optimizer: ndim={ref.ndim}")


def _as_diag_matrix_view(M: torch.Tensor) -> torch.Tensor:
    M = tensor_to_matrix(M)
    return M.t() if M.shape[0] > M.shape[1] else M


@torch.no_grad()
def _spectral_norm_power(A: torch.Tensor, iters: int = 8, eps: float = 1e-12) -> float:
    n = A.shape[0]
    v = torch.randn(n, device=A.device, dtype=A.dtype)
    v = v / v.norm().clamp_min(eps)
    for _ in range(iters):
        v = A @ v
        v = v / v.norm().clamp_min(eps)
    Av = A @ v
    return float(Av.norm().clamp_min(eps))


@torch.no_grad()
def diag_metrics_matrix(G: torch.Tensor, M: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12, spec_iters: int = 8) -> Dict[str, float]:
    Gv = _as_diag_matrix_view(G).float()
    Mv = _as_diag_matrix_view(M).float()
    Qv = _as_diag_matrix_view(Q).float()

    g = Gv.reshape(-1)
    m = Mv.reshape(-1)
    q = Qv.reshape(-1)

    def cos(a, b):
        return float((a @ b) / (a.norm() * b.norm()).clamp_min(eps))

    cos_mq = cos(m, q)
    cos_gq = cos(g, q)
    ratio_q_over_m = float(q.norm() / (m.norm().clamp_min(eps)))
    A = Qv @ Qv.T
    k = A.shape[0]
    I = torch.eye(k, device=A.device, dtype=A.dtype)
    E = A - I
    ortho_fro = float(E.norm(p="fro") / I.norm(p="fro").clamp_min(eps))
    E_off = E.clone()
    E_off.fill_diagonal_(0.0)
    off_max = float(E_off.abs().max())
    off_inf = float(E_off.abs().sum(dim=1).max())
    spec_err = _spectral_norm_power(E, iters=spec_iters, eps=eps)
    return {
        "cos_mq": cos_mq,
        "cos_gq": cos_gq,
        "ratio_q_over_m": ratio_q_over_m,
        "ortho_fro": ortho_fro,
        "off_max": off_max,
        "off_inf": off_inf,
        "spec_err": spec_err,
    }


class DiagAggregatorByKey:
    def __init__(self):
        self.reset()

    def reset(self):
        self._vals: Dict[str, Dict[str, List[float]]] = {}

    def add(self, key: str, metrics: Dict[str, float]):
        if key not in self._vals:
            self._vals[key] = {k: [] for k in metrics.keys()}
        for k, v in metrics.items():
            self._vals[key][k].append(float(v))

    def _summ_one(self, d: Dict[str, List[float]]) -> Dict[str, float]:
        out = {}
        for k, xs in d.items():
            t = torch.tensor(xs)
            out[f"{k}_mean"] = float(t.mean())
            out[f"{k}_std"] = float(t.std(unbiased=False)) if t.numel() > 1 else 0.0
        return out

    def summary(self, topk: int = 5, sort_metric: str = "off_inf_mean") -> Dict:
        if not self._vals:
            return {"global": None, "top": None}
        per = {name: self._summ_one(vals) for name, vals in self._vals.items()}
        global_metrics = {}
        keys = list(next(iter(per.values())).keys())
        for k in keys:
            global_metrics[k] = float(torch.tensor([per[n][k] for n in per]).mean())
        ranked = sorted(per.items(), key=lambda kv: kv[1].get(sort_metric, 0.0), reverse=True)
        return {"global": global_metrics, "top": ranked[:topk]}


# -----------------------------------------------------------------------------
# Muon / MUD core
# -----------------------------------------------------------------------------

_MUON_FIXED_COEFFS = (3.4445, -4.7750, 2.0315)
_MUON_PLUS_COEFFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]
_POLAR_EXPRESS_COEFFS_RAW = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


def _polar_express_coeffs_with_safety() -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for (a, b, c) in _POLAR_EXPRESS_COEFFS_RAW[:-1]:
        out.append((a / 1.01, b / (1.01 ** 3), c / (1.01 ** 5)))
    out.append(_POLAR_EXPRESS_COEFFS_RAW[-1])
    return out


_POLAR_EXPRESS_COEFFS = _polar_express_coeffs_with_safety()


def _get_ns_coeff_schedule(coeffs_mode: str, steps: int, fixed_coeffs: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
    if steps <= 0:
        return []
    if coeffs_mode == "fixed":
        return [tuple(fixed_coeffs)] * steps
    if coeffs_mode == "muon_plus":
        base = _MUON_PLUS_COEFFS
        if steps <= len(base):
            return base[-steps:]
        return base + [base[-1]] * (steps - len(base))
    if coeffs_mode == "polar_express":
        base = _POLAR_EXPRESS_COEFFS
        if steps <= len(base):
            return base[:steps]
        return base + [base[-1]] * (steps - len(base))
    raise ValueError("Unknown coeffs_mode")


def muon_ns_polar(X: torch.Tensor, niter: int, *, coeffs_mode: str = "fixed", precond_mode: str = "none",
                  fixed_coeffs: Tuple[float, float, float] = _MUON_FIXED_COEFFS, eps: float = 1e-7) -> torch.Tensor:
    assert X.ndim == 2
    assert niter >= 0
    assert precond_mode in ("none", "corr", "aol", "zca_corr")
    assert coeffs_mode in ("fixed", "muon_plus", "polar_express")

    transposed = (X.shape[0] > X.shape[1])
    if transposed:
        X = X.mT.contiguous()
    X = X.float()

    ### ZCA-Correlation whitening, row normalize first
    if precond_mode == "zca_corr":
        X = X / X.norm(dim=1, keepdim=True).clamp(min=eps)

    if precond_mode != "aol":
        if coeffs_mode == "polar_express":
            X = X / (X.norm(p="fro") + 1e-2)
            X = X / 1.01
        else:
            X = X / (X.norm(p="fro") + eps)
    coeffs_sched = _get_ns_coeff_schedule(coeffs_mode, niter, fixed_coeffs)
    for i, (a, b, c) in enumerate(coeffs_sched):
        A = X @ X.mT
        if precond_mode == "corr":
            s = (A.diagonal().abs() + 1e-8).rsqrt()
            X = s[:, None] * X
            A = (s[:, None] * A) * s[None, :]
        elif precond_mode == "aol" and i == 0:
            s = torch.rsqrt(torch.clamp_min(A.abs().sum(dim=-1), min=eps))
            X = X * s.unsqueeze(-1)
            A = A * s.unsqueeze(-1) * s.unsqueeze(-2)
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)

    if transposed:
        X = X.mT.contiguous()
    return X


def moonshot_lr_scale(n: int, m: int) -> float:
    return 0.2 * math.sqrt(float(max(n, m)))


# -----------------------------------------------------------------------------
# Param split for matrix optimizer
# -----------------------------------------------------------------------------

@dataclass
class ParamMeta:
    name: str
    module_name: str
    module_type: str
    shape: Tuple[int, ...]
    is_conv: bool
    kernel_size: Optional[Tuple[int, int]]
    is_linear: bool
    is_attention: bool
    is_classifier: bool
    is_patch_embed: bool
    is_first_conv: bool


def collect_param_metadata(model: nn.Module) -> Dict[int, ParamMeta]:
    meta: Dict[int, ParamMeta] = {}
    conv_seen = 0
    for module_name, module in model.named_modules():
        for param_name, p in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            is_conv = isinstance(module, nn.Conv2d)
            is_linear = isinstance(module, nn.Linear)
            is_attention = isinstance(module, nn.MultiheadAttention)
            kernel_size = tuple(module.kernel_size) if is_conv else None
            lname = full_name.lower()
            is_classifier = lname.startswith("heads") or ".heads." in lname or any(tok in lname for tok in [".fc", "classifier"])
            is_patch_embed = (module_name == "conv_proj")
            is_first_conv = False
            if is_conv and param_name == "weight":
                is_first_conv = (conv_seen == 0)
                conv_seen += 1
            meta[id(p)] = ParamMeta(
                name=full_name,
                module_name=module_name,
                module_type=type(module).__name__,
                shape=tuple(p.shape),
                is_conv=is_conv,
                kernel_size=kernel_size,
                is_linear=is_linear,
                is_attention=is_attention,
                is_classifier=is_classifier,
                is_patch_embed=is_patch_embed,
                is_first_conv=is_first_conv,
            )
    return meta


def split_params_for_matrix_optimizer(
    model: nn.Module,
    matrix_param_mode: str,
    exclude_classifier: bool,
    include_patch_embed: bool,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], Dict[int, str], Dict[int, ParamMeta]]:
    name_map: Dict[int, str] = {}
    meta = collect_param_metadata(model)
    matrix_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []

    vit_like = any(pm.is_patch_embed or "encoder.layers" in pm.name or "self_attention" in pm.name for pm in meta.values())
    resolved_mode = matrix_param_mode
    if matrix_param_mode == "auto":
        resolved_mode = "linear" if vit_like else "pointwise"

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name_map[id(p)] = name
        pm = meta[id(p)]
        eligible = False

        if pm.is_conv and p.ndim == 4 and p.shape[0] > 1:
            if resolved_mode == "pointwise":
                eligible = (pm.kernel_size == (1, 1))
            elif resolved_mode == "all_conv":
                eligible = True
            elif resolved_mode == "pointwise_linear":
                eligible = (pm.kernel_size == (1, 1))
            elif resolved_mode == "all_hidden":
                eligible = True
            elif resolved_mode in ("linear", "linear_patch"):
                eligible = bool(pm.is_patch_embed and (include_patch_embed or resolved_mode == "linear_patch"))
            elif resolved_mode == "all_matrix":
                eligible = True
            else:
                raise ValueError(f"Unknown matrix_param_mode: {matrix_param_mode}")
        elif p.ndim == 2 and p.shape[0] > 1:
            if resolved_mode in ("linear", "linear_patch", "all_matrix", "pointwise_linear", "all_hidden"):
                eligible = True

        if exclude_classifier and pm.is_classifier:
            eligible = False
        if pm.is_patch_embed and not (include_patch_embed or resolved_mode in ("linear_patch", "all_matrix")):
            eligible = False

        if eligible:
            matrix_params.append(p)
        else:
            other_params.append(p)
    return matrix_params, other_params, name_map, meta


# -----------------------------------------------------------------------------
# In-house optimizers
# -----------------------------------------------------------------------------


class _BaseMatrixOptimizer(torch.optim.Optimizer):
    def __init__(self, matrix_params, other_params, *, lr: float, weight_decay: float,
                 beta: float, adamw_betas: Tuple[float, float], adamw_eps: float,
                 param_name_map: Dict[int, str], diag_interval: int, diag_topk: int,
                 ada_muon: bool, ada_muon_bias_correction: bool = True,
                 ada_muon_target_rms: float = 0.2, matrix_base: str = "adamw",
                 matrix_final_alpha: float = 1.0, max_steps: int = 0):
        self._diag = DiagAggregatorByKey()
        self._param_name_map = param_name_map
        self._global_step = 0
        self._diag_interval = int(diag_interval)
        self._diag_topk = int(diag_topk)
        self._last_diag = None
        self._ada_muon = bool(ada_muon)
        self._ada_muon_bias_correction = bool(ada_muon_bias_correction)
        self._ada_muon_target_rms = float(ada_muon_target_rms)
        assert matrix_base in ("adamw", "nesterov")
        self._matrix_base = str(matrix_base)
        self._matrix_final_alpha = float(matrix_final_alpha)
        assert 0.0 <= self._matrix_final_alpha <= 1.0
        self._max_steps = int(max_steps)
        self._last_matrix_alpha = 1.0
        param_groups = [
            dict(params=list(matrix_params), lr=lr, beta=beta, weight_decay=weight_decay, use_matrix=True,
                 adamw_betas=adamw_betas, adamw_eps=adamw_eps),
            dict(params=list(other_params), lr=lr, weight_decay=weight_decay, use_matrix=False,
                 adamw_betas=adamw_betas, adamw_eps=adamw_eps),
        ]
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    def set_lr(self, lr: float) -> None:
        for g in self.param_groups:
            g["lr"] = lr

    def _matrix_alpha(self) -> float:
        if self._matrix_final_alpha >= 1.0 or self._max_steps <= 0:
            self._last_matrix_alpha = 1.0
            return 1.0
        t = min(self._global_step, self._max_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * (t / max(1, self._max_steps))))
        alpha = self._matrix_final_alpha + (1.0 - self._matrix_final_alpha) * coeff
        self._last_matrix_alpha = float(alpha)
        return float(alpha)

    def _ensure_matrix_state(self, p: torch.Tensor):
        state = self.state[p]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p)
        if "adamw_step" not in state:
            state["adamw_step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        if self._ada_muon and "v" not in state:
            state["v"] = torch.zeros_like(p)
            state["v_step"] = 0
        return state

    def _adamw_matrix_update(self, p: torch.Tensor, g: torch.Tensor, beta1: float, beta2: float, eps: float) -> torch.Tensor:
        state = self._ensure_matrix_state(p)
        state["adamw_step"] += 1
        t = int(state["adamw_step"])
        m = state["exp_avg"]
        v = state["exp_avg_sq"]
        m.mul_(beta1).add_(g, alpha=(1.0 - beta1))
        v.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2))
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        return m_hat / (v_hat.sqrt().add_(eps))

    def _adamw_step_group(self, group):
        lr = group["lr"]
        wd = group["weight_decay"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            p.mul_(1.0 - lr * wd)
            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            m.mul_(beta1).add_(g, alpha=(1 - beta1))
            v.mul_(beta2).addcmul_(g, g, value=(1 - beta2))
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


class MuonOptimizer(_BaseMatrixOptimizer):
    def __init__(self, matrix_params, other_params, *, lr: float, weight_decay: float, beta: float,
                 ns_steps: int, coeffs_mode: str = "fixed", precond: str = "none",
                 fixed_coeffs: Tuple[float, float, float] = _MUON_FIXED_COEFFS,
                 adamw_betas=(0.9, 0.95), adamw_eps=1e-8,
                 param_name_map=None, diag_interval=10**18, diag_topk=5,
                 ada_muon: bool = False, ada_muon_bias_correction: bool = True,
                 ada_muon_target_rms: float = 0.2, matrix_base: str = "adamw",
                 matrix_final_alpha: float = 1.0, max_steps: int = 0):
        super().__init__(matrix_params, other_params, lr=lr, weight_decay=weight_decay, beta=beta,
                         adamw_betas=adamw_betas, adamw_eps=adamw_eps,
                         param_name_map=param_name_map or {}, diag_interval=diag_interval,
                         diag_topk=diag_topk, ada_muon=ada_muon,
                         ada_muon_bias_correction=ada_muon_bias_correction,
                         ada_muon_target_rms=ada_muon_target_rms,
                         matrix_base=matrix_base, matrix_final_alpha=matrix_final_alpha,
                         max_steps=max_steps)
        self._ns_steps = int(ns_steps)
        self._coeffs_mode = str(coeffs_mode)
        self._precond = str(precond)
        self._fixed_coeffs = tuple(fixed_coeffs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        do_diag = (self._diag_interval > 0) and (self._global_step % self._diag_interval == 0)
        if do_diag:
            self._diag.reset()
        for group in self.param_groups:
            if group.get("use_matrix", False):
                self._muon_step_group(group, do_diag=do_diag)
            else:
                self._adamw_step_group(group)
        if do_diag:
            self._last_diag = self._diag.summary(topk=self._diag_topk)
        self._global_step += 1
        return loss

    def _muon_step_group(self, group, do_diag: bool):
        lr = group["lr"]
        beta = group["beta"]
        wd = group["weight_decay"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        matrix_alpha = self._matrix_alpha()
        for idx, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            g = p.grad
            state = self._ensure_matrix_state(p)
            p.mul_(1.0 - lr * wd)

            A = self._adamw_matrix_update(p, g, beta1, beta2, eps)
            mom = state["momentum"]
            mom.mul_(beta).add_(g)
            M = g + beta * mom
            base_tensor = A if self._matrix_base == "adamw" else M

            M2 = tensor_to_matrix(base_tensor).float()
            transposed = M2.shape[0] > M2.shape[1]
            X_work = M2.mT.contiguous() if transposed else M2.contiguous()
            X_in = torch.sign(X_work) if self._ada_muon else X_work
            Q2 = muon_ns_polar(
                X_in, niter=self._ns_steps, coeffs_mode=self._coeffs_mode,
                precond_mode=self._precond, fixed_coeffs=self._fixed_coeffs,
            )
            if transposed:
                Q2 = Q2.mT.contiguous()
            Q = matrix_to_tensor(Q2.to(dtype=p.dtype), p)
            if self._ada_muon:
                v = state["v"]
                state["v_step"] += 1
                t = int(state["v_step"])
                beta2_muon = beta
                v.mul_(beta2_muon).addcmul_(Q, Q, value=(1.0 - beta2_muon))
                if getattr(self, "_ada_muon_bias_correction", True):
                    v_hat = v / (1.0 - (beta2_muon ** t))
                else:
                    v_hat = v
                Q = Q / (v_hat.sqrt().add_(1e-8))
                target_rms = getattr(self, "_ada_muon_target_rms", 0.2)
                rms = Q.pow(2).mean().sqrt().clamp_min(1e-12)
                Q = Q * (target_rms / rms)
                scale = 1.0
            else:
                scale = moonshot_lr_scale(int(X_work.shape[0]), int(X_work.shape[1]))

            adamw_update = A.to(dtype=p.dtype)
            blended_update = matrix_alpha * (scale * Q)
            if matrix_alpha < 1.0:
                blended_update = blended_update + (1.0 - matrix_alpha) * adamw_update

            if do_diag:
                key = self._param_name_map.get(id(p), f"mat[{idx}]_{tuple(p.shape)}")
                self._diag.add(key, diag_metrics_matrix(g, base_tensor, Q))
            p.add_(blended_update, alpha=-lr)


class MUDOptimizer(_BaseMatrixOptimizer):
    def __init__(self, matrix_params, other_params, *, lr: float, weight_decay: float, beta: float,
                 passes: int, mud_mode: str, pre_ns_steps: int,
                 pre_ns_coeffs_mode: str = "fixed", pre_ns_precond: str = "none",
                 pre_ns_fixed_coeffs: Tuple[float, float, float] = _MUON_FIXED_COEFFS,
                 use_triton_tril: bool = False, adamw_betas=(0.9, 0.95), adamw_eps=1e-8,
                 param_name_map=None, diag_interval=10**18, diag_topk=5,
                 ada_muon: bool = False, ada_muon_bias_correction: bool = True,
                 ada_muon_target_rms: float = 0.2, matrix_base: str = "adamw",
                 matrix_final_alpha: float = 1.0, max_steps: int = 0,
                 mud_eta: float = 1.0):
        super().__init__(matrix_params, other_params, lr=lr, weight_decay=weight_decay, beta=beta,
                         adamw_betas=adamw_betas, adamw_eps=adamw_eps,
                         param_name_map=param_name_map or {}, diag_interval=diag_interval,
                         diag_topk=diag_topk, ada_muon=ada_muon,
                         ada_muon_bias_correction=ada_muon_bias_correction,
                         ada_muon_target_rms=ada_muon_target_rms,
                         matrix_base=matrix_base, matrix_final_alpha=matrix_final_alpha,
                         max_steps=max_steps)
        self._passes = int(passes)
        assert mud_mode in ("gs", "chol")
        self._mud_mode = mud_mode
        self._pre_ns_steps = int(pre_ns_steps)
        self._pre_ns_coeffs_mode = str(pre_ns_coeffs_mode)
        self._pre_ns_precond = str(pre_ns_precond)
        self._pre_ns_fixed_coeffs = tuple(pre_ns_fixed_coeffs)
        self._use_triton_tril = bool(use_triton_tril) and _TRITON_TRIL_AVAILABLE
        self._mud_eta = float(mud_eta)
        assert 0.0 <= self._mud_eta <= 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        do_diag = (self._diag_interval > 0) and (self._global_step % self._diag_interval == 0)
        if do_diag:
            self._diag.reset()
        for group in self.param_groups:
            if group.get("use_matrix", False):
                self._mud_step_group(group, do_diag=do_diag)
            else:
                self._adamw_step_group(group)
        if do_diag:
            self._last_diag = self._diag.summary(topk=self._diag_topk)
        self._global_step += 1
        return loss

    def _mud_step_group(self, group, do_diag: bool):
        lr = group["lr"]
        beta = group["beta"]
        wd = group["weight_decay"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        matrix_alpha = self._matrix_alpha()
        for idx, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            g = p.grad
            state = self._ensure_matrix_state(p)
            p.mul_(1.0 - lr * wd)

            A = self._adamw_matrix_update(p, g, beta1, beta2, eps)
            mom = state["momentum"]
            mom.mul_(beta).add_(g)
            M = g + beta * mom
            base_tensor = A if self._matrix_base == "adamw" else M

            M2 = tensor_to_matrix(base_tensor).float()
            transposed = M2.shape[0] > M2.shape[1]
            X_work = M2.mT.contiguous() if transposed else M2.contiguous()
            Q2 = X_work
            if self._pre_ns_steps > 0:
                Q2 = muon_ns_polar(
                    Q2, niter=self._pre_ns_steps, coeffs_mode=self._pre_ns_coeffs_mode,
                    precond_mode=self._pre_ns_precond, fixed_coeffs=self._pre_ns_fixed_coeffs,
                )
            for _ in range(self._passes):
                Q2 = self._mud_core(Q2)
            if transposed:
                Q2 = Q2.mT.contiguous()
            Q = matrix_to_tensor(Q2.to(dtype=p.dtype), p)
            if self._ada_muon:
                v = state["v"]
                state["v_step"] += 1
                t = int(state["v_step"])
                beta2_muon = beta
                v.mul_(beta2_muon).addcmul_(Q, Q, value=(1.0 - beta2_muon))
                if getattr(self, "_ada_muon_bias_correction", True):
                    v_hat = v / (1.0 - (beta2_muon ** t))
                else:
                    v_hat = v
                Q = Q / (v_hat.sqrt().add_(1e-8))
                target_rms = getattr(self, "_ada_muon_target_rms", 0.2)
                rms = Q.pow(2).mean().sqrt().clamp_min(1e-12)
                Q = Q * (target_rms / rms)
                scale = 1.0
            else:
                scale = moonshot_lr_scale(int(X_work.shape[0]), int(X_work.shape[1]))

            adamw_update = A.to(dtype=p.dtype)
            blended_update = matrix_alpha * (scale * Q)
            if matrix_alpha < 1.0:
                blended_update = blended_update + (1.0 - matrix_alpha) * adamw_update

            if do_diag:
                key = self._param_name_map.get(id(p), f"mat[{idx}]_{tuple(p.shape)}")
                self._diag.add(key, diag_metrics_matrix(g, base_tensor, Q))
            p.add_(blended_update, alpha=-lr)

    def _mud_cholesky_solve(self, G: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        k = G.shape[0]
        I = torch.eye(k, device=G.device, dtype=G.dtype)
        diag_mean = float(G.diagonal().mean().clamp_min(eps))
        jitter = 1e-5 * diag_mean
        for _ in range(5):
            L, info = torch.linalg.cholesky_ex(G + jitter * I)
            if int(info) == 0:
                Q_try = torch.linalg.solve_triangular(L, Q, upper=False)
                if torch.isfinite(Q_try).all():
                    return Q_try
            jitter *= 10.0
        return torch.linalg.solve_triangular(torch.tril(G), Q, upper=False)

    def _mud_core(self, Q: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        Q = Q.float()
        Q = Q / Q.norm(dim=1, keepdim=True).clamp(min=eps)
        I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
        G = Q @ Q.mT
        if self._mud_mode == "gs" and self._use_triton_tril and Q.is_cuda:
            T = _TRITON_TRIL_FN(Q)
            T = torch.tril(T, diagonal=-1)
            if self._mud_eta != 1.0:
                T = T * self._mud_eta
            T = T + I
        else:
            G = G * self._mud_eta
            G.diagonal().fill_(1.0)
            if self._mud_mode == "gs":
                T = torch.tril(G, diagonal=-1) + I
            else:
                T = G
        if self._mud_mode == "gs":
            Q = torch.linalg.solve_triangular(T, Q, upper=False)
        else:
            Q = self._mud_cholesky_solve(T, Q, eps=eps)
        Q = Q / Q.norm(dim=1, keepdim=True).clamp(min=eps)
        return Q


# -----------------------------------------------------------------------------
# Metrics / eval
# -----------------------------------------------------------------------------


def halley_polar_cholesky(
    M: torch.Tensor,
    *,
    iters: int = 2,
    scale_mode: str = "trace",
    trace_target: float = 1.0,
    eps: float = 1e-8,
    chol_jitter: float = 1e-5,
    chol_max_tries: int = 5,
) -> torch.Tensor:
    assert M.ndim == 2
    assert iters >= 0
    assert scale_mode in ("none", "trace", "frob")

    transposed = M.shape[0] > M.shape[1]
    X = M.mT.contiguous() if transposed else M.contiguous()
    X = X.float()
    k = X.shape[0]

    if scale_mode != "none":
        frob = X.norm(p="fro").clamp_min(eps)
        if scale_mode == "trace":
            X = X * (math.sqrt(max(trace_target, eps) * k) / frob)
        elif scale_mode == "frob":
            X = X / frob

    I = torch.eye(k, device=X.device, dtype=X.dtype)
    for _ in range(iters):
        A = X @ X.mT
        RHS = 3.0 * X + (A @ X)
        C = 3.0 * A
        C.diagonal().add_(1.0)

        jitter = float(chol_jitter)
        Y = None
        for _try in range(max(1, chol_max_tries)):
            L, info = torch.linalg.cholesky_ex(C + jitter * I)
            if int(info) == 0:
                Z = torch.linalg.solve_triangular(L, RHS, upper=False)
                Y_try = torch.linalg.solve_triangular(L.mT, Z, upper=True)
                if torch.isfinite(Y_try).all():
                    Y = Y_try
                    break
            jitter *= 10.0
        if Y is None:
            L = torch.linalg.cholesky(C + jitter * I)
            Z = torch.linalg.solve_triangular(L, RHS, upper=False)
            Y = torch.linalg.solve_triangular(L.mT, Z, upper=True)
        X = Y

    if transposed:
        X = X.mT.contiguous()
    return X.to(dtype=M.dtype)


class HalleyOptimizer(_BaseMatrixOptimizer):
    def __init__(self, matrix_params, other_params, *, lr: float, weight_decay: float, beta: float,
                 halley_iters: int, halley_scale_mode: str, halley_trace_target: float,
                 halley_chol_jitter: float, halley_chol_max_tries: int,
                 adamw_betas=(0.9, 0.95), adamw_eps=1e-8,
                 param_name_map=None, diag_interval=10**18, diag_topk=5,
                 matrix_base: str = "adamw", matrix_final_alpha: float = 1.0,
                 max_steps: int = 0):
        super().__init__(matrix_params, other_params, lr=lr, weight_decay=weight_decay, beta=beta,
                         adamw_betas=adamw_betas, adamw_eps=adamw_eps,
                         param_name_map=param_name_map or {}, diag_interval=diag_interval,
                         diag_topk=diag_topk, ada_muon=False,
                         matrix_base=matrix_base, matrix_final_alpha=matrix_final_alpha,
                         max_steps=max_steps)
        self._halley_iters = int(halley_iters)
        self._halley_scale_mode = str(halley_scale_mode)
        self._halley_trace_target = float(halley_trace_target)
        self._halley_chol_jitter = float(halley_chol_jitter)
        self._halley_chol_max_tries = int(halley_chol_max_tries)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        do_diag = (self._diag_interval > 0) and (self._global_step % self._diag_interval == 0)
        if do_diag:
            self._diag.reset()
        for group in self.param_groups:
            if group.get("use_matrix", False):
                self._halley_step_group(group, do_diag=do_diag)
            else:
                self._adamw_step_group(group)
        if do_diag:
            self._last_diag = self._diag.summary(topk=self._diag_topk)
        self._global_step += 1
        return loss

    def _halley_step_group(self, group, do_diag: bool):
        lr = group["lr"]
        beta = group["beta"]
        wd = group["weight_decay"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        matrix_alpha = self._matrix_alpha()
        for idx, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            g = p.grad
            state = self._ensure_matrix_state(p)
            p.mul_(1.0 - lr * wd)
            A = self._adamw_matrix_update(p, g, beta1, beta2, eps)
            mom = state["momentum"]
            mom.mul_(beta).add_(g)
            M = g + beta * mom
            base_tensor = A if self._matrix_base == "adamw" else M
            M2 = tensor_to_matrix(base_tensor).float()
            Q2 = halley_polar_cholesky(
                M2,
                iters=self._halley_iters,
                scale_mode=self._halley_scale_mode,
                trace_target=self._halley_trace_target,
                chol_jitter=self._halley_chol_jitter,
                chol_max_tries=self._halley_chol_max_tries,
            )
            Q = matrix_to_tensor(Q2.to(dtype=p.dtype), p)
            scale = moonshot_lr_scale(int(_as_diag_matrix_view(M2).shape[0]), int(_as_diag_matrix_view(M2).shape[1]))
            adamw_update = A.to(dtype=p.dtype)
            blended_update = matrix_alpha * (scale * Q)
            if matrix_alpha < 1.0:
                blended_update = blended_update + (1.0 - matrix_alpha) * adamw_update
            if do_diag:
                key = self._param_name_map.get(id(p), f"mat[{idx}]_{tuple(p.shape)}")
                self._diag.add(key, diag_metrics_matrix(g, base_tensor, Q))
            p.add_(blended_update, alpha=-lr)


def _macro_from_counts(correct_k: torch.Tensor, counts: torch.Tensor) -> float:
    mask = counts > 0
    if not torch.any(mask):
        return 0.0
    per_class = correct_k[mask].float() / counts[mask].float().clamp_min(1.0)
    return float(per_class.mean().item() * 100.0)


def update_classwise_topk(logits: torch.Tensor, target: torch.Tensor, correct1: torch.Tensor, correct5: torch.Tensor, counts: torch.Tensor) -> None:
    maxk = min(5, logits.size(1))
    _, pred = logits.topk(maxk, 1, True, True)
    hit1 = pred[:, 0].eq(target)
    hit5 = pred.eq(target.unsqueeze(1)).any(dim=1)
    ones = torch.ones_like(target, dtype=counts.dtype)
    counts.index_add_(0, target, ones)
    correct1.index_add_(0, target, hit1.to(correct1.dtype))
    correct5.index_add_(0, target, hit5.to(correct5.dtype))


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(float(correct_k.mul_(100.0 / target.size(0))))
        return res


@torch.no_grad()
def estimate_metrics(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype, max_batches: int = 0, num_classes: Optional[int] = None):
    model.eval()
    losses = []
    top1_sum = 0.0
    top5_sum = 0.0
    n = 0
    macro_counts = None
    macro_correct1 = None
    macro_correct5 = None
    for bidx, (x, y) in enumerate(loader):
        if max_batches > 0 and bidx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        bs = y.size(0)
        losses.append(loss.item() * bs)
        t1, t5 = accuracy(logits, y, topk=(1, min(5, logits.size(1))))
        top1_sum += t1 * bs / 100.0
        top5_sum += t5 * bs / 100.0
        n += bs
        if num_classes is None:
            num_classes = logits.size(1)
        if macro_counts is None:
            macro_counts = torch.zeros(num_classes, dtype=torch.long, device=logits.device)
            macro_correct1 = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
            macro_correct5 = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
        update_classwise_topk(logits, y, macro_correct1, macro_correct5, macro_counts)
    model.train()
    macro_top1 = _macro_from_counts(macro_correct1, macro_counts) if macro_counts is not None else 0.0
    macro_top5 = _macro_from_counts(macro_correct5, macro_counts) if macro_counts is not None else 0.0
    return float(sum(losses) / max(1, n)), 100.0 * top1_sum / max(1, n), 100.0 * top5_sum / max(1, n), macro_top1, macro_top5


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_source", type=str, default="imagefolder", choices=["imagefolder", "hf"])
    parser.add_argument("--data_dir", type=str, default="./plantnet300k")
    parser.add_argument("--hf_dataset", type=str, default="")
    parser.add_argument("--hf_cache_dir", type=str, default="")
    parser.add_argument("--train_split", type=str, default="images_train")
    parser.add_argument("--val_split", type=str, default="images_val")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_batches", type=int, default=0, help="0 means full validation loader")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup_steps", type=int, default=500)

    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--pretrained_imagenet", action="store_true")

    parser.add_argument("--train_crop_min_scale", type=float, default=0.08)
    parser.add_argument("--eval_crop_ratio", type=float, default=0.875)
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--randaugment_ops", type=int, default=2)
    parser.add_argument("--randaugment_magnitude", type=int, default=9)
    parser.add_argument("--random_erasing_prob", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.8)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)

    parser.add_argument("--model", type=str, default="vit_b_16", choices=["vit_b_16", "xception"])

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "rmsprop", "muon", "mud", "halley"])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--adamw_beta1", type=float, default=0.9)
    parser.add_argument("--adamw_beta2", type=float, default=0.999)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--sgd_nesterov", action="store_true")
    parser.add_argument("--rmsprop_alpha", type=float, default=0.99)
    parser.add_argument("--rmsprop_eps", type=float, default=1e-8)

    parser.add_argument("--matrix_param_mode", type=str, default="auto", choices=["auto", "pointwise", "linear", "linear_patch", "all_matrix", "all_conv", "pointwise_linear", "all_hidden"])
    parser.add_argument("--include_classifier_in_matrix", action="store_true")
    parser.add_argument("--include_patch_embed_in_matrix", action="store_true")

    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_ns_coeffs", type=float, nargs=3, default=[3.4445, -4.7750, 2.0315])
    parser.add_argument("--precond", type=str, default="none", choices=["none", "corr", "aol", "zca_corr"])
    parser.add_argument("--coeffs", type=str, default="polar_express", choices=["fixed", "muon_plus", "polar_express"])
    parser.add_argument("--matrix_base", type=str, default="nesterov", choices=["adamw", "nesterov"],
                        help="Base matrix update for Muon / MUD. AdamW is the default; Nesterov restores the older behavior.")
    parser.add_argument("--matrix_final_alpha", type=float, default=1.0,
                        help="Final cosine-decayed fraction on the orthogonalized matrix update. Starts at 1.0 at step 0; AdamW share is 1 - alpha.")

    parser.add_argument("--mud_passes", type=int, default=1)
    parser.add_argument("--mud_mode", type=str, default="gs", choices=["gs", "chol"])
    parser.add_argument("--pre_ns_steps", type=int, default=0)
    parser.add_argument("--use_triton_tril", action="store_true")
    parser.add_argument("--mud_eta", type=float, default=1.0,
                        help="Off-diagonal damping for MUD Gram / triangular factor. 1.0 is standard MUD; 0.0 reduces to row-normalized base updates.")

    parser.add_argument("--halley_iters", type=int, default=2)
    parser.add_argument("--halley_scale_mode", type=str, default="trace", choices=["none", "trace", "frob"])
    parser.add_argument("--halley_trace_target", type=float, default=1.0)
    parser.add_argument("--halley_chol_jitter", type=float, default=1e-5)
    parser.add_argument("--halley_chol_max_tries", type=int, default=5)

    parser.add_argument("--diag_interval", type=int, default=-1)
    parser.add_argument("--diag_topk", type=int, default=5)
    parser.add_argument("--csv_path", type=str, default="")

    parser.add_argument("--ada_muon", action="store_true")
    parser.add_argument("--no_ada_muon_bias_correction", action="store_true")
    parser.add_argument("--ada_muon_target_rms", type=float, default=0.2)

    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--no_prefer_bf16", action="store_true")

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    if args.hf_cache_dir == "":
        args.hf_cache_dir = None
    if args.dataset_source == "hf" and args.hf_dataset == "":
        raise ValueError("--hf_dataset must be provided when --dataset_source hf")
    if not (0.0 <= args.matrix_final_alpha <= 1.0):
        raise ValueError("--matrix_final_alpha must be in [0, 1]")
    if not (0.0 <= args.mud_eta <= 1.0):
        raise ValueError("--mud_eta must be in [0, 1]")

    compile_model = not args.no_compile
    prefer_bf16 = not args.no_prefer_bf16

    set_seed(args.seed)
    device, amp_dtype = device_and_amp_dtype(prefer_bf16=prefer_bf16)

    train_loader, val_loader, num_classes, class_names = build_dataloaders(args, device)
    train_iter = InfiniteLoader(train_loader)

    set_seed(args.seed)
    model = build_model(args, num_classes=num_classes).to(device)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    mixup_fn = MixupCutmix(
        num_classes=num_classes,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        prob=args.mixup_prob,
        switch_prob=args.mixup_switch_prob,
        label_smoothing=args.label_smoothing,
    )
    ce_with_ls = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    total_params = sum(p.numel() for p in model.parameters())
    print(args)
    print(f"Dataset source: {args.dataset_source} | classes: {num_classes} | train batches/epoch ~ {len(train_loader)}")
    print(f"Model: {args.model} | total parameters: {total_params}")
    if args.use_triton_tril:
        print(f"Triton tril kernel requested: available={_TRITON_TRIL_AVAILABLE}")
    if args.optimizer in ("muon", "mud", "halley"):
        print(
            f"Matrix optimizer base: {args.matrix_base} | final orth alpha: {args.matrix_final_alpha:.3f} "
            f"| final AdamW share: {1.0 - args.matrix_final_alpha:.3f}"
        )
        if args.optimizer == "mud":
            print(f"MUD eta: {args.mud_eta:.3f}")
        if args.optimizer == "halley":
            print(f"Halley iters: {args.halley_iters} | scale_mode: {args.halley_scale_mode} | trace_target: {args.halley_trace_target:.3f}")

    matrix_params, other_params, name_map, param_meta = split_params_for_matrix_optimizer(
        model,
        matrix_param_mode=args.matrix_param_mode,
        exclude_classifier=(not args.include_classifier_in_matrix),
        include_patch_embed=args.include_patch_embed_in_matrix,
    )
    print(f"Matrix params: {sum(p.numel() for p in matrix_params):,} across {len(matrix_params)} tensors")
    print(f"Other params : {sum(p.numel() for p in other_params):,} across {len(other_params)} tensors")
    if matrix_params:
        print("Selected matrix tensors (first 16):")
        for p in matrix_params[:16]:
            pm = param_meta[id(p)]
            print(f"  - {pm.name}: {pm.shape} | {pm.module_type} | kernel={pm.kernel_size}")
        if len(matrix_params) > 16:
            print(f"  ... and {len(matrix_params) - 16} more")

    adamw_betas = (args.adamw_beta1, args.adamw_beta2)
    diag_interval = int(args.diag_interval)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=adamw_betas, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.sgd_momentum,
                                    weight_decay=args.weight_decay, nesterov=args.sgd_nesterov)
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.rmsprop_alpha,
                                        eps=args.rmsprop_eps, momentum=args.sgd_momentum,
                                        weight_decay=args.weight_decay)
    elif args.optimizer == "muon":
        optimizer = MuonOptimizer(
            matrix_params, other_params,
            lr=args.lr, weight_decay=args.weight_decay, beta=args.beta,
            ns_steps=args.muon_ns_steps, coeffs_mode=args.coeffs, precond=args.precond,
            fixed_coeffs=tuple(args.muon_ns_coeffs), adamw_betas=adamw_betas,
            param_name_map=name_map, diag_interval=diag_interval, diag_topk=args.diag_topk,
            ada_muon=args.ada_muon,
            ada_muon_bias_correction=(not args.no_ada_muon_bias_correction),
            ada_muon_target_rms=args.ada_muon_target_rms,
            matrix_base=args.matrix_base,
            matrix_final_alpha=args.matrix_final_alpha,
            max_steps=args.max_steps,
        )
    elif args.optimizer == "mud":
        optimizer = MUDOptimizer(
            matrix_params, other_params,
            lr=args.lr, weight_decay=args.weight_decay, beta=args.beta,
            passes=args.mud_passes, mud_mode=args.mud_mode, pre_ns_steps=args.pre_ns_steps,
            pre_ns_coeffs_mode=args.coeffs, pre_ns_precond=args.precond,
            pre_ns_fixed_coeffs=tuple(args.muon_ns_coeffs), use_triton_tril=args.use_triton_tril,
            adamw_betas=adamw_betas, param_name_map=name_map,
            diag_interval=diag_interval, diag_topk=args.diag_topk,
            ada_muon=args.ada_muon,
            ada_muon_bias_correction=(not args.no_ada_muon_bias_correction),
            ada_muon_target_rms=args.ada_muon_target_rms,
            matrix_base=args.matrix_base,
            matrix_final_alpha=args.matrix_final_alpha,
            max_steps=args.max_steps,
            mud_eta=args.mud_eta,
        )
    elif args.optimizer == "halley":
        optimizer = HalleyOptimizer(
            matrix_params, other_params,
            lr=args.lr, weight_decay=args.weight_decay, beta=args.beta,
            halley_iters=args.halley_iters,
            halley_scale_mode=args.halley_scale_mode,
            halley_trace_target=args.halley_trace_target,
            halley_chol_jitter=args.halley_chol_jitter,
            halley_chol_max_tries=args.halley_chol_max_tries,
            adamw_betas=adamw_betas, param_name_map=name_map,
            diag_interval=diag_interval, diag_topk=args.diag_topk,
            matrix_base=args.matrix_base,
            matrix_final_alpha=args.matrix_final_alpha,
            max_steps=args.max_steps,
        )
    else:
        raise ValueError("Unknown optimizer")

    use_scaler = (device.type == "cuda" and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    csv_logger: Optional[CSVLogger] = None
    if args.csv_path:
        csv_fields = [
            "step", "optimizer", "model", "event", "lr",
            "train_loss", "train_top1", "train_top5", "train_macro_top1", "train_macro_top5",
            "images_total", "train_time_total_s",
            "images_window", "train_time_window_s", "images_per_s_window",
            "val_loss", "val_top1", "val_top5", "val_macro_top1", "val_macro_top5", "eval_time_s", "eval_time_total_s",
            "wall_time_s", "last_aug", "last_mix_lam",
            "cos_mq", "cos_gq", "ortho_fro", "off_inf", "off_max", "spec_err", "ratio_q_over_m",
        ]
        csv_logger = CSVLogger(args.csv_path, csv_fields)
        print(f"[csv] writing logs to {args.csv_path}")

    model.train()
    wall_t0 = time.time()
    train_time_total_s = 0.0
    eval_time_total_s = 0.0
    images_total = 0
    train_time_window_s = 0.0
    images_window = 0
    effective_batch = int(args.batch_size * args.grad_accum)
    last_aug = "none"
    last_mix_lam = 1.0
    train_macro_counts = torch.zeros(num_classes, dtype=torch.long)
    train_macro_correct1 = torch.zeros(num_classes, dtype=torch.float32)
    train_macro_correct5 = torch.zeros(num_classes, dtype=torch.float32)

    for step in range(args.max_steps + 1):
        lr_t = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.lr * args.min_lr_ratio)
        if hasattr(optimizer, "set_lr"):
            optimizer.set_lr(lr_t)
        else:
            for g in optimizer.param_groups:
                g["lr"] = lr_t

        t_train0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_top1 = 0.0
        running_top5 = 0.0
        for _ in range(args.grad_accum):
            x, y = train_iter.next()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hard = y
            x, y_train, last_aug, last_mix_lam = mixup_fn(x, y)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(x)
                if y_train.dtype.is_floating_point:
                    loss = soft_target_cross_entropy(logits, y_train)
                else:
                    loss = ce_with_ls(logits, y_train)
                loss_scaled = loss / args.grad_accum
            logits_det = logits.detach()
            t1, t5 = accuracy(logits_det, y_hard, topk=(1, min(5, num_classes)))
            update_classwise_topk(
                logits_det.float().cpu(),
                y_hard.detach().cpu(),
                train_macro_correct1,
                train_macro_correct5,
                train_macro_counts,
            )
            running_loss += float(loss.detach().item())
            running_top1 += t1
            running_top5 += t5
            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

        if args.clip_grad is not None and args.clip_grad > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        dt_train = time.time() - t_train0
        train_time_total_s += dt_train
        train_time_window_s += dt_train
        images_total += effective_batch
        images_window += effective_batch
        wall_elapsed_s = time.time() - wall_t0

        train_loss = running_loss / max(1, args.grad_accum)
        train_top1 = running_top1 / max(1, args.grad_accum)
        train_top5 = running_top5 / max(1, args.grad_accum)
        train_macro_top1 = _macro_from_counts(train_macro_correct1, train_macro_counts)
        train_macro_top5 = _macro_from_counts(train_macro_correct5, train_macro_counts)

        did_log = (step % args.log_interval == 0 and step > 0)
        did_eval = (step % args.eval_interval == 0 and step > 0)

        if did_log:
            imgs_per_s = images_window / max(train_time_window_s, 1e-9)
            print(
                f"[{args.optimizer}] step {step:6d} | lr {lr_t:.3e} | "
                f"train_loss {train_loss:.4f} | top1 {train_top1:5.2f} | top5 {train_top5:5.2f} | "
                f"macro1 {train_macro_top1:5.2f} | macro5 {train_macro_top5:5.2f} | imgs/s {imgs_per_s:,.0f} | aug {last_aug}"
            )

        val_loss = None
        val_top1 = None
        val_top5 = None
        val_macro_top1 = None
        val_macro_top5 = None
        eval_time_s = 0.0
        if did_eval:
            t_eval0 = time.time()
            val_loss, val_top1, val_top5, val_macro_top1, val_macro_top5 = estimate_metrics(
                model, val_loader, device, amp_dtype, max_batches=args.eval_batches, num_classes=num_classes
            )
            eval_time_s = time.time() - t_eval0
            eval_time_total_s += eval_time_s
            wall_elapsed_s = time.time() - wall_t0
            print(
                f"[{args.optimizer}] step {step:6d} | lr {lr_t:.3e} | "
                f"val_loss {val_loss:.4f} | top1 {val_top1:5.2f} | top5 {val_top5:5.2f} | "
                f"macro1 {val_macro_top1:5.2f} | macro5 {val_macro_top5:5.2f} | "
                f"train_elapsed {train_time_total_s/60:.3f} min | wall {wall_elapsed_s/60:.3f} min"
            )
            if hasattr(optimizer, "_last_diag") and optimizer._last_diag is not None:
                d = optimizer._last_diag.get("global", None)
                if d is not None:
                    print(
                        f"\t           | cos_mq={d['cos_mq_mean']:.2f}+-{d['cos_mq_std']:.2f} "
                        f"| cos_gq={d['cos_gq_mean']:.2f}+-{d['cos_gq_std']:.2f} "
                        f"| ortF={d['ortho_fro_mean']:.2e} "
                        f"| off_inf={d['off_inf_mean']:.2e} "
                        f"| off_max={d['off_max_mean']:.2e} "
                        f"| spec={d['spec_err_mean']:.2e} "
                        f"| rq/m={d['ratio_q_over_m_mean']:.2e}"
                    )
                    top = optimizer._last_diag.get("top", None)
                    if top:
                        print("\t           | worst matrices by off_inf:")
                        for name, summ in top:
                            print(
                                f"\t             - {name}: "
                                f"off_inf={summ['off_inf_mean']:.2e}, off_max={summ['off_max_mean']:.2e}, "
                                f"ortF={summ['ortho_fro_mean']:.2e}, cos_mq={summ['cos_mq_mean']:.2f}"
                            )
                else:
                    print("\t           | diagnostics: (none)")

        if csv_logger is not None and (did_log or did_eval):
            diag = None
            if hasattr(optimizer, "_last_diag") and optimizer._last_diag is not None:
                diag = optimizer._last_diag.get("global", None)
            row = {
                "step": step,
                "optimizer": args.optimizer,
                "model": args.model,
                "event": ("train+eval" if (did_log and did_eval) else ("train" if did_log else "eval")),
                "lr": lr_t,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "train_top5": train_top5,
                "train_macro_top1": train_macro_top1,
                "train_macro_top5": train_macro_top5,
                "images_total": images_total,
                "train_time_total_s": train_time_total_s,
                "images_window": images_window,
                "train_time_window_s": train_time_window_s,
                "images_per_s_window": (images_window / max(train_time_window_s, 1e-9)) if train_time_window_s > 0 else "",
                "val_loss": val_loss if val_loss is not None else "",
                "val_top1": val_top1 if val_top1 is not None else "",
                "val_top5": val_top5 if val_top5 is not None else "",
                "val_macro_top1": val_macro_top1 if val_macro_top1 is not None else "",
                "val_macro_top5": val_macro_top5 if val_macro_top5 is not None else "",
                "eval_time_s": eval_time_s if did_eval else "",
                "eval_time_total_s": eval_time_total_s,
                "wall_time_s": wall_elapsed_s,
                "last_aug": last_aug,
                "last_mix_lam": last_mix_lam,
            }
            if diag is not None:
                row.update({
                    "cos_mq": diag.get("cos_mq_mean", ""),
                    "cos_gq": diag.get("cos_gq_mean", ""),
                    "ortho_fro": diag.get("ortho_fro_mean", ""),
                    "off_inf": diag.get("off_inf_mean", ""),
                    "off_max": diag.get("off_max_mean", ""),
                    "spec_err": diag.get("spec_err_mean", ""),
                    "ratio_q_over_m": diag.get("ratio_q_over_m_mean", ""),
                })
            csv_logger.write(row)

        if did_log:
            train_time_window_s = 0.0
            images_window = 0
            train_macro_counts.zero_()
            train_macro_correct1.zero_()
            train_macro_correct5.zero_()

    if csv_logger is not None:
        csv_logger.close()


if __name__ == "__main__":
    main()
