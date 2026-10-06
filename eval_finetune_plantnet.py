# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms
from torchvision.transforms import InterpolationMode

from timm.utils import accuracy
from timm.models.layers import trunc_normal_

import util.misc as misc
import util.lr_decay as lrd
import models_vit
from util.muon import Muon
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

NUM_CLASSES = 1081  # PlantNet-300K species

_POLAR_EXPRESS_COEFFS = [
    (8.28721201814563   / 1.01, -23.595886519098837 / (1.01**3), 17.300387312530933  / (1.01**5)),
    (4.107059111542203  / 1.01,  -2.9478499167379106 / (1.01**3),  0.5448431082926601 / (1.01**5)),
    (3.9486908534822946 / 1.01,  -2.908902115962949  / (1.01**3),  0.5518191394370137 / (1.01**5)),
    (3.3184196573706015 / 1.01,  -2.488488024314874  / (1.01**3),  0.51004894012372   / (1.01**5)),
    (2.300652019954817  / 1.01,  -1.6689039845747493 / (1.01**3),  0.4188073119525673 / (1.01**5)),
    (1.891301407787398  / 1.01,  -1.2679958271945868 / (1.01**3),  0.37680408948524835/ (1.01**5)),
    (1.8750014808534479 / 1.01,  -1.2500016453999487 / (1.01**3),  0.3750001645474248 / (1.01**5)),
    (1.875,                       -1.25,                            0.375),
]


def zeropower_polar_express(G: torch.Tensor) -> torch.Tensor:
    assert G.ndim == 2
    transposed = G.shape[0] > G.shape[1]
    X = (G.mT if transposed else G).float()
    X = X / (X.norm(p="fro") + 1e-2)
    X = X / 1.01
    for a, b, c in _POLAR_EXPRESS_COEFFS:
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return (X.mT if transposed else X).to(dtype=G.dtype)


# ============ MAE utility replacements ============

def get_sha():
    try:
        return subprocess.check_output(
            ['git', 'describe', '--always'], stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except Exception:
        return "N/A"


def restart_from_checkpoint(ckpt_path, run_variables=None, **kwargs):
    if not os.path.isfile(ckpt_path):
        return
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    for key, obj in kwargs.items():
        if key in ckpt:
            try:
                obj.load_state_dict(ckpt[key])
            except Exception as e:
                print(f"Could not restore {key}: {e}")
    if run_variables is not None:
        for var in run_variables:
            if var in ckpt:
                run_variables[var] = ckpt[var]


# ============ data helpers ============

class PlantNetDataset(torch.utils.data.Dataset):
    """Wraps HuggingFace mikehemberger/plantnet300K dataset."""

    def __init__(self, split="train", transform=None, cache_dir=None):
        from datasets import load_dataset
        self.hf = load_dataset("mikehemberger/plantnet300K", split=split, cache_dir=cache_dir)
        self.transform = transform

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        item = self.hf[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, item["label"]


class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


# ============ LR schedule ============

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def set_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr'] = new_lr * g.get('lr_scale', 1.0)


# ============ augmentation helpers ============

def one_hot(target: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    off = smoothing / max(1, num_classes)
    on = 1.0 - smoothing + off
    y = torch.full((target.shape[0], num_classes), off, device=target.device, dtype=torch.float32)
    y.scatter_(1, target.unsqueeze(1), on)
    return y


class MixupCutmix:
    def __init__(self, num_classes, mixup_alpha, cutmix_alpha, prob, switch_prob, label_smoothing):
        self.num_classes = int(num_classes)
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.prob = float(prob)
        self.switch_prob = float(switch_prob)
        self.label_smoothing = float(label_smoothing)

    def _sample_lambda(self, alpha: float) -> float:
        if alpha <= 0.0:
            return 1.0
        return float(np.random.beta(alpha, alpha))

    def _rand_bbox(self, x: torch.Tensor, lam: float):
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

    def __call__(self, x: torch.Tensor, target: torch.Tensor):
        if (self.mixup_alpha <= 0.0 and self.cutmix_alpha <= 0.0) or random.random() > self.prob:
            return x, one_hot(target, self.num_classes, self.label_smoothing), "none", 1.0

        use_cutmix = False
        if self.cutmix_alpha > 0.0 and self.mixup_alpha > 0.0:
            use_cutmix = (random.random() < self.switch_prob)
        elif self.cutmix_alpha > 0.0:
            use_cutmix = True

        perm    = torch.randperm(x.size(0), device=x.device)
        target1 = one_hot(target, self.num_classes, self.label_smoothing)
        target2 = target1[perm]

        if use_cutmix:
            lam = self._sample_lambda(self.cutmix_alpha)
            x1, y1, x2, y2 = self._rand_bbox(x, lam)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            area = max(1, (x2 - x1) * (y2 - y1))
            lam  = 1.0 - area / float(x.shape[-1] * x.shape[-2])
            return x, lam * target1 + (1.0 - lam) * target2, "cutmix", float(lam)

        lam = self._sample_lambda(self.mixup_alpha)
        return lam * x + (1.0 - lam) * x[perm], lam * target1 + (1.0 - lam) * target2, "mixup", float(lam)


def soft_target_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


# ============ macro metrics ============

def _macro_from_counts(correct_k: torch.Tensor, counts: torch.Tensor) -> float:
    mask = counts > 0
    if not torch.any(mask):
        return 0.0
    return float((correct_k[mask].float() / counts[mask].float().clamp_min(1.0)).mean().item() * 100.0)


@torch.no_grad()
def _update_classwise_topk(logits, target, correct1, correct5, counts):
    maxk = min(5, logits.size(1))
    _, pred = logits.topk(maxk, 1, True, True)
    hit1 = pred[:, 0].eq(target)
    hit5 = pred.eq(target.unsqueeze(1)).any(dim=1)
    ones = torch.ones_like(target, dtype=counts.dtype)
    counts.index_add_(0, target, ones)
    correct1.index_add_(0, target, hit1.to(correct1.dtype))
    correct5.index_add_(0, target, hit5.to(correct5.dtype))


# ============ polar-express Muon optimizer ============

class MuonPolar(torch.optim.Optimizer):
    """Muon with Polar Express Newton-Schulz schedule; same API as Muon."""

    def __init__(self, muon_params, adamw_params, lr=1e-3, wd=0.1,
                 momentum=0.95, nesterov=True,
                 adamw_betas=(0.95, 0.95), adamw_eps=1e-8):
        defaults = dict(lr=lr, wd=wd, momentum=momentum, nesterov=nesterov,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps)
        params = [
            {"params": list(muon_params),  "use_muon": True},
            {"params": list(adamw_params), "use_muon": False},
        ]
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr  = group["lr"]
            wd  = group["wd"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if group["use_muon"]:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(g)
                    u = g.add(buf, alpha=mom) if nesterov else buf
                    u = zeropower_polar_express(u)
                    adjusted_lr = lr * 0.2 * math.sqrt(max(p.shape[0], p.shape[1]))
                    p.mul_(1.0 - lr * wd)
                    p.add_(u, alpha=-adjusted_lr)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"]    = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    if p.ndim > 1:
                        p.mul_(1.0 - lr * wd)
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    scale = (1 - beta1**t) / math.sqrt(1 - beta2**t)
                    p.addcdiv_(m, v.sqrt().add_(eps), value=-lr * scale)

        return loss


# ============ optimizer ============

def build_optimizer(model, args):
    if args.optimizer == 'muon':
        param_groups = lrd.param_groups_lrd_muon(
            model, args.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=args.layer_decay,
        )
        return Muon(param_groups, lr=args.lr, wd=args.weight_decay,
                    momentum=0.95, adamw_betas=(0.9, 0.95))

    if args.optimizer == 'muon_polar':
        param_groups = lrd.param_groups_lrd_muon(
            model, args.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=args.layer_decay,
        )
        muon_p  = [p for g in param_groups for p in g['params'] if g['use_muon']]
        adamw_p = [p for g in param_groups for p in g['params'] if not g['use_muon']]
        return MuonPolar(muon_params=muon_p, adamw_params=adamw_p,
                         lr=args.lr, wd=args.weight_decay)

    # adamw
    param_groups = lrd.param_groups_lrd(
        model, args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    return torch.optim.AdamW(param_groups, lr=args.lr)


# ============ validation ============

@torch.no_grad()
def validate(loader, model, num_classes):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    counts   = torch.zeros(num_classes, dtype=torch.long,    device="cuda")
    correct1 = torch.zeros(num_classes, dtype=torch.float32, device="cuda")
    correct5 = torch.zeros(num_classes, dtype=torch.float32, device="cuda")

    for inp, target in metric_logger.log_every(loader, 20, "Val:"):
        inp    = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(inp)
        loss   = F.cross_entropy(logits.float(), target)
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=inp.shape[0])
        metric_logger.meters["acc5"].update(acc5.item(), n=inp.shape[0])
        _update_classwise_topk(logits, target, correct1, correct5, counts)

    macro1 = _macro_from_counts(correct1, counts)
    macro5 = _macro_from_counts(correct5, counts)
    print(
        f"* Acc@1 {metric_logger.acc1.global_avg:.3f}"
        f"  Acc@5 {metric_logger.acc5.global_avg:.3f}"
        f"  MacroAcc@1 {macro1:.3f}"
        f"  MacroAcc@5 {macro5:.3f}"
        f"  loss {metric_logger.loss.global_avg:.3f}"
    )
    model.train()
    return {
        **{k: meter.global_avg for k, meter in metric_logger.meters.items()},
        "macro_acc1": macro1,
        "macro_acc5": macro5,
    }


def finetune_plantnet(args):
    print("git:\n  {}\n".format(get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ============
    model = models_vit.__dict__[args.model](
        num_classes=args.num_labels,
        drop_path_rate=args.drop_path,
        global_pool='avg',
    ).cuda()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built ({args.model}). Trainable params: {n_trainable:,}")

    if args.pretrained_weights:
        ckpt = torch.load(args.pretrained_weights, map_location='cpu', weights_only=False)
        print(f"Load pre-trained checkpoint from: {args.pretrained_weights}")
        ckpt_model = ckpt[args.checkpoint_key]
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del ckpt_model[k]
        interpolate_pos_embed(model, ckpt_model)
        msg = model.load_state_dict(ckpt_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=2e-5)

    # ============ transforms ============
    train_ops = [
        pth_transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
        pth_transforms.RandomHorizontalFlip(p=0.5),
    ]
    if args.randaugment_ops > 0:
        train_ops.append(pth_transforms.RandAugment(
            num_ops=args.randaugment_ops,
            magnitude=args.randaugment_magnitude,
            interpolation=InterpolationMode.BICUBIC,
        ))
    train_ops.extend([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if args.random_erasing_prob > 0:
        train_ops.append(pth_transforms.RandomErasing(p=args.random_erasing_prob, value="random"))
    train_transform = pth_transforms.Compose(train_ops)

    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # ============ evaluate-only path ============
    if args.evaluate:
        ckpt_path = args.eval_checkpoint or os.path.join(args.output_dir, "best_checkpoint.pth")
        restart_from_checkpoint(ckpt_path, model=model)
        dataset_test = PlantNetDataset(split="test", transform=val_transform, cache_dir=args.hf_cache_dir)
        test_loader  = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True,
        )
        test_stats = validate(test_loader, model, args.num_labels)
        print(
            f"Test acc@1={test_stats['acc1']:.2f}%  acc@5={test_stats['acc5']:.2f}%"
            f"  macro@1={test_stats['macro_acc1']:.2f}%  macro@5={test_stats['macro_acc5']:.2f}%"
        )
        return

    # ============ data loaders ============
    dataset_train = PlantNetDataset(split="train",      transform=train_transform, cache_dir=args.hf_cache_dir)
    dataset_val   = PlantNetDataset(split="validation", transform=val_transform,   cache_dir=args.hf_cache_dir)
    train_loader  = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Data loaded: {len(dataset_train)} train, {len(dataset_val)} val images.")
    train_iter = InfiniteLoader(train_loader)

    # ============ optimizer and scaler ============
    optimizer    = build_optimizer(model, args)
    loss_scaler  = NativeScaler()
    min_lr       = args.lr * args.min_lr_ratio

    mixup_fn = MixupCutmix(
        num_classes=args.num_labels,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        prob=args.mixup_prob,
        switch_prob=args.mixup_switch_prob,
        label_smoothing=args.label_smoothing,
    )

    # ============ resume from checkpoint ============
    to_restore = {"step": 0, "best_acc": 0.0, "best_macro_acc1": 0.0}
    restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    start_step      = to_restore["step"]
    best_acc        = to_restore["best_acc"]
    best_macro_acc1 = to_restore["best_macro_acc1"]

    # ============ training loop ============
    model.train()
    last_aug = "none"
    last_lam = 1.0

    for step in range(start_step, args.max_steps + 1):
        lr_t = get_lr(step, args.warmup_steps, args.max_steps, args.lr, min_lr)
        set_lr(optimizer, lr_t)

        running_loss = 0.0

        for microstep in range(args.grad_accum):
            inp, target = train_iter.next()
            inp    = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            inp, soft_target, last_aug, last_lam = mixup_fn(inp, target)

            is_last = (microstep == args.grad_accum - 1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(inp)
                loss   = soft_target_cross_entropy(logits, soft_target) / args.grad_accum

            running_loss += loss.item() * args.grad_accum
            loss_scaler(loss, optimizer,
                        clip_grad=args.clip_grad if is_last else None,
                        parameters=model.parameters(),
                        update_grad=is_last)

        optimizer.zero_grad()

        if args.log_interval > 0 and step % args.log_interval == 0 and step > 0:
            print(
                f"step {step:6d}/{args.max_steps}"
                f"  lr {lr_t:.3e}"
                f"  loss {running_loss:.4f}"
                f"  aug {last_aug}  lam {last_lam:.3f}"
            )

        if step % args.eval_interval == 0 and step > 0:
            val_stats = validate(val_loader, model, args.num_labels)
            acc1 = val_stats["acc1"]
            print(
                f"[step {step}] acc@1={acc1:.2f}%  acc@5={val_stats['acc5']:.2f}%"
                f"  macro@1={val_stats['macro_acc1']:.2f}%  macro@5={val_stats['macro_acc5']:.2f}%"
                f"  best={max(best_acc, acc1):.2f}%"
            )

            log_stats = {"step": step, "lr": lr_t, **{f"val_{k}": v for k, v in val_stats.items()}}
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if acc1 > best_acc:
                best_acc        = acc1
                best_macro_acc1 = val_stats["macro_acc1"]
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss_scaler": loss_scaler.state_dict(),
                    "best_acc": best_acc,
                    "best_macro_acc1": best_macro_acc1,
                }, os.path.join(args.output_dir, "best_checkpoint.pth"))

            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_scaler": loss_scaler.state_dict(),
                "best_acc": best_acc,
                "best_macro_acc1": best_macro_acc1,
            }, os.path.join(args.output_dir, "checkpoint.pth"))

    print(f"Training complete. Best val acc@1: {best_acc:.2f}%  macro@1: {best_macro_acc1:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PlantNet-300K full finetuning of MAE pretrained backbones")

    # Architecture
    parser.add_argument("--model", default="vit_base_patch16", type=str,
        help="Model name (vit_base_patch16 | vit_large_patch16)")
    parser.add_argument("--drop_path", default=0.1, type=float,
        help="Drop path rate")

    # Pretrained weights
    parser.add_argument("--pretrained_weights", default="", type=str)
    parser.add_argument("--checkpoint_key", default="model", type=str,
        help="Key in the checkpoint dict that holds the model state dict")

    # Optimizer
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon", "muon_polar"],
        help="adamw: AdamW with layer-wise LR decay; muon/muon_polar: Muon for 2D params")
    parser.add_argument("--lr", default=1e-3, type=float,
        help="Peak LR (no batch-size scaling)")
    parser.add_argument("--layer_decay", default=0.75, type=float,
        help="Layer-wise LR decay factor (BEiT/MAE convention)")
    parser.add_argument("--min_lr_ratio", default=0.01, type=float,
        help="min_lr = lr * min_lr_ratio (cosine decay floor)")
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--clip_grad", default=None, type=float,
        help="Gradient clip norm. None to disable.")

    # Training
    parser.add_argument("--max_steps", default=10000, type=int)
    parser.add_argument("--grad_accum", default=1, type=int,
        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--eval_interval", default=250, type=int,
        help="Run validation and save checkpoint every N steps")
    parser.add_argument("--log_interval", default=50, type=int,
        help="Print training stats every N steps. 0 to disable.")
    parser.add_argument("--num_labels", default=NUM_CLASSES, type=int)

    # Augmentation
    parser.add_argument("--randaugment_ops", default=2, type=int,
        help="Number of RandAugment ops. 0 to disable.")
    parser.add_argument("--randaugment_magnitude", default=9, type=int)
    parser.add_argument("--random_erasing_prob", default=0.25, type=float,
        help="Random Erasing probability. 0 to disable.")
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--mixup_alpha", default=0.8, type=float,
        help="Mixup alpha. 0 to disable.")
    parser.add_argument("--cutmix_alpha", default=1.0, type=float,
        help="CutMix alpha. 0 to disable.")
    parser.add_argument("--mixup_prob", default=1.0, type=float,
        help="Probability of applying Mixup or CutMix per batch.")
    parser.add_argument("--mixup_switch_prob", default=0.5, type=float,
        help="Probability of switching to CutMix when both are enabled.")

    # Data
    parser.add_argument("--hf_cache_dir", default=None, type=str,
        help="Override HuggingFace datasets cache directory")

    # Misc
    parser.add_argument("--output_dir", default=".", type=str)
    parser.add_argument("--evaluate", action="store_true",
        help="Run inference on the test split and exit (loads best_checkpoint.pth by default)")
    parser.add_argument("--eval_checkpoint", default="", type=str,
        help="Checkpoint to use for --evaluate (overrides best_checkpoint.pth)")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    finetune_plantnet(args)
