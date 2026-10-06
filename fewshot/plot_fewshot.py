"""Plot few-shot curves: train loss (per step, from tensorboard events), val loss and
val acc1 (from log.txt), one figure per draw under output_dir/fewshot/<draw>/.

    uv run python fewshot/plot_fewshot.py [ROOT] [--smooth 40]

Color = init (pretrain checkpoint / scratch), line style = finetune LR.
Train loss is the mixup/cutmix + label-smoothing soft-target loss, so it is not on
the same scale as val loss (plain CE).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FONT_SIZE = 7
TICK_SIZE = 5
plt.rcParams.update({
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
})

RUN_RE = re.compile(r"^(?P<tag>.+)_ft-(?P<opt>[a-z_]+)_lr(?P<lr>[0-9.e+-]+)_seed(?P<seed>\d+)$")
STYLES = ["-", "--", ":", "-."]


def train_loss(run_dir):
    """Per-step train loss from all event files in run_dir; x in epochs."""
    pts = {}
    for ev in sorted(run_dir.glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        acc.Reload()
        if "loss" in acc.Tags()["scalars"]:
            for e in acc.Scalars("loss"):
                pts[e.step] = e.value  # later files win on duplicate steps
    if not pts:
        return None, None
    steps = np.array(sorted(pts))
    return steps / 1000.0, np.array([pts[s] for s in steps])  # step axis is epoch_1000x


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def val_curves(run_dir):
    recs = [json.loads(l) for l in (run_dir / "log.txt").read_text().splitlines() if l.strip()]
    recs = [r for r in recs if "test_acc1" in r]
    ep = np.array([r["epoch"] + 1 for r in recs])
    return ep, np.array([r["test_loss"] for r in recs]), np.array([r["test_acc1"] for r in recs])


def short_tag(tag):
    m = re.search(r"_ep(\d+)$", tag)
    return f"ckpt {m.group(1)}" if m else tag


def plot_draw(draw_dir, k):
    runs = []
    for d in sorted(p for p in draw_dir.iterdir() if (p / "log.txt").is_file()):
        m = RUN_RE.match(d.name)
        if m:
            runs.append((d, m.groupdict()))
    if not runs:
        return None

    tags = sorted({r["tag"] for _, r in runs}, key=lambda t: (t == "scratch", t))
    runs.sort(key=lambda dr: (tags.index(dr[1]["tag"]), dr[1]["opt"], float(dr[1]["lr"]), dr[1]["seed"]))
    color = {t: f"C{i}" for i, t in enumerate(tags)}
    multi_opt = len({r["opt"] for _, r in runs}) > 1
    lr_style = {}  # per tag, so each init's LR grid uses the same style sequence
    for t in tags:
        for i, lr in enumerate(sorted({r["lr"] for _, r in runs if r["tag"] == t}, key=float)):
            lr_style[(t, lr)] = STYLES[i % len(STYLES)]

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
    for d, r in runs:
        label = f"{short_tag(r['tag'])}, lr {r['lr']}" + (f", {r['opt']}" if multi_opt else "")
        kw = dict(color=color[r["tag"]], linestyle=lr_style[(r["tag"], r["lr"])], linewidth=0.9)

        x, y = train_loss(d)
        if x is not None:
            ys = smooth(y, k)
            axes[0].plot(x[len(x) - len(ys):], ys, label=label, **kw)

        ep, vloss, vacc = val_curves(d)
        axes[1].plot(ep, vloss, label=label, **kw)
        axes[2].plot(ep, vacc, label=label, **kw)

    axes[0].set_title(f"train loss (per step, {k}-step moving avg)")
    axes[1].set_title("val loss")
    axes[2].set_title("val acc@1 (%)")
    n_classes = len(json.loads((runs[0][0] / "data_meta.json").read_text())["classes"]) \
        if (runs[0][0] / "data_meta.json").is_file() else None
    if n_classes:
        axes[2].axhline(100 / n_classes, color="gray", linewidth=0.6, linestyle=":")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    axes[2].legend(frameon=False, loc="best")
    fig.suptitle(draw_dir.name, fontsize=FONT_SIZE)
    fig.tight_layout()

    out = draw_dir / "curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="output_dir/fewshot")
    p.add_argument("--smooth", type=int, default=40, help="moving-average window for train loss, in steps")
    args = p.parse_args()

    outs = [plot_draw(d, args.smooth) for d in sorted(Path(args.root).iterdir()) if d.is_dir()]
    outs = [o for o in outs if o]
    if not outs:
        print(f"no runs under {args.root}")
    for o in outs:
        print(f"saved {o}")


if __name__ == "__main__":
    main()
