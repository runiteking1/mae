"""Summarize few-shot runs under output_dir/fewshot/<draw>/<run>/.

    uv run python fewshot/summarize_fewshot.py [ROOT] [--last K] [--smooth N] [--no_plot]

Prints a table of val acc1 per run: final, mean over the last K evals, max (picked
on the val set, so optimistic), epochs done.

Also writes output_dir/fewshot/<draw>/curves_e<EPOCHS>.png (one per epoch budget): train loss (per step, from
tensorboard events, N-step moving average), val loss and val acc1 (from log.txt)
vs. epoch. Color = init (pretrain checkpoint / scratch), line style = finetune LR.
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

RUN_RE = re.compile(r"^(?P<tag>.+)_ft-(?P<opt>[a-z_]+)_lr(?P<lr>[0-9.e+-]+)(?:_e(?P<budget>\d+))?_seed(?P<seed>\d+)$")
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


def epoch_budget(run_dir, r):
    """Planned epochs from the run name; older runs without it: the epochs logged."""
    if r["budget"]:
        return int(r["budget"])
    return int(val_curves(run_dir)[0][-1]) if (run_dir / "log.txt").stat().st_size else 0


def plot_draw(draw_dir, k):
    by_budget = {}
    for d in sorted(p for p in draw_dir.iterdir() if (p / "log.txt").is_file()):
        m = RUN_RE.match(d.name)
        if m:
            r = m.groupdict()
            by_budget.setdefault(epoch_budget(d, r), []).append((d, r))
    return [plot_runs(runs, draw_dir, budget, k) for budget, runs in sorted(by_budget.items())]


def plot_runs(runs, draw_dir, budget, k):

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
    fig.suptitle(f"{draw_dir.name}, {budget} epochs", fontsize=FONT_SIZE)
    fig.tight_layout()

    out = draw_dir / f"curves_e{budget}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def print_table(root, last):
    rows = []
    for log in sorted(Path(root).glob("*/*/log.txt")):
        recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        acc = [r["test_acc1"] for r in recs if "test_acc1" in r]
        if not acc:
            continue
        tail = acc[-last:]
        rows.append((f"{log.parent.parent.name}/{log.parent.name}",
                     acc[-1], sum(tail) / len(tail), max(acc), recs[-1]["epoch"] + 1))

    if not rows:
        print(f"no log.txt under {root}")
        return False
    w = max(len(r[0]) for r in rows)
    print(f"{'run':<{w}}  {'final':>6}  {'last' + str(last):>6}  {'max':>6}  {'epochs':>6}")
    for name, final, tail, best, ep in rows:
        print(f"{name:<{w}}  {final:6.2f}  {tail:6.2f}  {best:6.2f}  {ep:6d}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="output_dir/fewshot")
    p.add_argument("--last", type=int, default=4, help="average acc1 over the last K evals")
    p.add_argument("--smooth", type=int, default=40, help="moving-average window for train loss, in steps")
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    if not print_table(args.root, args.last) or args.no_plot:
        return
    print()
    for d in sorted(Path(args.root).iterdir()):
        if d.is_dir():
            for out in plot_draw(d, args.smooth):
                print(f"saved {out}")


if __name__ == "__main__":
    main()
