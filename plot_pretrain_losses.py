import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ablation_dir = Path("output_dir/ablation")

runs = [
    ("ViT-B AdamW", ablation_dir / "base_adamw_lr2.4e-3"  / "log.txt", "tab:blue",   "-",  "base"),
    ("ViT-B Muon",  ablation_dir / "base_muon_lr2.4e-3"   / "log.txt", "tab:orange", "-",  "base"),
    ("ViT-L AdamW", ablation_dir / "large_adamw_lr2.4e-3" / "log.txt", "tab:blue",   "--", "large"),
    ("ViT-L Muon",  ablation_dir / "large_muon_lr2.4e-3"  / "log.txt", "tab:orange", "--", "large"),
]

loaded = []
for label, log_path, color, ls, size in runs:
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    epochs = [r["epoch"] for r in records]
    losses = [r["train_loss"] for r in records]
    loaded.append((label, epochs, losses, color, ls, size))

# cap at the minimum last epoch within each model size, judged by epoch value not line count
max_epoch_per_run = {(label, size): max(epochs) for label, epochs, _, _, _, size in loaded}
min_last_epoch = {
    size: min(v for (_, s), v in max_epoch_per_run.items() if s == size)
    for size in ("base", "large")
}

ZOOM_START, ZOOM_END = 250, 540
FONT_SIZE = 7
TICK_SIZE = 5

plt.rcParams.update({
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": FONT_SIZE,
})

fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(7, 2.5))

handles = []
for label, epochs, losses, color, ls, size in loaded:
    cap = min_last_epoch[size]
    mask = [e <= cap for e in epochs]
    ep = [e for e, m in zip(epochs, mask) if m]
    lo = [l for l, m in zip(losses, mask) if m]

    (line,) = ax_full.plot(ep, lo, color=color, linestyle=ls)
    handles.append((line, label))

    zoom_mask = [e >= ZOOM_START for e in ep]
    ax_zoom.plot(
        [e for e, m in zip(ep, zoom_mask) if m],
        [l for l, m in zip(lo, zoom_mask) if m],
        color=color, linestyle=ls,
    )

for ax in (ax_full, ax_zoom):
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

ax_full.set_ylabel("Train Loss", fontsize=FONT_SIZE)

ZOOM_YMIN, ZOOM_YMAX = 4e-1, 4.8e-1
rect = mpatches.Rectangle(
    (ZOOM_START, ZOOM_YMIN),
    ZOOM_END - ZOOM_START,
    ZOOM_YMAX - ZOOM_YMIN,
    linewidth=0.8, edgecolor="black", facecolor="none", linestyle="--", zorder=5,
)
ax_full.add_patch(rect)

ax_full.legend(
    [h for h, _ in handles],
    [l for _, l in handles],
    fontsize=FONT_SIZE,
    ncol=1,
    frameon=False,
)

fig.tight_layout()
plt.savefig("pretrain_losses.png", dpi=300, bbox_inches="tight")
plt.show()
