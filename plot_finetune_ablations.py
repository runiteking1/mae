import json
import re
from pathlib import Path
import matplotlib.pyplot as plt

base_dir = Path("output_dir/finetune_ablation")

FONT_SIZE = 7
TICK_SIZE = 5

plt.rcParams.update({
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": FONT_SIZE,
})

# color by pretrain opt, marker by finetune opt
COLOR = {"adamw": "tab:blue", "muon": "tab:orange"}
MARKER = {"adamw": None, "muon": "o"}
MARKEVERY = 5

dirs = sorted([
    d for d in base_dir.iterdir()
    if d.is_dir() and re.match(
        r"^(base_.*_lr2\.4e-3_finetuneopt_.*_ckpt500|large_.*_lr2\.4e-3_finetuneopt_.*_ckpt400)$",
        d.name,
    )
])


def parse_log(d):
    epochs, accs = [], []
    with open(d / "log.txt") as f:
        for line in f:
            row = json.loads(line)
            epochs.append(row["epoch"])
            accs.append(row["test_acc1"])
    return epochs, accs


def parse_name(name):
    m = re.match(
        r"(base|large)_(adamw|muon)_lr[\d.e-]+_finetuneopt_(adamw|muon)_ckpt\d+",
        name,
    )
    return m.group(1), m.group(2), m.group(3)  # size, pretrain_opt, finetune_opt


fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))

group = [d for d in dirs if d.name.startswith("large_")]
group.sort(key=lambda d: parse_name(d.name)[1:])

data = {}
for d in group:
    _, pretrain_opt, finetune_opt = parse_name(d.name)
    data[(pretrain_opt, finetune_opt)] = parse_log(d)

base_epochs, base_accs = data[("muon", "muon")]

for (pretrain_opt, finetune_opt), (epochs, accs) in sorted(data.items()):
    if (pretrain_opt, finetune_opt) == ("muon", "muon"):
        continue
    diff = [a - b for a, b in zip(accs, base_accs)]
    line, = ax.plot(
        epochs, diff,
        color=COLOR[pretrain_opt],
        marker=MARKER[finetune_opt],
        markevery=MARKEVERY,
        markersize=2.5,
        linewidth=1.0,
        label=f"PT={pretrain_opt}, FT={finetune_opt}",
    )

baseline, = ax.plot([], [], color="tab:orange", linewidth=0.8, linestyle="--")
ax.axhline(0, color="tab:orange", linewidth=0.8, linestyle="--")

ax.set_title("ViT-L", fontsize=FONT_SIZE)
ax.set_xlabel("Epoch", fontsize=FONT_SIZE)
ax.set_ylabel("Δ Val Acc@1 vs. Muon/Muon (%)", fontsize=FONT_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
ax.grid(True, alpha=0.3)
ax.legend(
    fontsize=FONT_SIZE,
    loc="upper right",
    frameon=False,
)

fig.tight_layout()
out = Path("finetune_ablation_acc1.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved to {out}")
