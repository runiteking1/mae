"""Plot val_macro_acc@1 over training steps for all ViT-B PlantNet runs.

Usage:
    uv run python plot_plantnet_finetune.py
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

ROOT = Path("output_dir/plantnet")

PRETRAIN_CONFIGS = [
    ("base_adamw_lr2.4e-3", "PT: AdamW", None,  1),   # no marker
    ("base_muon_lr2.4e-3",  "PT: Muon",  "o",   1000),  # circle every 1000 steps
]

# Display order for augmentation configs
AUG_CONFIGS = [
    ("full",           "Full aug"),
    ("no_rand",        "No RandAug"),
    ("no_mix",         "No Mixup"),
    ("no_mix_no_rand", "No Mixup+RandAug"),
]

# Finetune optimizer → (linestyle, legend label)
FT_OPT_CONFIGS = [
    ("adamw",      "-",  "FT: AdamW"),
    # ("muon",       ":",  "FT: Muon"),
    ("muon_polar", "--", "FT: MuonPolar"),
]

# One color per augmentation recipe (complementary pairs: orange/blue, green/purple)
COLORS = {
    "full":           "#e8851a",  # orange
    "no_rand":        "#2171b5",  # blue
    "no_mix":         "#2ca02c",  # green
    "no_mix_no_rand": "#9467bd",  # purple
}

FONT_SIZE = 7
TICK_SIZE = 6

plt.rcParams.update({
    "axes.labelsize": FONT_SIZE + 1,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": FONT_SIZE,
    "font.family": "sans-serif",
})

fig, ax = plt.subplots(figsize=(7, 4))

missing = []
for pt_name, _, pt_marker, pt_markevery in PRETRAIN_CONFIGS:
    for aug_name, _ in AUG_CONFIGS:
        for ft_name, ft_ls, _ in FT_OPT_CONFIGS:
            log_path = ROOT / pt_name / f"ep500_{aug_name}_{ft_name}" / "log.txt"
            if not log_path.is_file():
                missing.append(str(log_path))
                continue

            records = [
                json.loads(line)
                for line in log_path.read_text().splitlines()
                if line.strip()
            ]
            steps = [r["step"] for r in records]
            macro_acc1 = [r["val_macro_acc1"] for r in records]

            # markevery by step value: find indices where step % pt_markevery == 0
            if pt_marker is not None:
                mark_idx = [i for i, s in enumerate(steps) if s % pt_markevery == 0]
            else:
                mark_idx = []

            ax.plot(
                steps,
                macro_acc1,
                color=COLORS[aug_name],
                linestyle=ft_ls,
                linewidth=0.9,
                alpha=0.85,
                marker=pt_marker,
                markevery=mark_idx,
                markersize=3,
                markeredgewidth=0,
            )

if missing:
    print("WARNING: missing log files:")
    for p in missing:
        print(f"  {p}")

ax.set_xlabel("Training step")
ax.set_ylabel("Macro Acc@1 (%)")
ax.set_title("ViT-B PlantNet — Macro Acc@1 vs. training step")
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.set_xlim(left=0)

# --- Three-section legend placed to the right ---
# Section 1: color → augmentation recipe
color_handles = [
    mpatches.Patch(color=COLORS[aug_name], label=aug_label)
    for aug_name, aug_label in AUG_CONFIGS
]

separator = mpatches.Patch(visible=False, label="")

# Section 2: marker → pretrain optimizer
marker_handles = [
    mlines.Line2D([], [], color="gray", linestyle="-", linewidth=1.1,
                  marker=marker, markersize=4, markeredgewidth=0, label=label)
    for _, label, marker, _ in PRETRAIN_CONFIGS
]

# Section 3: linestyle → finetune optimizer
style_handles = [
    mlines.Line2D([], [], color="black", linestyle=ls, linewidth=1.1, label=label)
    for _, ls, label in FT_OPT_CONFIGS
]

legend = ax.legend(
    handles=color_handles + [separator] + marker_handles + [separator] + style_handles,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    frameon=True,
    framealpha=0.9,
    edgecolor="#cccccc",
    fontsize=FONT_SIZE,
    ncol=1,
    handlelength=2.0,
)

out_path = ROOT / "plot_macro_acc1.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"saved {out_path}")
