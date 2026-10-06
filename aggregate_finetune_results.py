"""Aggregate finetune ablation results into per-epoch and summary CSVs.

Walks ``output_dir/finetune_ablation/*/log.txt`` (the JSONL written by
``main_finetune.py``, one record per epoch) and produces:

- ``per_epoch.csv``: one row per (run, epoch).
- ``summary.csv``: one row per run with best/final accuracy.

Run names follow ``{PRETRAIN_NAME}_finetuneopt_{OPT}_ckpt{N}`` as written by
``run_scripts/submit_finetune_ablations.sh``.

Usage:
    uv run python aggregate_finetune_results.py
    uv run python aggregate_finetune_results.py --root output_dir/finetune_ablation
"""

import argparse
import csv
import json
import re
from pathlib import Path

RUN_NAME_RE = re.compile(r"^(?P<pretrain>.+)_finetuneopt_(?P<opt>adamw|muon)_ckpt(?P<ckpt>\d+)$")


def parse_run_name(name):
    m = RUN_NAME_RE.match(name)
    if not m:
        return None
    pretrain = m.group("pretrain")
    if pretrain.startswith("base_"):
        model = "vit_base_patch16"
    elif pretrain.startswith("large_"):
        model = "vit_large_patch16"
    else:
        model = "unknown"
    return {
        "pretrain_ablation": pretrain,
        "finetune_optimizer": m.group("opt"),
        "ckpt_epoch": int(m.group("ckpt")),
        "model": model,
    }


def load_log(path):
    epochs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                epochs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="output_dir/finetune_ablation",
        help="Directory containing one subdir per finetune run (default: %(default)s)",
    )
    parser.add_argument(
        "--per-epoch-csv",
        default=None,
        help="Output path for per-epoch CSV (default: <root>/per_epoch.csv)",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Output path for summary CSV (default: <root>/summary.csv)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    per_epoch_path = Path(args.per_epoch_csv) if args.per_epoch_csv else root / "per_epoch.csv"
    summary_path = Path(args.summary_csv) if args.summary_csv else root / "summary.csv"

    per_epoch_rows = []
    summary_rows = []
    skipped = []

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        meta = parse_run_name(run_dir.name)
        if meta is None:
            skipped.append((run_dir.name, "name does not match expected pattern"))
            continue

        log_path = run_dir / "log.txt"
        if not log_path.is_file():
            skipped.append((run_dir.name, "no log.txt"))
            continue

        epochs = load_log(log_path)
        if not epochs:
            skipped.append((run_dir.name, "log.txt is empty"))
            continue

        for rec in epochs:
            per_epoch_rows.append({
                "pretrain_ablation": meta["pretrain_ablation"],
                "finetune_optimizer": meta["finetune_optimizer"],
                "ckpt_epoch": meta["ckpt_epoch"],
                "model": meta["model"],
                "epoch": rec.get("epoch"),
                "train_loss": rec.get("train_loss"),
                "train_lr": rec.get("train_lr"),
                "test_acc1": rec.get("test_acc1"),
                "test_acc5": rec.get("test_acc5"),
                "test_loss": rec.get("test_loss"),
            })

        # Summary: best-acc1 epoch + final epoch.
        valid = [r for r in epochs if isinstance(r.get("test_acc1"), (int, float))]
        if valid:
            best = max(valid, key=lambda r: r["test_acc1"])
            best_acc1 = best["test_acc1"]
            best_acc1_epoch = best.get("epoch")
            best_acc5 = best.get("test_acc5")
        else:
            best_acc1 = best_acc1_epoch = best_acc5 = None
        final = epochs[-1]
        summary_rows.append({
            "pretrain_ablation": meta["pretrain_ablation"],
            "finetune_optimizer": meta["finetune_optimizer"],
            "ckpt_epoch": meta["ckpt_epoch"],
            "model": meta["model"],
            "finetune_epochs_completed": len(epochs),
            "best_acc1": best_acc1,
            "best_acc1_epoch": best_acc1_epoch,
            "best_acc5": best_acc5,
            "final_acc1": final.get("test_acc1"),
            "final_train_loss": final.get("train_loss"),
        })

    sort_key = lambda r: (r["pretrain_ablation"], r["finetune_optimizer"], r.get("epoch", 0))
    per_epoch_rows.sort(key=sort_key)
    summary_rows.sort(key=lambda r: (r["pretrain_ablation"], r["finetune_optimizer"]))

    per_epoch_fields = [
        "pretrain_ablation", "finetune_optimizer", "ckpt_epoch", "model",
        "epoch", "train_loss", "train_lr", "test_acc1", "test_acc5", "test_loss",
    ]
    with open(per_epoch_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_epoch_fields)
        writer.writeheader()
        writer.writerows(per_epoch_rows)

    summary_fields = [
        "pretrain_ablation", "finetune_optimizer", "ckpt_epoch", "model",
        "finetune_epochs_completed", "best_acc1", "best_acc1_epoch", "best_acc5",
        "final_acc1", "final_train_loss",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote {per_epoch_path} ({len(per_epoch_rows)} rows)")
    print(f"wrote {summary_path} ({len(summary_rows)} rows)")
    if skipped:
        print(f"\nskipped {len(skipped)} dir(s):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
