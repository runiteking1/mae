"""Build a few-shot ImageNet subset as an ImageFolder tree from the HF parquet cache.

Output layout (consumed by main_finetune.py --data_source imagefolder):

    OUT/train/<cc>_<orig_label>/<file>   N_CLASSES x SHOTS images
    OUT/val/<cc>_<orig_label>/<file>     all 50 val images per class
    OUT/meta.json

Class dirs are prefixed with a zero-padded rank, so ImageFolder labels are 0..N-1
in selection order. Image bytes are copied verbatim from the parquet (no re-encode).

Example (8 classes x 8 shots, classes and shots drawn with seed 0):
    uv run python fewshot/make_fewshot_imagenet.py \
        --data_path ${HOME}/.cache/huggingface/datasets/imagenet/imagenet/data \
        --out ${HOME}/data/imagenet_fewshot/c8_s8_cs0_ss0 \
        --n_classes 8 --shots 8 --class_seed 0 --shot_seed 0
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import Image, load_dataset


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", required=True, help="HF parquet dir (same as --data_path for training)")
    p.add_argument("--out", required=True)
    p.add_argument("--n_classes", type=int, default=8)
    p.add_argument("--shots", type=int, default=8)
    p.add_argument("--classes", type=str, default="",
                   help="comma-separated ImageNet label indices; overrides --n_classes/--class_seed")
    p.add_argument("--class_seed", type=int, default=0, help="seed for which classes are drawn")
    p.add_argument("--shot_seed", type=int, default=0, help="seed for which train images are drawn")
    p.add_argument("--val_per_class", type=int, default=0, help="0 = all val images of each class")
    return p.parse_args()


def labels_of(ds):
    return ds.data.column("label").to_numpy()


def export(ds, idx_by_class, class_dirs, split_dir):
    flat = [(c, i) for c, idxs in enumerate(idx_by_class) for i in idxs]
    sub = ds.select([i for _, i in flat]).cast_column("image", Image(decode=False))
    for (c, i), row in zip(flat, sub):
        img = row["image"]
        data = img["bytes"]
        if data is None:
            with open(img["path"], "rb") as f:
                data = f.read()
        name = os.path.basename(img["path"]) if img.get("path") else f"{i:08d}"
        if not name.lower().endswith((".jpeg", ".jpg", ".png")):
            name += ".JPEG"  # ImageFolder skips files without an image extension
        d = split_dir / class_dirs[c]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / name, "wb") as f:
            f.write(data)


def main():
    args = get_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite")

    train = load_dataset(args.data_path, split="train")
    val = load_dataset(args.data_path, split="validation")
    names = train.features["label"].names

    train_labels = labels_of(train)
    val_labels = labels_of(val)
    n_total = len(names)

    if args.classes:
        classes = [int(c) for c in args.classes.split(",")]
    else:
        classes = sorted(np.random.default_rng(args.class_seed)
                         .choice(n_total, size=args.n_classes, replace=False).tolist())
    width = len(str(len(classes) - 1))
    class_dirs = [f"{r:0{width}d}_{c:04d}" for r, c in enumerate(classes)]

    shot_rng = np.random.default_rng(args.shot_seed)
    train_idx = []
    for c in classes:
        pool = np.flatnonzero(train_labels == c)
        train_idx.append(sorted(shot_rng.choice(pool, size=args.shots, replace=False).tolist()))

    val_idx = []
    for c in classes:
        pool = np.flatnonzero(val_labels == c)
        if args.val_per_class:
            pool = np.sort(np.random.default_rng(0).choice(pool, size=args.val_per_class, replace=False))
        val_idx.append(pool.tolist())

    export(train, train_idx, class_dirs, out / "train")
    export(val, val_idx, class_dirs, out / "val")

    meta = {
        "classes": classes,
        "class_names": [names[c] for c in classes],
        "class_dirs": class_dirs,
        "shots": args.shots,
        "class_seed": None if args.classes else args.class_seed,
        "shot_seed": args.shot_seed,
        "train_indices": train_idx,
        "n_val": sum(len(v) for v in val_idx),
        "data_path": args.data_path,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {out}")
    for d, n in zip(class_dirs, meta["class_names"]):
        print(f"  {d}  {n}")
    print(f"train: {len(classes)} x {args.shots}  val: {meta['n_val']}")


if __name__ == "__main__":
    main()
