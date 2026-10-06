"""Tabulate val accuracy of few-shot runs: output_dir/fewshot/<draw>/<run>/log.txt.

    uv run python fewshot/summarize_fewshot.py [ROOT] [--last K]

Columns: final acc1, mean acc1 over the last K evals, max acc1 (picked on the
val set, so optimistic), epochs done. Sorted by run name.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="output_dir/fewshot")
    p.add_argument("--last", type=int, default=4, help="average acc1 over the last K evals")
    args = p.parse_args()

    rows = []
    for log in sorted(Path(args.root).glob("*/*/log.txt")):
        recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        acc = [r["test_acc1"] for r in recs if "test_acc1" in r]
        if not acc:
            continue
        tail = acc[-args.last:]
        rows.append((f"{log.parent.parent.name}/{log.parent.name}",
                     acc[-1], sum(tail) / len(tail), max(acc), recs[-1]["epoch"] + 1))

    if not rows:
        print(f"no log.txt under {args.root}")
        return
    w = max(len(r[0]) for r in rows)
    print(f"{'run':<{w}}  {'final':>6}  {'last' + str(args.last):>6}  {'max':>6}  {'epochs':>6}")
    for name, final, tail, best, ep in rows:
        print(f"{name:<{w}}  {final:6.2f}  {tail:6.2f}  {best:6.2f}  {ep:6d}")


if __name__ == "__main__":
    main()
