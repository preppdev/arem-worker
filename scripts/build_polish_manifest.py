"""Build a pairs.csv for the Stage-2 polish fine-tune.

Layout expected:
    <root>/
        <name>.jpg          # pipeline output (input)
        after/
            <name>.jpg      # Lightroom-edited target (same filename)

Writes <root>/pairs.csv with columns: pair_id, input, output.
Inputs/outputs are stored RELATIVE to <root> so the trainer's --dataset-root
+ --path-prefix flow works unchanged.

Pairs where the after/<name> is missing or unreadable are skipped (with a
report at the end).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (inputs at top, targets in <root>/after/)")
    ap.add_argument("--out", default=None, help="CSV path (default: <root>/pairs.csv)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"root not found: {root}")
    after = root / "after"
    if not after.is_dir():
        sys.exit(f"after/ subdir not found: {after}")

    out_csv = Path(args.out) if args.out else root / "pairs.csv"

    inputs = sorted(p for p in root.iterdir() if p.suffix in EXTS and p.is_file())
    if not inputs:
        sys.exit(f"no input images in {root}")

    rows = []
    missing = []
    for ip in inputs:
        tp = after / ip.name
        if not tp.is_file():
            missing.append(ip.name)
            continue
        rows.append({
            "pair_id": ip.stem,
            "input": ip.name,                 # relative to root
            "output": f"after/{ip.name}",     # relative to root
        })

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "input", "output"])
        w.writeheader()
        w.writerows(rows)

    print(f"manifest: {out_csv}")
    print(f"  pairs:   {len(rows)}")
    print(f"  missing: {len(missing)}")
    if missing:
        print(f"  first 5 missing: {missing[:5]}")


if __name__ == "__main__":
    main()
