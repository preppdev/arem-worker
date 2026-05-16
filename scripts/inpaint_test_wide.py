"""Wide inpaint test driver — runs scripts/inpaint_text_edit.py across
many conditions in sequence, posting results to the dashboard's
/api/internal/inpaint-test endpoint so /inpaint-tests/<condition>
renders side-by-side galleries.

Designed for "I'll be away for a few hours, queue everything up" runs.

Default plan (Replicate FLUX Kontext Pro at ~$0.04/img):

  condition           limit   est_cost
  reflection            50    $2.00
  dead-fixture          30    $1.20
  finger-in-photo       30    $1.20
  sun-flare             30    $1.20
  lens-hood-in-photo    20    $0.80
  glare                 20    $0.80
  ---
  total                180    $7.20
  wall time            ~85 min at ~28s/img

Usage:
    python -m scripts.inpaint_test_wide \\
        --provider replicate --model black-forest-labs/flux-kontext-pro \\
        --tag wide-v1

    # Override per-condition limits
    python -m scripts.inpaint_test_wide \\
        --conditions reflection:50,dead-fixture:30,finger-in-photo:30

Resumable: each per-condition run skips images already in the dashboard
for the same (condition, modelRun). Safe to re-invoke anytime.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

DEFAULT_PLAN = [
    ("reflection", 50),
    ("dead-fixture", 30),
    ("finger-in-photo", 30),
    ("sun-flare", 30),
    ("lens-hood-in-photo", 20),
    ("glare", 20),
]


def parse_conditions(s: str | None) -> list[tuple[str, int]]:
    if not s:
        return DEFAULT_PLAN
    out: list[tuple[str, int]] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            out.append((chunk, 30))
            continue
        cond, lim = chunk.split(":", 1)
        out.append((cond.strip(), int(lim.strip())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="replicate")
    ap.add_argument("--model", default="black-forest-labs/flux-kontext-pro")
    ap.add_argument("--tag", default="wide-v1")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated 'cond[:limit]' overrides default plan")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = parse_conditions(args.conditions)
    total = sum(lim for _, lim in plan)
    est = total * 0.04
    print(f"=== WIDE INPAINT TEST PLAN ===", flush=True)
    print(f"  provider: {args.provider}", flush=True)
    print(f"  model:    {args.model}", flush=True)
    print(f"  tag:      {args.tag}", flush=True)
    print(f"  plan:", flush=True)
    for cond, lim in plan:
        print(f"    {cond}: {lim}", flush=True)
    print(f"  ----", flush=True)
    print(f"  total images: {total}  est cost (Replicate FLUX): ${est:.2f}",
          flush=True)
    print(f"==============================", flush=True)
    if args.dry_run:
        print("(dry-run; not executing)", flush=True)
        return 0

    t0 = time.time()
    for i, (cond, lim) in enumerate(plan, start=1):
        elapsed = time.time() - t0
        print(f"\n=== [{i}/{len(plan)}] {cond} (limit {lim}) "
              f"(elapsed {elapsed/60:.1f}min) ===", flush=True)
        cmd = [
            sys.executable, "-m", "scripts.inpaint_text_edit",
            "--condition", cond,
            "--provider", args.provider,
            "--model", args.model,
            "--tag", args.tag,
            "--limit", str(lim),
        ]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  WARN condition {cond} exited rc={r.returncode}",
                  flush=True)
    total_elapsed = time.time() - t0
    print(f"\n=== WIDE TEST DONE in {total_elapsed/60:.1f} min ===",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
