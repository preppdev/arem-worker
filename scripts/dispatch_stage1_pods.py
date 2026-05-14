"""One-shot dispatcher: spin up N RunPod pods to fan out the Stage-1 backfill.

Reuses the existing arem-worker container image (which already has
rclone configured for Dropbox + R2 and the Stage-1 NAFNet checkpoint
downloaded at startup), but each pod runs in MODE=backfill_stage1
instead of the production handler. Production endpoint is untouched.

Strategy:
  - Query the existing prod RunPod template (RUNPOD_TEMPLATE_ID) to get
    its image, env list, and containerRegistryAuthId — same auth our
    serverless workers use to pull from GHCR.
  - Split the candidate Job list across N slices, one pod per slice.
  - For each slice, podFindAndDeployOnDemand with MODE/OFFSET/LIMIT/SINCE
    overrides on top of the template's base env.
  - Pods run their slice and exit; RunPod stops billing when they stop.

Run from a workstation with RUNPOD_API_KEY + RUNPOD_TEMPLATE_ID in env:

    set -a && . /tmp/arem-prod.env && set +a
    # one-shot dispatch
    python scripts/dispatch_stage1_pods.py \\
        --pods 4 \\
        --gpu "NVIDIA RTX A4000" \\
        --total-jobs 644 \\
        --since 2025-01-01

The `--since 2025-01-01` flag is what makes this cover *all* historical
jobs (not just the post-may26 cutoff the endpoint defaults to). Pair it
with `--total-jobs` to size each pod's slice correctly.

After dispatch, the script prints pod IDs. Monitor them at
https://www.runpod.io/console/pods or via the GraphQL API.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def gql(api_key: str, query: str, variables: dict | None = None) -> dict:
    """Match the CI workflow's auth + headers — Bearer/header auth gets
    blocked by Cloudflare on some networks; query-string auth works."""
    payload = json.dumps({
        "query": query,
        "variables": variables or {},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={api_key}",
        data=payload, method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "arem-stage1-dispatcher/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
            if "errors" in data:
                print("GraphQL errors:", json.dumps(data["errors"], indent=2),
                      file=sys.stderr)
                raise SystemExit(1)
            return data["data"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise SystemExit(f"HTTP {e.code}: {body}") from e


def get_template(api_key: str, template_id: str) -> dict:
    """Pull the existing prod template — we'll reuse its image, env, and
    registry auth so the backfill pods authenticate to GHCR exactly the
    way the serverless workers do."""
    q = """query { myself { podTemplates {
        id name imageName containerDiskInGb volumeInGb volumeMountPath
        dockerArgs ports readme isServerless isPublic
        containerRegistryAuthId env { key value }
    } } }"""
    resp = gql(api_key, q)
    tpls = resp["myself"]["podTemplates"]
    tpl = next((t for t in tpls if t["id"] == template_id), None)
    if tpl is None:
        ids = [t["id"] for t in tpls]
        raise SystemExit(f"Template {template_id} not found in {ids}")
    return tpl


def merge_env(base_env: list[dict], overrides: dict[str, str]) -> list[dict]:
    """Combine the template's env list with our overrides — preserving
    existing keys and replacing those we want to override."""
    out: dict[str, str] = {e["key"]: e["value"] for e in (base_env or [])}
    out.update(overrides)
    return [{"key": k, "value": v} for k, v in out.items()]


def create_pod(
    api_key: str,
    name: str,
    template: dict,
    env_overrides: dict[str, str],
    gpu_type_id: str,
    container_disk_gb: int,
) -> dict:
    """Create one on-demand pod. Returns the pod row from RunPod."""
    env = merge_env(template.get("env") or [], env_overrides)
    input_var = {
        "cloudType": "ALL",
        "gpuCount": 1,
        "gpuTypeId": gpu_type_id,
        "minMemoryInGb": 8,
        "minVcpuCount": 2,
        "name": name,
        "imageName": template["imageName"],
        "dockerArgs": template.get("dockerArgs") or "",
        "containerDiskInGb": container_disk_gb,
        "volumeInGb": 0,
        "volumeMountPath": "",
        "ports": template.get("ports") or "",
        "env": env,
        "containerRegistryAuthId": template["containerRegistryAuthId"],
        "supportPublicIp": False,
    }
    m = """mutation P($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id desiredStatus imageName machineId
        }
    }"""
    resp = gql(api_key, m, {"input": input_var})
    return resp["podFindAndDeployOnDemand"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pods", type=int, default=4,
                    help="number of pods to spin up (default 4)")
    ap.add_argument("--total-jobs", type=int, required=True,
                    help="total candidate jobs across all pods (split evenly)")
    ap.add_argument("--offset-start", type=int, default=0,
                    help="offset of the first slice (default 0)")
    ap.add_argument("--since", type=str, default=None,
                    help="ISO date passed to the backfill script's --since")
    ap.add_argument("--gpu", type=str, default="NVIDIA RTX A4000",
                    help="GPU type ID. Other common: 'NVIDIA RTX A6000', "
                         "'NVIDIA A40'")
    ap.add_argument("--container-disk-gb", type=int, default=30)
    ap.add_argument("--name-prefix", type=str, default="arem-backfill-stage1")
    ap.add_argument("--dry-run", action="store_true",
                    help="print pod specs but don't create")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY") or ""
    template_id = os.environ.get("RUNPOD_TEMPLATE_ID") or ""
    if not api_key or not template_id:
        print("ERROR: set RUNPOD_API_KEY and RUNPOD_TEMPLATE_ID env vars",
              file=sys.stderr)
        return 2

    print(f"[dispatch] fetching template {template_id}...")
    tpl = get_template(api_key, template_id)
    print(f"[dispatch] template image: {tpl['imageName']}")
    print(f"[dispatch] env keys: {sorted({e['key'] for e in tpl.get('env') or []})}")

    # Split total jobs across pods, last pod gets the remainder.
    per_pod = args.total_jobs // args.pods
    remainder = args.total_jobs % args.pods
    slices: list[tuple[int, int]] = []
    cursor = args.offset_start
    for i in range(args.pods):
        limit = per_pod + (1 if i < remainder else 0)
        slices.append((cursor, limit))
        cursor += limit
    print(f"[dispatch] slicing {args.total_jobs} jobs across {args.pods} pods:")
    for i, (off, lim) in enumerate(slices):
        print(f"  pod {i}: offset={off:>4d}  limit={lim:>4d}")
    print()

    if args.dry_run:
        print("[dispatch] dry-run; not creating pods")
        return 0

    pod_ids: list[str] = []
    for i, (off, lim) in enumerate(slices):
        overrides = {
            "MODE": "backfill_stage1",
            "OFFSET": str(off),
            "LIMIT": str(lim),
        }
        if args.since:
            overrides["SINCE"] = args.since
        name = f"{args.name_prefix}-{i:02d}"
        print(f"[dispatch] creating pod {name}  offset={off}  limit={lim}...")
        try:
            pod = create_pod(
                api_key=api_key, name=name, template=tpl,
                env_overrides=overrides, gpu_type_id=args.gpu,
                container_disk_gb=args.container_disk_gb,
            )
            print(f"  → id={pod['id']}  status={pod['desiredStatus']}")
            pod_ids.append(pod["id"])
        except SystemExit as e:
            print(f"  FAIL: {e}", file=sys.stderr)
            # Don't abort the whole run on one pod failure
            continue

    print()
    print(f"[dispatch] launched {len(pod_ids)}/{args.pods} pods:")
    for pid in pod_ids:
        print(f"  https://www.runpod.io/console/pods/{pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
