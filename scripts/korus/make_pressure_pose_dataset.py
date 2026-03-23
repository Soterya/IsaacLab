#!/usr/bin/env python3
"""
Convert episodic Korus GT npz files into per-(frame,env) npz files named:

  posename_{pose_name}_timestamp_{timestamp}.npz

Episode number is ignored.

Input episodic format (your current saver):
  timestamps        : [T]
  pose_names        : [B]
  body_names        : [Nb]
  smpl_joint_order  : [J]
  root_pos_w        : [T,B,3]
  root_rotvec_w     : [T,B,3]
  spine_pos_w       : [T,B,3]
  spine_rotvec_w    : [T,B,3]
  body_pos_w        : [T,B,Nb,3]
  body_rotvec_w     : [T,B,Nb,3]
  pose_body_rotvec  : [T,B,J,3]
  pressure_maps     : [T,B,Ncells,H,W]

Output per file (single env, single frame):
  timestamp         : scalar
  pose_name         : string
  body_names        : [Nb]
  smpl_joint_order  : [J]
  root_pos_w        : [3]
  root_rotvec_w     : [3]
  spine_pos_w       : [3]
  spine_rotvec_w    : [3]
  body_pos_w        : [Nb,3]
  body_rotvec_w     : [Nb,3]
  pose_body_rotvec  : [J,3]
  pressure_maps     : [Ncells,H,W]
"""

import argparse
import os
import re
from pathlib import Path
import numpy as np


EP_RE = re.compile(r"episode_(\d+).*_gt\.npz$")


def _natural_sort_key(p: Path):
    m = EP_RE.search(p.name)
    if m:
        return (0, int(m.group(1)), p.name)
    return (1, p.name)


def _safe_slug(s: str) -> str:
    """
    Make a filename-safe token:
      - keep letters/digits/_/-
      - convert spaces to _
      - drop other chars
    """
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    # avoid empty
    return s if s else "unknown"


def _format_timestamp(ts: float, decimals: int) -> str:
    """
    Use fixed decimals so filenames are consistent and sortable.
    """
    fmt = f"{{:.{decimals}f}}"
    s = fmt.format(float(ts))
    # make it filename-safe and consistent: 12.340000 -> 12p340000
    # (avoids extra '.' confusion in some tooling)
    s = s.replace(".", "p")
    # also avoid negative sign weirdness
    s = s.replace("-", "m")
    return s


def load_episode(path: Path):
    data = np.load(path, allow_pickle=True)
    required = ["timestamps", "pose_names", "body_names", "smpl_joint_order", "pressure_maps"]
    for k in required:
        if k not in data:
            raise RuntimeError(f"{path} missing key '{k}'")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing episode_XXXX_*_gt.npz files")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write posename_*_timestamp_*.npz files")
    ap.add_argument("--ts_decimals", type=int, default=6, help="Timestamp decimals in filename (default: 6)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files with same name")
    ap.add_argument("--dedupe", action="store_true",
                    help="If a name collision occurs, append _dup{n} instead of erroring (ignored if --overwrite)")
    ap.add_argument("--dry_run", action="store_true", help="Print what would be written, but do not write files")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(
        [p for p in in_dir.iterdir() if p.suffix == ".npz" and EP_RE.search(p.name)],
        key=_natural_sort_key
    )
    if not episode_files:
        raise SystemExit(f"[ERROR] No episode_*_gt.npz files found in: {in_dir}")

    print(f"[INFO] Found {len(episode_files)} episode files.")
    print(f"[INFO] Writing to: {out_dir}")
    print(f"[INFO] Filename timestamp decimals: {args.ts_decimals}")

    optional_keys = [
        "root_pos_w", "root_rotvec_w",
        "spine_pos_w", "spine_rotvec_w",
        "body_pos_w", "body_rotvec_w",
        "pose_body_rotvec",
    ]

    written = 0

    for ep_path in episode_files:
        ep = load_episode(ep_path)

        timestamps = ep["timestamps"]               # [T]
        pose_names = ep["pose_names"]               # [B]
        body_names = ep["body_names"]               # [Nb]
        smpl_joint_order = ep["smpl_joint_order"]   # [J]
        pressure_maps = ep["pressure_maps"]         # [T,B,Ncells,H,W]

        if pressure_maps.ndim != 5:
            raise RuntimeError(f"{ep_path}: pressure_maps must be [T,B,Ncells,H,W], got {pressure_maps.shape}")

        T, B = pressure_maps.shape[0], pressure_maps.shape[1]
        if len(pose_names) != B:
            raise RuntimeError(f"{ep_path}: pose_names length {len(pose_names)} != B {B}")

        print(f"[INFO] {ep_path.name}: T={T}, B={B}")

        # Track duplicates for this run (pose,timestamp collisions)
        dup_counter = {}

        for t in range(T):
            ts = float(timestamps[t])
            ts_token = _format_timestamp(ts, args.ts_decimals)

            for env in range(B):
                pose = pose_names[env]
                pose_token = _safe_slug(str(pose))

                base_name = f"posename_{pose_token}_timestamp_{ts_token}.npz"
                out_path = out_dir / base_name

                if out_path.exists() and not args.overwrite:
                    if args.dedupe:
                        key = base_name
                        n = dup_counter.get(key, 0) + 1
                        dup_counter[key] = n
                        out_path = out_dir / f"posename_{pose_token}_timestamp_{ts_token}_dup{n}.npz"
                    else:
                        raise SystemExit(
                            f"[ERROR] Collision: {out_path} already exists. "
                            f"Use --overwrite or --dedupe."
                        )

                payload = {
                    "timestamp": np.float32(ts),
                    "pose_name": np.asarray(pose),
                    "body_names": np.asarray(body_names),
                    "smpl_joint_order": np.asarray(smpl_joint_order),
                    "pressure_maps": pressure_maps[t, env].astype(np.float32),  # [Ncells,H,W]
                }

                for k in optional_keys:
                    if k in ep:
                        payload[k] = ep[k][t, env].astype(np.float32)

                if args.dry_run:
                    print(f"  [DRY] write {out_path.name}")
                else:
                    np.savez_compressed(out_path, **payload)

                written += 1

    print(f"[INFO] Done. Wrote {written} files.")


if __name__ == "__main__":
    main()
