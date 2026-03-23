# utils/resume_manager.py
"""
Resume / override manager for Korus dataset generation.

Usage (in your main script):
    from utils.resume_manager import EpisodeResumeManager

    mgr = EpisodeResumeManager(
        record_root=args_cli.record_dir,
        in_pose_dir=args_cli.in_pose_dir,
        override=args_cli.override,
        expected_B=scene.num_envs,
        mirror_after="data_npz",  # optional
        verbose=True,
    )

    record_dir = mgr.record_dir
    start_ep   = mgr.resume_episode_idx

    # then set:
    episode_idx = start_ep
    current_episode = start_ep

    # before saving:
    mgr.ensure_ready_dir()

    # to wipe:
    # mgr.override = True; mgr.prepare()  # or just pass override=True
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ResumeScanResult:
    completed: Dict[int, int]                 # {episode_idx: B}
    corrupt: List[Tuple[str, str]]            # [(path, reason)]


class EpisodeResumeManager:
    """
    Manages:
      - resolving the final output directory (optionally mirroring after a path segment)
      - scanning existing episode files
      - validating B matches expected_B
      - choosing resume episode index (first missing from 0)
      - override cleanup (delete existing episode files)
    """

    _EP_RE = re.compile(r"^episode_(\d+)_B(\d+)_gt\.npz$")

    def __init__(
        self,
        record_root: str,
        in_pose_dir: str,
        override: bool,
        expected_B: int,
        mirror_after: Optional[str] = "data_npz",
        verbose: bool = True,
    ):
        self.record_root = str(record_root)
        self.in_pose_dir = str(in_pose_dir)
        self.override = bool(override)
        self.expected_B = int(expected_B)
        self.mirror_after = mirror_after
        self.verbose = bool(verbose)

        self.record_dir: str = self.resolve_record_dir()
        self.resume_episode_idx: int = 0
        self.scan_result: Optional[ResumeScanResult] = None

        # Compute resume state immediately (safe to call again via prepare()).
        self.prepare()

    # -----------------------------
    # Public API
    # -----------------------------
    def prepare(self) -> None:
        """Compute output dir and resume index; apply override if requested."""
        self.record_dir = self.resolve_record_dir()

        if self.override:
            deleted = self.cleanup_output_episodes()
            if self.verbose:
                print(
                    f"[INFO] EpisodeResumeManager: --override set; deleted {deleted} episode file(s) in: {self.record_dir}"
                )
            self.resume_episode_idx = 0
            self.scan_result = ResumeScanResult(completed={}, corrupt=[])
            return

        self.resume_episode_idx = self.detect_resume_state(expected_B=self.expected_B)

        if self.verbose:
            if self.resume_episode_idx > 0:
                print(
                    f"[INFO] EpisodeResumeManager: Resuming from episode {self.resume_episode_idx:04d} in: {self.record_dir}"
                )
            else:
                print(f"[INFO] EpisodeResumeManager: No prior progress found; starting at episode 0000 in: {self.record_dir}")

    def ensure_ready_dir(self) -> None:
        """Create output directory if needed."""
        os.makedirs(self.record_dir, exist_ok=True)

    def is_done(self, total_episodes: int) -> bool:
        """Returns True if resume index indicates all episodes already exist."""
        return int(self.resume_episode_idx) >= int(total_episodes)

    # -----------------------------
    # Core logic
    # -----------------------------
    def resolve_record_dir(self) -> str:
        """
        Mirror the folder structure after a given path segment (default: 'data_npz').

        Example:
          in_pose_dir  = .../scripts/korus/data/data_npz/general_supine/foo
          record_root  = .../scripts/korus/korus_pressure_pose_dataset
          -> record_dir = .../scripts/korus/korus_pressure_pose_dataset/general_supine/foo

        If mirror_after is None or not found in in_pose_dir, returns record_root unchanged.
        If record_root already ends with the derived suffix, avoids duplication.
        """
        rec = Path(self.record_root)
        inp = Path(self.in_pose_dir)

        if not self.mirror_after:
            return str(rec)

        parts = inp.parts
        if self.mirror_after not in parts:
            return str(rec)

        idx = parts.index(self.mirror_after)
        suffix = Path(*parts[idx + 1 :])  # e.g. general_supine/foo

        # Avoid double-appending if user already passed the full target folder.
        try:
            rec_res = rec.resolve()
            target = (rec / suffix).resolve()
            if str(rec_res).endswith(str(suffix)) or rec_res == target:
                return str(rec_res)
            return str(target)
        except Exception:
            # resolve() can fail if path doesn't exist; fallback
            if str(rec).endswith(str(suffix)):
                return str(rec)
            return str(rec / suffix)

    def scan_completed_episodes(self) -> ResumeScanResult:
        """
        Scan record_dir for episode_####_B{B}_gt.npz and validate readability.

        Returns:
          completed: {episode_idx: B} where B comes from file content (pose_names length) if possible
          corrupt:   list of files that matched the filename pattern but couldn't be loaded / missing keys
        """
        completed: Dict[int, int] = {}
        corrupt: List[Tuple[str, str]] = []

        out_dir = self.record_dir
        if not os.path.isdir(out_dir):
            return ResumeScanResult(completed=completed, corrupt=corrupt)

        for fn in os.listdir(out_dir):
            m = self._EP_RE.match(fn)
            if not m:
                continue

            ep_idx = int(m.group(1))
            b_from_name = int(m.group(2))
            path = os.path.join(out_dir, fn)

            try:
                with np.load(path, allow_pickle=True) as z:
                    if "pose_names" not in z.files:
                        raise RuntimeError("missing key 'pose_names'")
                    b_from_file = int(len(z["pose_names"]))
                    B = b_from_file if b_from_file > 0 else b_from_name
            except Exception as e:
                corrupt.append((path, f"{type(e).__name__}: {e}"))
                continue

            completed[ep_idx] = B

        return ResumeScanResult(completed=completed, corrupt=corrupt)

    def detect_resume_state(self, expected_B: int) -> int:
        """
        Determine the resume episode index (first missing starting from 0).
        Raises if existing episodes have mismatched B or multiple Bs.
        """
        self.ensure_ready_dir()
        scan = self.scan_completed_episodes()
        self.scan_result = scan

        if scan.corrupt and self.verbose:
            print("[WARN] EpisodeResumeManager: Found corrupt/unreadable episode files; they will be ignored:")
            for p, r in scan.corrupt[:10]:
                print(f"  - {p}  ({r})")
            if len(scan.corrupt) > 10:
                print(f"  ... and {len(scan.corrupt) - 10} more")

        if not scan.completed:
            return 0

        Bs = set(scan.completed.values())
        if len(Bs) != 1:
            raise RuntimeError(
                f"Output dir has episodes with multiple batch sizes B={sorted(Bs)}. "
                f"Use --override or clean the folder: {self.record_dir}"
            )
        prev_B = next(iter(Bs))
        if int(prev_B) != int(expected_B):
            raise RuntimeError(
                f"Resume refused: output episodes were saved with B={prev_B}, but current run has B={expected_B}. "
                f"Either run with --num_envs {prev_B} or use --override to start from scratch.\n"
                f"Output dir: {self.record_dir}"
            )

        present = sorted(scan.completed.keys())
        return self.first_missing_nonnegative(present)

    @staticmethod
    def first_missing_nonnegative(sorted_present: List[int]) -> int:
        """
        Given sorted present episode indices, return the first missing episode index starting at 0.
        """
        want = 0
        for ep in sorted_present:
            if ep < want:
                continue
            if ep == want:
                want += 1
                continue
            # gap found
            break
        return want

    def cleanup_output_episodes(self) -> int:
        """
        Deletes episode_####_B*_gt.npz in record_dir.
        Returns number of deleted files.
        """
        out_dir = self.record_dir
        if not os.path.isdir(out_dir):
            return 0

        n = 0
        for fn in os.listdir(out_dir):
            if self._EP_RE.match(fn):
                try:
                    os.remove(os.path.join(out_dir, fn))
                    n += 1
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] EpisodeResumeManager: Failed to delete {fn}: {e}")
        return n
