import os
import glob
import re
from typing import Optional, Dict, List
import numpy as np

from utils.rotation_utils import rotvec_to_euler_xyz


def npz_index(path: str) -> int:
    """
    Extract trailing numeric index from filenames like 0000.npz, 12.npz, pose_0010.npz.
    Returns -1 if no number is found.
    """
    base = os.path.basename(path)
    m = re.search(r"(\d+)(?=\.npz$)", base)
    return int(m.group(1)) if m else -1

def list_npz_files(npz_dir: str, pattern: str = "*.npz",
                   start: Optional[int] = None, end: Optional[int] = None) -> list[str]:
    paths = glob.glob(os.path.join(npz_dir, pattern))
    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort(key=npz_index)
    if start is not None:
        paths = [p for p in paths if npz_index(p) >= start]
    if end is not None:
        paths = [p for p in paths if npz_index(p) <= end]
    return paths

def build_joint_config_from_npz(path: str, SMPL_TO_ISAAC:dict[str, tuple[str, str, str]], SMPL_BODY_JOINT_ORDER:list[str]) -> dict[str, float]:
    """
    Builds Joint Configuration (for Implicit Actuator ) given a .npz smpl pose file.
    Produces Euler XYZ (radians) per Isaac hinge triplet. 
    """    
    data = np.load(path, allow_pickle=True)
    body = data["pose_body"][0].astype(np.float32).reshape(-1, 3)

    joint_cfg: dict[str, float] = {}
    for smpl_name, isaac_triplet in SMPL_TO_ISAAC.items():
        smpl_idx = SMPL_BODY_JOINT_ORDER.index(smpl_name)  # 0..22
        rotvec = body[smpl_idx]  # (3,)
        rx, ry, rz = rotvec_to_euler_xyz(rotvec, degrees=False)  # radians
        jx, jy, jz = isaac_triplet
        joint_cfg[jx] = float(rx)
        joint_cfg[jy] = float(ry)
        joint_cfg[jz] = float(rz)
    return joint_cfg