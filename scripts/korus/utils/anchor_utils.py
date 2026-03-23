import torch
from typing import Optional
from dataclasses import dataclass

# --- Data Structure for kinematic anchoring of deformables --- 
TOPPLATE_SIZE = (1.0, 1.0, 0.1)
@dataclass
class AnchorCache:
    bottom_corners_idxs: torch.Tensor          # (4,) node indices (same for all envs)
    kinematic_targets: torch.Tensor                     # (B, N, 4)
    corners_buffer: torch.Tensor             # (B, 4, 3)
    last_plate_position: Optional[torch.Tensor] = None  # (B, 3)

def plate_has_moved_enough(last: Optional[torch.Tensor], current: torch.Tensor, epsilon: float) -> bool:
    """returns true if the top plate has moved enough from the previous position"""
    if (last == None):
        return True
    return bool(torch.max(torch.abs(current - last)).item() > epsilon) 

def compute_plate_corners(plate_pos: torch.Tensor, half_x:float=TOPPLATE_SIZE[0]*0.5, half_y:float=TOPPLATE_SIZE[1]*0.5, half_z:float=TOPPLATE_SIZE[2]*0.5) -> torch.Tensor: 
    """
    plate_pos: (B,3) center of the plate; B -> num_envs
    returns corner positions: (B,4,3) in order [BL, TL, BR, TR] on the TOP SURFACE OF THE PLATE
    """
    B = plate_pos.shape[0]
    corners = plate_pos.unsqueeze(1).repeat(1,4,1) # (B,4,3)
    corners[:, :, 2] = corners[:, :, 2] + half_z
    corners[:, 0, 0] -= half_x
    corners[:, 0, 1] -= half_y
    corners[:, 1, 0] -= half_x
    corners[:, 1, 1] += half_y
    corners[:, 2, 0] += half_x
    corners[:, 2, 1] -= half_y
    corners[:, 3, 0] += half_x
    corners[:, 3, 1] += half_y
    return corners