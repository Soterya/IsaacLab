import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import (
    DeformableObject, DeformableObjectCfg,
    RigidObject, RigidObjectCfg,
    Articulation, ArticulationCfg
)
from isaaclab.actuators import ImplicitActuatorCfg


class ROSBedController:
    """Manages ROS commands for bed joints (named or grid) and clamps to limits."""
    def __init__(self, ros_node: Node, rows: int, cols: int):
        self.ros_node = ros_node
        self.rows = rows
        self.cols = cols
        self.name_to_id: Dict[str, int] = {}
        self.pris_ids: List[int] = []
        self.q_cmd = None
        self.q_min = None
        self.q_max = None

        class _Buf: pass
        self.buf = _Buf()
        self.buf.val = None# type: ignore
        self.buf.mask = None# type: ignore

        self.ros_node.create_subscription(JointState,        "/korusbed/joint_pos_target",     self._cb_named, 10)
        self.ros_node.create_subscription(Float32MultiArray, "/korusbed/cell_grid/pos_target", self._cb_grid,  10)

    # attach once after we have the bed
    def attach_bed(self, bed: Articulation):
        try:
            jnames = getattr(bed.data, "joint_names", None) or []
            self.name_to_id = {nm: i for i, nm in enumerate(jnames) if isinstance(nm, str)}
            self.pris_ids = [i for i, nm in enumerate(jnames) if isinstance(nm, str) and nm.startswith("PrismaticJoint")]
        except Exception:
            self.name_to_id = {}; self.pris_ids = []

        B_bed, DoF = bed.data.joint_pos.shape
        lo = hi = None
        try:
            if hasattr(bed.data, "joint_pos_limits"):
                lims = bed.data.joint_pos_limits
                if lims.ndim == 3 and lims.shape[-1] == 2: lo, hi = lims[..., 0], lims[..., 1]
            elif hasattr(bed.data, "joint_limits"):
                lims = bed.data.joint_limits
                if lims.ndim == 3 and lims.shape[-1] == 2: lo, hi = lims[..., 0], lims[..., 1]
        except Exception:
            lo = hi = None
        if lo is None or hi is None:
            lo = torch.full((B_bed, DoF), -0.05, device=bed.data.joint_pos.device)
            hi = torch.full((B_bed, DoF),  +0.05, device=bed.data.joint_pos.device)

        q_now = bed.data.joint_pos.clone()
        self.q_cmd = q_now.clone()
        self.q_min = lo.clone()
        self.q_max = hi.clone()
        if self.pris_ids:
            pid = torch.tensor(self.pris_ids, device=q_now.device, dtype=torch.long)
            self.q_cmd[:, pid] = torch.clamp(q_now[:, pid], self.q_min[:, pid], self.q_max[:, pid])

        # ROS buffer (host)
        self.buf.val  = np.array(self.q_cmd[0].detach().cpu().numpy(), dtype=float) # type: ignore
        self.buf.mask = np.zeros(DoF, dtype=bool) # type: ignore

    def apply_keyboard_delta(self, delta: float):
        if self.q_cmd is None or not self.pris_ids or delta == 0.0:
            return
        pid = torch.tensor(self.pris_ids, device=self.q_cmd.device, dtype=torch.long)
        self.q_cmd[:, pid] = torch.clamp(self.q_cmd[:, pid] + delta, self.q_min[:, pid], self.q_max[:, pid]) # type: ignore

    def apply_ros_targets(self):
        if self.q_cmd is None or self.buf.val is None: # type: ignore
            return
        idxs = np.nonzero(self.buf.mask)[0].tolist() # type: ignore
        if not idxs:
            return
        idxt = torch.tensor(idxs, device=self.q_cmd.device, dtype=torch.long)
        vals = torch.tensor(self.buf.val[idxs], device=self.q_cmd.device, dtype=self.q_cmd.dtype) # type: ignore 
        self.q_cmd[:, idxt] = torch.clamp(vals, self.q_min[:, idxt], self.q_max[:, idxt]) # type: ignore
        self.buf.mask[idxs] = False# type: ignore

    def _cb_named(self, msg: JointState):
        if self.buf.val is None or self.buf.mask is None: return # type: ignore
        for nm, pos in zip(msg.name, msg.position):
            j = self.name_to_id.get(nm)
            if j is not None and j < self.buf.val.shape[0]: # type: ignore
                self.buf.val[j]  = float(pos) # type: ignore
                self.buf.mask[j] = True # type: ignore

    def _cb_grid(self, msg: Float32MultiArray):
        if self.buf.val is None or self.buf.mask is None: return# type: ignore
        data = np.array(msg.data, dtype=float)
        if data.size == 0: return
        R, C = self.rows, self.cols
        try:
            dims = msg.layout.dim
            if len(dims) >= 2:
                R = int(dims[0].size); C = int(dims[1].size)# type: ignore
        except Exception:
            pass
        try:
            grid = data.reshape(R, C)
        except Exception:
            rc = self.rows * self.cols
            flat = np.zeros(rc, dtype=float)
            flat[: min(rc, data.size)] = data[: min(rc, data.size)]
            grid = flat.reshape(self.rows, self.cols)
        for r in range(min(self.rows, grid.shape[0])):
            for c in range(min(self.cols, grid.shape[1])):
                idx_cell = r * self.cols + c
                jname = f"PrismaticJoint{idx_cell}"
                j = self.name_to_id.get(jname)
                if j is not None and j < self.buf.val.shape[0]:# type: ignore
                    self.buf.val[j]  = float(grid[r, c])# type: ignore
                    self.buf.mask[j] = True# type: ignore