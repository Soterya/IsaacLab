# math imports
import torch
import numpy as np
# formatting imports
from typing import List
# ros2 imports
from rclpy.node import Node
from sensor_msgs.msg import JointState

class PrismaticJointTargetSubscriber:
    def __init__(self, ros_node: Node, topic="/korusbed/joint_pos_target"):
        self.ros_node = ros_node
        self.topic = topic
        self._subscriber = ros_node.create_subscription(JointState, topic, self._callback, 10)

        self._name_to_id: dict[str, int] = {}
        self._prismatic_ids: list[int] = []

        self._joint_cmd: torch.Tensor | None = None   # [B, DoF]
        self._joint_min: torch.Tensor | None = None   # [B, DoF]
        self._joint_max: torch.Tensor | None = None   # [B, DoF]

        self._latest_named: dict[str, float] = {}
        self._latest_prismatic_vector: np.ndarray | None = None
        self._has_update: bool = False

    def attach_bed(self, joint_names: List[str], joint_angles: torch.Tensor, joint_limits: torch.Tensor):
        """
        Call after you reset the bed.
        joint_angles: [B, DoF]
        joint_limits: ideally [B, DoF, 2] (lo/hi). If [DoF, 2], we broadcast to B.
        """
        self._name_to_id = {nm: i for i, nm in enumerate(joint_names) if isinstance(nm, str)}
        self._prismatic_ids = [
            i for i, nm in enumerate(joint_names)
            if isinstance(nm, str) and nm.startswith("PrismaticJoint")
        ]

        B, DoF = joint_angles.shape

        # Normalize joint_limits shape
        if joint_limits.ndim == 2 and joint_limits.shape == (DoF, 2):
            joint_limits = joint_limits.unsqueeze(0).expand(B, DoF, 2)
        elif joint_limits.ndim != 3 or joint_limits.shape[-1] != 2:
            raise ValueError(f"joint_limits must be [B,DoF,2] or [DoF,2], got {tuple(joint_limits.shape)}")

        low_limits = joint_limits[..., 0].to(device=joint_angles.device, dtype=joint_angles.dtype)
        high_limits = joint_limits[..., 1].to(device=joint_angles.device, dtype=joint_angles.dtype)

        self._joint_cmd = joint_angles.clone()
        self._joint_min = low_limits.clone()
        self._joint_max = high_limits.clone()

        self._latest_named.clear()
        self._latest_prismatic_vector = None
        self._has_update = False

        self.ros_node.get_logger().info(
            f"[PrismaticJointTargetSubscriber] Attached bed. DoF={DoF}, prismatic={len(self._prismatic_ids)}"
        )

    def _callback(self, msg: JointState):
        """Executed when a message is received."""
        if msg.name:
            for name, pos in zip(msg.name, msg.position):
                self._latest_named[name] = float(pos)
            self._latest_prismatic_vector = None
            self._has_update = True
            return

        if len(msg.position) > 0:
            self._latest_prismatic_vector = np.asarray(msg.position, dtype=np.float32)
            self._latest_named.clear()
            self._has_update = True

    def apply_latest(self):
        """
        Call each sim step (or at your --every rate).
        Updates self._joint_cmd in-place.
        """
        if (not self._has_update) or (self._joint_cmd is None) or (self._joint_min is None) or (self._joint_max is None):
            return

        joint_angles = self._joint_cmd

        # Named update
        if self._latest_named:
            idxs: list[int] = []
            vals: list[float] = []

            for name, val in self._latest_named.items():
                j = self._name_to_id.get(name)
                if j is not None:
                    idxs.append(j)
                    vals.append(val)

            if idxs:
                idxt = torch.tensor(idxs, device=joint_angles.device, dtype=torch.long)
                valt = torch.tensor(vals, device=joint_angles.device, dtype=joint_angles.dtype).view(1, -1)
                valt = valt.expand(joint_angles.shape[0], -1)  # broadcast to all envs

                joint_angles[:, idxt] = torch.clamp(
                    valt,
                    self._joint_min[:, idxt],
                    self._joint_max[:, idxt],
                )

        # Vector update (names omitted) -> apply in prismatic order
        elif self._latest_prismatic_vector is not None and self._prismatic_ids:
            v = self._latest_prismatic_vector
            n = min(len(v), len(self._prismatic_ids))
            pid = torch.tensor(self._prismatic_ids[:n], device=joint_angles.device, dtype=torch.long)
            valt = torch.tensor(v[:n], device=joint_angles.device, dtype=joint_angles.dtype).view(1, -1)
            valt = valt.expand(joint_angles.shape[0], -1)

            joint_angles[:, pid] = torch.clamp(
                valt,
                self._joint_min[:, pid],
                self._joint_max[:, pid],
            )

        self._has_update = False

    @property
    def joint_cmd(self) -> torch.Tensor | None:
        return self._joint_cmd