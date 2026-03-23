from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple
from isaaclab.assets import (
    DeformableObject,
)

class GridBuilder:
    
    @staticmethod
    def build_colmesh_grid(cube: DeformableObject, z_target: torch.Tensor, decimals: int) -> torch.Tensor:
        """
        Build (H,W) grid of DEFAULT-state (collision mesh) nodal indices on the z_target plane.
        Returns:
          index_grid: torch.long (H,W) containing indices into default nodal arrays.
        """
        top_idx = GridBuilder.get_surface_indices_by_z(cube, z_target, decimals)
        if top_idx.numel() == 0:
            raise RuntimeError("[colmesh] No nodes found on target Z plane.")

        default_pos = cube.data.default_nodal_state_w[..., :3][0]
        top_pos = default_pos.index_select(0, top_idx)

        s = float(10 ** decimals)
        x = top_pos[:, 0]
        y = top_pos[:, 1]
        x0 = x.min()
        y0 = y.min()

        # integer bins in "bin-space"
        x_bin = torch.round((x - x0) * s).to(torch.long)
        y_bin = torch.round((y - y0) * s).to(torch.long)

        # unique sorted bins + inverse mapping to [0..W-1], [0..H-1]
        xs, x_inv = torch.unique(x_bin, sorted=True, return_inverse=True)
        ys, y_inv = torch.unique(y_bin, sorted=True, return_inverse=True)
        W = int(xs.numel())
        H = int(ys.numel())

        index_grid = torch.full((H, W), -1, dtype=torch.long, device=top_idx.device)
        used = torch.zeros((H, W), dtype=torch.bool, device=top_idx.device)

        # Choose a representative per (r,c): prefer points closest to exact bin center in bin-space.
        # This avoids any float reconstruction like (x0 + x_bin/s).
        dx = torch.abs((x - x0) * s - x_bin.to(x.dtype))
        dy = torch.abs((y - y0) * s - y_bin.to(y.dtype))
        order = torch.argsort(dx + dy)

        # Fill grid (one node per bin cell)
        for k in order.tolist():
            r = int(y_inv[k].item())
            c = int(x_inv[k].item())
            if not used[r, c]:
                index_grid[r, c] = top_idx[k]
                used[r, c] = True

        holes = int((index_grid < 0).sum().item())
        if holes:
            raise AssertionError(f"[colmesh] grid has {holes} holes; try decimals=1 or 3.")

        return index_grid

    # NOTE: This is the numpy version, that works with non interactive environments 
    @staticmethod
    def build_sim_grid(cube: DeformableObject, decimals: int) -> np.ndarray:
        """
        Build (H,W) grid of SIM nodal indices on the *top surface* using SIM positions.
        Returns:
          index_grid: (H,W) int64 indices into sim nodal arrays
          sim_pos:    (N,3) float64 positions used (CPU)
        """
        # sim_pos_any = cube.root_physx_view.get_sim_nodal_positions()[0] # TODO: this type of extraction might be useful later while working with collision data. 
        sim_pos_any = cube.data.nodal_pos_w[0]
        sim_pos     = sim_pos_any.detach().cpu().numpy() if hasattr(sim_pos_any, "detach") else np.asarray(sim_pos_any)
        sim_pos     = sim_pos.astype(np.float64, copy=False)
        s = float(10 ** decimals)
        z = sim_pos[:, 2]
        z_bin = np.rint((z - z.min()) * s).astype(np.int64)
        top_bin = int(z_bin.max())
        top_idx = np.nonzero(z_bin == top_bin)[0]
        if top_idx.size == 0:
            raise RuntimeError("No SIM top nodes found.")
        top_pos = sim_pos[top_idx]
        x = top_pos[:, 0]
        y = top_pos[:, 1]
        x0 = float(x.min())
        y0 = float(y.min())
        x_bin = np.rint((x - x0) * s).astype(np.int64)
        y_bin = np.rint((y - y0) * s).astype(np.int64)
        xs, x_inv = np.unique(x_bin, return_inverse=True)
        ys, y_inv = np.unique(y_bin, return_inverse=True)
        W = int(xs.size)
        H = int(ys.size)
        index_grid = -np.ones((H, W), dtype=np.int64)
        dx = np.abs((x - x0) * s - x_bin.astype(np.float64))
        dy = np.abs((y - y0) * s - y_bin.astype(np.float64))
        score = dx + dy
        for k in np.argsort(score):
            r = int(y_inv[k])
            c = int(x_inv[k])
            if index_grid[r, c] < 0:
                index_grid[r, c] = int(top_idx[k])
        holes = int((index_grid < 0).sum())
        if holes:
            raise RuntimeError(f"[sim] grid has {holes} holes; try decimals=1 or 3.")
        return index_grid

    @staticmethod
    def get_surface_indices_by_z(cube: DeformableObject, z_target: torch.Tensor, decimals: int) -> torch.Tensor:
        """
        Return indices of nodes whose DEFAULT-state z lies on the plane nearest z_target,
        using integer bins (robust on GPU).
        """
        # extract z from all nodal postion of a cube. 
        default_nodal_pos   = cube.data.default_nodal_state_w[..., :3][0]
        default_nodal_pos_z = default_nodal_pos[:, 2]
        s = float(10 ** decimals)        
        # find indices of all target z by comparing nodal positions with target z values
        z_bin   = torch.round(default_nodal_pos_z * s).to(torch.long)
        tgt_bin = (z_target * s).to(torch.long) 
        idx = torch.nonzero(z_bin == tgt_bin, as_tuple=False).squeeze(1)
        # if z target is not found, then get the nearest plane to it
        if idx.numel() > 0:
            return idx
        planes = torch.unique(z_bin)
        nearest = planes[torch.argmin(torch.abs(planes - tgt_bin))]
        return torch.nonzero(z_bin == nearest, as_tuple=False).squeeze(1)

    @staticmethod
    def corners_from_surface_idx(cube: DeformableObject, surface_idx: torch.Tensor, env_id: int = 0) -> torch.Tensor:
        """
        Pick 4 unique corner nodes (BL, TL, BR, TR) from a set of surface indices
        by nearest XY distance to (xmin/ymin etc.). Robust and simple.
        """
        default_pos = cube.data.default_nodal_state_w[env_id, :, :3]
        pts = default_pos.index_select(0, surface_idx)
        xy = pts[:, :2]
        x = xy[:, 0]
        y = xy[:, 1]

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        targets = torch.stack(
            [
                torch.stack((xmin, ymin)),  # BL
                torch.stack((xmin, ymax)),  # TL
                torch.stack((xmax, ymin)),  # BR
                torch.stack((xmax, ymax)),  # TR
            ],
            dim=0,
        )  # (4,2)

        diffs = xy[:, None, :] - targets[None, :, :]
        dists = (diffs * diffs).sum(dim=2)  # (Ns,4)

        chosen = []
        for j in range(4):
            order = torch.argsort(dists[:, j]).tolist()
            pick = next((cand for cand in order if cand not in chosen), order[0])
            chosen.append(pick)

        chosen_t = torch.as_tensor(chosen, dtype=torch.long, device=surface_idx.device)
        return surface_idx.index_select(0, chosen_t)

