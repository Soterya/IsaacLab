import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


class PressureMapper:
    """
    Precompute triangle sets touching the top cells and compute area-weighted pressure quickly.
    """    
    # -----------------------
    # 0)  
    # -----------------------
    
    # --- new method (with additional condition for boundary surface)--- 
    @staticmethod
    def precompute_top_surface_boundary_triangle_arrays(
        valid_tetrahedrals: np.ndarray,            # (Ne,4)
        top_index_grid: np.ndarray                 # (H,W) indices of top-plane vertices
    ):
        H, W = top_index_grid.shape
        Hm, Wm = H - 1, W - 1
        idx_to_rc = {int(top_index_grid[r, c]): (r, c) for r in range(H) for c in range(W)}
        top_nodes = set(int(v) for v in top_index_grid.ravel())
        faces_local = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int64)
        # --- build all faces and count occurrences ---
        # store: key=(i,j,k) sorted, value=(count, example_ei, example_face_nodes_original_order)
        face_count = {}
        for ei, tet in enumerate(valid_tetrahedrals):
            for f in faces_local:
                tri = tet[f]
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                key = tuple(sorted((a, b, c)))
                if key in face_count:
                    face_count[key][0] += 1
                else:
                    face_count[key] = [1, ei, (a, b, c)]  # keep a representative orientation
        triangle_nodes = []
        triangle_elements = []
        cell_ids = []
        # --- keep only boundary faces (count==1) that lie on top node set ---
        for key, (cnt, ei, tri) in face_count.items():
            if cnt != 1:
                continue  # internal face
            a, b, c = tri
            if (a in top_nodes) and (b in top_nodes) and (c in top_nodes):
                # map to grid cell: use min row/col of its vertices (same as your logic)
                rs, cs = [], []
                for v in (a, b, c):
                    r, col = idx_to_rc[v]
                    rs.append(r); cs.append(col)
                r0 = min(rs); c0 = min(cs)
                if r0 < Hm and c0 < Wm:
                    triangle_nodes.append([a, b, c])
                    triangle_elements.append(ei)
                    cell_ids.append(r0 * Wm + c0)

        if not triangle_nodes:
            return (np.zeros((0,3), np.int64),
                    np.zeros((0,),  np.int64),
                    np.zeros((0,),  np.int64),
                    Hm, Wm)

        return (np.asarray(triangle_nodes, dtype=np.int64),
                np.asarray(triangle_elements, dtype=np.int64),
                np.asarray(cell_ids, dtype=np.int64),
                Hm, Wm)

    # --- old method (check above for new version)--- 
    @staticmethod
    def precompute_top_surface_triangle_arrays(valid_tetrahedrals: np.ndarray, simulation_grid_top_surface_idxs: np.ndarray):
        """"""
        H, W = simulation_grid_top_surface_idxs.shape
        Hm, Wm = H - 1, W - 1
        
        surface_idx_to_row_col_map  = {int(simulation_grid_top_surface_idxs[r, c]): (r, c) for r in range(H) for c in range(W)}
        top_surface_node_idxs       = set(int(v) for v in simulation_grid_top_surface_idxs.ravel().tolist())
        local_faces_permutations    = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int64)
        triangle_nodes      = []
        triangle_elements   = []
        cell_ids            = []
        for ei, tetrahedral in enumerate(valid_tetrahedrals):
            for f in local_faces_permutations:
                triangle_face = tetrahedral[f]
                a, b, d = int(triangle_face[0]), int(triangle_face[1]), int(triangle_face[2])
                if (a in top_surface_node_idxs) and (b in top_surface_node_idxs) and (d in top_surface_node_idxs): # checks if a given triangle face's nodes exists in the top surface node indexes  
                    rs, cs = [], []
                    for v in (a, b, d):
                        r, c = surface_idx_to_row_col_map[v]
                        rs.append(r); cs.append(c)
                    r0 = min(rs); c0 = min(cs)
                    if r0 < Hm and c0 < Wm:
                        triangle_nodes.append([a, b, d])
                        triangle_elements.append(ei)
                        cell_ids.append(r0 * Wm + c0)
        if not triangle_nodes:
            return (np.zeros((0,3), np.int64),
                    np.zeros((0,),  np.int64),
                    np.zeros((0,),  np.int64),
                    Hm, Wm)

        return (np.asarray(triangle_nodes, dtype=np.int64),
                np.asarray(triangle_elements, dtype=np.int64),
                np.asarray(cell_ids, dtype=np.int64),
                Hm, Wm)
    
    # NOTE: This approach to finding contact pressure is not proper. 
    # TODO: Find practical methods that will convert tetrahedral von misses stresses to accurate contact maps. 
    @staticmethod 
    def compute_pressure_grid(
        top_triangle_nodes: torch.Tensor,
        top_triangle_elements: torch.Tensor,
        cell_ids: torch.Tensor,
        valid_simulation_element_stresses: torch.Tensor,   # (Ne_valid,3,3)
        simulation_nodal_positions: torch.Tensor,   # (Nv,3)
        Hm: int,
        Wm: int
    ) -> torch.Tensor:
        """
        generates pressure map based on top surface tetrahedrals
        """        
        if top_triangle_nodes.numel() == 0:
            raise RuntimeError(f"Top Triangle Nodes not Detected... Check the precompute_top_surface_triangle_arrays() function")

        v0 = simulation_nodal_positions.index_select(0, top_triangle_nodes[:, 0])
        v1 = simulation_nodal_positions.index_select(0, top_triangle_nodes[:, 1])
        v2 = simulation_nodal_positions.index_select(0, top_triangle_nodes[:, 2])

        n = torch.cross(v1 - v0, v2 - v0, dim=1)
        area = 0.5 * torch.linalg.norm(n, dim=1)
        safe = area > 1e-12
        if not torch.any(safe):
            return torch.zeros((Hm, Wm), dtype=simulation_nodal_positions.dtype, device=simulation_nodal_positions.device)

        n = n[safe]
        area = area[safe]
        tri_elem = top_triangle_elements[safe]
        cell_ids = cell_ids[safe]

        n_unit = n / (2.0 * area).unsqueeze(1)
        flip = n_unit[:, 2] < 0
        n_unit[flip] = -n_unit[flip]

        sigma = valid_simulation_element_stresses.index_select(0, tri_elem)
        p = torch.einsum('bi,bij,bj->b', n_unit, sigma, n_unit)
        p = torch.clamp(-p, min=0.0)

        num = torch.zeros(Hm * Wm, dtype=simulation_nodal_positions.dtype, device=simulation_nodal_positions.device)
        den = torch.zeros(Hm * Wm, dtype=simulation_nodal_positions.dtype, device=simulation_nodal_positions.device)
        num.scatter_add_(0, cell_ids, p * area)
        den.scatter_add_(0, cell_ids, area)
        den = torch.where(den > 0, den, torch.ones_like(den))
        pressure_grid = (num / den).reshape(Hm, Wm)
        
        return pressure_grid
    
    # @staticmethod
    # def compute_pressure_grid_torch(
    #     tri_nodes_t: torch.Tensor,
    #     tri_elem_t:  torch.Tensor,
    #     cell_ids_t:  torch.Tensor,
    #     S_valid_t:   torch.Tensor,   # (Ne_valid,3,3)
    #     sim_pos_t:   torch.Tensor,   # (Nv,3)
    #     Hm: int, Wm: int
    # ) -> torch.Tensor:
    #     if tri_nodes_t.numel() == 0:
    #         return torch.zeros((Hm, Wm), dtype=sim_pos_t.dtype, device=sim_pos_t.device)

    #     v0 = sim_pos_t.index_select(0, tri_nodes_t[:, 0])
    #     v1 = sim_pos_t.index_select(0, tri_nodes_t[:, 1])
    #     v2 = sim_pos_t.index_select(0, tri_nodes_t[:, 2])

    #     n = torch.cross(v1 - v0, v2 - v0, dim=1)
    #     area = 0.5 * torch.linalg.norm(n, dim=1)
    #     safe = area > 1e-12
    #     if not torch.any(safe):
    #         return torch.zeros((Hm, Wm), dtype=sim_pos_t.dtype, device=sim_pos_t.device)

    #     n = n[safe]
    #     area = area[safe]
    #     tri_elem = tri_elem_t[safe]
    #     cell_ids = cell_ids_t[safe]

    #     n_unit = n / (2.0 * area).unsqueeze(1)
    #     flip = n_unit[:, 2] < 0
    #     n_unit[flip] = -n_unit[flip]

    #     sigma = S_valid_t.index_select(0, tri_elem)
    #     p = torch.einsum('bi,bij,bj->b', n_unit, sigma, n_unit)
    #     p = torch.clamp(-p, min=0.0)

    #     num = torch.zeros(Hm * Wm, dtype=sim_pos_t.dtype, device=sim_pos_t.device)
    #     den = torch.zeros(Hm * Wm, dtype=sim_pos_t.dtype, device=sim_pos_t.device)
    #     num.scatter_add_(0, cell_ids, p * area)
    #     den.scatter_add_(0, cell_ids, area)
    #     den = torch.where(den > 0, den, torch.ones_like(den))
    #     pressure = (num / den).reshape(Hm, Wm)
        
    #     return pressure

    # ------------------------------------------------------------------------------------
    # 2) Actuator Aware Pressure Map (based on Indentation Map) using Simulation Nodes --- 
    # ------------------------------------------------------------------------------------
    # NOTE: for single environment. 
    # @staticmethod
    # def precompute_top_surface_indentation_cache(
    #     simulation_grid_top_surface_idxs_np: np.ndarray, 
    #     sim_default_nodal_positions: torch.Tensor, 
    #     plate_pos_w: torch.tensor, 
    #     env_id: int = 0
    # ):
    #     H, W = simulation_grid_top_surface_idxs_np.shape
    #     simulation_grid_top_surface_idxs_torch = torch.as_tensor(simulation_grid_top_surface_idxs_np.reshape(-1), device=sim_default_nodal_positions.device, dtype=torch.long)
    #     z_rest = sim_default_nodal_positions.index_select(0, simulation_grid_top_surface_idxs_torch)[:, 2]
    #     z_plate_rest = plate_pos_w[env_id, 2].to(sim_default_nodal_positions.device)  # scalar

    #     return {
    #         "simulation_grid_top_surface_idxs_torch": simulation_grid_top_surface_idxs_torch, 
    #         "H": H,
    #         "W": W,
    #         "z_rest": z_rest, 
    #         "z_plate_rest": z_plate_rest
    #     }
    
    @staticmethod
    def precompute_top_surface_indentation_cache(
        simulation_grid_top_surface_idxs_np: np.ndarray,
        sim_default_nodal_positions_w: torch.Tensor,  # (B,Nv,3)
        plate_pos_w: torch.Tensor,                   # (B,3)
    ):
        H, W = simulation_grid_top_surface_idxs_np.shape
        top_ids = torch.as_tensor(
            simulation_grid_top_surface_idxs_np.reshape(-1),
            device=sim_default_nodal_positions_w.device,
            dtype=torch.long,
        )

        # baseline z for each env (B, HW)
        z_rest = sim_default_nodal_positions_w[:, top_ids, 2]
        z_plate_rest = plate_pos_w[:, 2]                             # (B,)
        z_rel_rest = z_rest - z_plate_rest[:, None]                  # (B, HW)

        # optional: compute footprint from env0 (same for all envs)
        xy0 = sim_default_nodal_positions_w[0, top_ids, :2]           # (HW,2)
        Lx = (xy0[:, 0].max() - xy0[:, 0].min()).item()
        Ly = (xy0[:, 1].max() - xy0[:, 1].min()).item()

        return {
            "top_ids": top_ids,
            "H": H,
            "W": W,
            "z_rel_rest": z_rel_rest,   # (B,HW)
            "Lx": Lx,
            "Ly": Ly,
        }
        
    
    # @staticmethod
    # def compute_indentation_grid(
    #     indentation_cache: dict, 
    #     sim_nodal_pos_w: torch.Tensor, 
    #     plate_pos_w: torch.tensor, 
    #     env_id: int = 0, 
    #     out: str = "cell",
    #     reduce: str = "mean", 
    #     clamp_max: float|None = None
    # ) -> torch.Tensor:

    #     top_surface_node_ids = indentation_cache["simulation_grid_top_surface_idxs_torch"]
    #     H = int(indentation_cache["H"])
    #     W = int(indentation_cache["W"])
    #     z_rest = indentation_cache["z_rest"]
    #     z_plate_rest = indentation_cache["z_plate_rest"]

    #     z_now = sim_nodal_pos_w.index_select(0, top_surface_node_ids)[:, 2].to(z_rest.dtype)  # (H*W,)
    #     z_plate_now = plate_pos_w[env_id, 2].to(z_rest.dtype)

    #     # plate-relative indentation
    #     z_rel_rest = z_rest - z_plate_rest
    #     z_rel_now  = z_now  - z_plate_now
    #     indent = torch.clamp(z_rel_rest - z_rel_now, min=0.0)  # (H*W,)

    #     if clamp_max is not None:
    #         indent = torch.clamp(indent, max=float(clamp_max))

    #     # node grid
    #     indent_node = indent.reshape(H, W)

    #     if out == "node":
    #         return indent_node

    #     # cell grid (Hm,Wm) from 4 corners of each quad:
    #     # corners: (r,c), (r+1,c), (r,c+1), (r+1,c+1)
    #     if out != "cell":
    #         raise ValueError(f"out must be 'node' or 'cell', got {out}")
        
    #     a = indent_node[:-1, :-1]
    #     b = indent_node[1:,  :-1]
    #     c = indent_node[:-1, 1:]
    #     d = indent_node[1:,  1:]

    #     if reduce == "mean":
    #         indent_cell = (a + b + c + d) * 0.25
    #     elif reduce == "max":
    #         indent_cell = torch.maximum(torch.maximum(a, b), torch.maximum(c, d))
    #     else:
    #         raise ValueError(f"reduce must be 'mean' or 'max', got {reduce}")

    #     # (Hm,Wm)
    #     return indent_cell
    @staticmethod
    def compute_indentation_grid(
        indentation_cache: dict,
        sim_nodal_pos_w: torch.Tensor,   # (B,Nv,3)
        plate_pos_w: torch.Tensor,       # (B,3)
        out: str = "cell",
        reduce: str = "mean",
        clamp_max: float | None = None,
    ) -> torch.Tensor:
        top_ids = indentation_cache["top_ids"]     # (HW,)
        H = int(indentation_cache["H"])
        W = int(indentation_cache["W"])
        z_rel_rest = indentation_cache["z_rel_rest"]  # (B,HW)

        # current z for each env (B, HW)
        z_now = sim_nodal_pos_w[:, top_ids, 2]
        z_plate_now = plate_pos_w[:, 2]                 # (B,)
        z_rel_now = z_now - z_plate_now[:, None]        # (B,HW)

        indent = torch.clamp(z_rel_rest - z_rel_now, min=0.0)  # (B,HW)
        if clamp_max is not None:
            indent = torch.clamp(indent, max=float(clamp_max))

        indent_node = indent.reshape(indent.shape[0], H, W)    # (B,H,W)

        if out == "node":
            return indent_node
        if out != "cell":
            raise ValueError(f"out must be 'node' or 'cell', got {out}")

        a = indent_node[:, :-1, :-1]
        b = indent_node[:, 1:,  :-1]
        c = indent_node[:, :-1,  1:]
        d = indent_node[:, 1:,   1:]

        if reduce == "mean":
            return (a + b + c + d) * 0.25     # (B,Hm,Wm)
        if reduce == "max":
            return torch.maximum(torch.maximum(a, b), torch.maximum(c, d))
        raise ValueError(f"reduce must be 'mean' or 'max', got {reduce}")


    
    @staticmethod
    def indentation_to_pressure_grid(
            indent_cell: torch.Tensor, 
            k_area: float, 
            n: float = 1, 
            clamp_max_pa: float | None = None,
            eps: float = 1e-12
    ) -> torch.Tensor:
        """
        convert indentation (m) to pressure (Pa) using p = k_area * indent^n. 
        """ 
        indent_cell_clamped = torch.clamp(indent_cell, min=0.0)
        if n == 1: 
            pressure = indent_cell_clamped * float(k_area) 
        else: 
            pressure = (indent_cell_clamped + eps).pow(float(n)) * float(k_area)

        if clamp_max_pa is not None:
            pressure = torch.clamp(pressure, max=float(clamp_max_pa))
        return pressure
    
    @staticmethod
    def normalize_pressure_to_total_force(
        pressure_pa: torch.Tensor,     # (Hm,Wm)
        cell_area_m2: float,           # scalar area per cell (approx)
        target_force_n: float,         # e.g. m*g
        eps: float = 1e-12
    ) -> torch.Tensor:
        """
        Scale pressure map so that sum(p*A) matches target_force_n.
        This is a simple calibration step that preserves the shape of the map.
        """
        total_force = (pressure_pa * float(cell_area_m2)).sum()
        scale = float(target_force_n) / float(total_force + eps)
        return pressure_pa * scale
    


        
