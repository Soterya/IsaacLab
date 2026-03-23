from scipy.spatial.transform import Rotation as R
import numpy as np

def rotvec_to_euler_xyz(rotvec: np.ndarray, degrees: bool=False) -> np.ndarray:
    """
    Converts from axis-angle form to euler xyz (either in degrees or radians)
    """
    r = R.from_rotvec(np.asarray(rotvec, dtype=np.float64))
    e = r.as_euler("XYZ", degrees=degrees)
    return e.astype(np.float32)

def rotvec_to_quat_wxyz(rotvec_xyz: np.ndarray) -> np.ndarray:
    """returns quat in Isaac/your convention: (w,x,y,z)"""
    q_xyzw = R.from_rotvec(rotvec_xyz.astype(np.float64)).as_quat() 
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

def quat_wxyz_to_rotvec(wxyz: np.ndarray) -> np.ndarray:
    """Isaac gives (w,x,y,z). SciPy expects (x,y,z,w). Return rotvec float32."""
    w, x, y, z = float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])
    return R.from_quat([x, y, z, w]).as_rotvec().astype(np.float32)

def euler_xyz_to_rotvec(e_xyz: np.ndarray) -> np.ndarray:
    """Euler XYZ (radians) -> rotvec."""
    return R.from_euler("XYZ", e_xyz.astype(np.float64), degrees=False).as_rotvec().astype(np.float32)