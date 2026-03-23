"""
TODO: this script works but the scale is not adjusted
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2

class TopSurfaceViewer(Node):
    def __init__(self):
        super().__init__("top_surface_viewer")

        # Params for fixed color scaling (meters)
        self.declare_parameter("vmin", 0.0)     # e.g., lowest expected z
        self.declare_parameter("vmax", 0.10)    # e.g., highest expected z
        self.declare_parameter("colormap", "JET")   # JET, TURBO, VIRIDIS (see mapping below)
        self.declare_parameter("resize", 2.0)       # display scale factor

        self.vmin = float(self.get_parameter("vmin").value)
        self.vmax = float(self.get_parameter("vmax").value)
        self.resize_factor = float(self.get_parameter("resize").value)

        cmap_name = str(self.get_parameter("colormap").value).upper()
        cmap_map = {
            "JET": cv2.COLORMAP_JET,
            "TURBO": cv2.COLORMAP_TURBO,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
            "HOT": cv2.COLORMAP_HOT,
            "MAGMA": cv2.COLORMAP_MAGMA,
            "PLASMA": cv2.COLORMAP_PLASMA,
            "INFERNO": cv2.COLORMAP_INFERNO,
        }
        self.cmap = cmap_map.get(cmap_name, cv2.COLORMAP_JET)

        if not (self.vmax > self.vmin):
            self.get_logger().warn("Invalid vmin/vmax; resetting to vmin=0.0, vmax=0.1")
            self.vmin, self.vmax = 0.0, 0.1

        self.sub = self.create_subscription(
            Float32MultiArray, "/foam_bed/top_surface/z_grid", self.cb, 10
        )
        self.window_name = "Foam top surface heightfield"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.get_logger().info(f"Fixed color scale -> vmin={self.vmin} m, vmax={self.vmax} m")

    def cb(self, msg: Float32MultiArray):
        dims = msg.layout.dim
        if len(dims) != 2:
            self.get_logger().warn("Expected 2D layout [rows, cols]")
            return
        rows, cols = dims[0].size, dims[1].size
        data = np.asarray(msg.data, dtype=np.float32)

        if data.size != rows * cols:
            self.get_logger().warn(f"Size mismatch: {data.size} vs {rows*cols}")
            return

        z = data.reshape(rows, cols)

        # Handle NaNs/Infs safely
        z = np.nan_to_num(z, nan=self.vmin, posinf=self.vmax, neginf=self.vmin)

        # Fixed-range scaling: clip -> [vmin, vmax] then map to [0,255]
        z_clipped = np.clip(z, self.vmin, self.vmax)
        z_scaled = ((z_clipped - self.vmin) / (self.vmax - self.vmin) * 255.0).astype(np.uint8)

        # Colorize
        z_color = cv2.applyColorMap(z_scaled, self.cmap)

        # Optional: enlarge for viewing
        if self.resize_factor != 1.0:
            z_color = cv2.resize(
                z_color,
                (int(cols * self.resize_factor), int(rows * self.resize_factor)),
                interpolation=cv2.INTER_NEAREST,
            )

        cv2.imshow(self.window_name, z_color)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = TopSurfaceViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
