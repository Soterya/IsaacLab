#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt

class TopSurfaceViewer(Node):
    def __init__(self):
        super().__init__("top_surface_viewer")
        self.sub = self.create_subscription(
            Float32MultiArray,
            "/foam_bed/top_surface/z_grid",
            self.cb,
            10,
        )
        self.im = None
        self.fig = None

    def cb(self, msg: Float32MultiArray):
        dims = msg.layout.dim
        if len(dims) != 2:
            self.get_logger().warn("Expected 2D layout [rows, cols]")
            return
        rows, cols = dims[0].size, dims[1].size
        data = np.array(msg.data, dtype=np.float32)
        if data.size != rows * cols:
            self.get_logger().warn(f"Size mismatch: {data.size} vs {rows*cols}")
            return
        z = data.reshape(rows, cols)

        if self.im is None:
            self.fig = plt.figure("Foam top surface heightfield")
            self.im = plt.imshow(z, origin="lower")
            plt.colorbar(label="z (m)")
            plt.tight_layout()
            plt.ion(); plt.show()
        else:
            self.im.set_data(z)
            # If range changes a lot, uncomment:
            # self.im.set_clim(vmin=z.min(), vmax=z.max())
        plt.pause(0.001)

def main():
    rclpy.init()
    node = TopSurfaceViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
