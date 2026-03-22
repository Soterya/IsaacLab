# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleTalker(Node):
    """Minimal ROS 2 talker that publishes std_msgs/String."""

    def __init__(self):
        super().__init__("isaaclab_talker")
        self.publisher = self.create_publisher(String, "chatter", 10)
        self.timer = self.create_timer(1.0, self._publish_message)
        self.counter = 0

    def _publish_message(self):
        msg = String()
        msg.data = f"Hello from Isaac Lab ({self.counter})"
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')
        self.counter += 1

def main():
    """Main function."""
    rclpy.init()
    talker = SimpleTalker()

    try:
        # Initialize the simulation context
        sim_cfg = SimulationCfg(dt=0.01)
        sim = SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))

        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")

        # Simulate physics and process ROS callbacks.
        while simulation_app.is_running():
            # perform step
            sim.step()
            rclpy.spin_once(talker, timeout_sec=0.0)
    finally:
        talker.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
