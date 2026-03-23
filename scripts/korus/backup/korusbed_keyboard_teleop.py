# korusbed_keyboard_teleop_curses.py
#
# Type digits to select a joint index, then Up/Down (or W/S) to move it.
# Publishes absolute JointState targets to /korusbed/joint_pos_target.

import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import curses

@dataclass
class Params:
    topic: str = "/korusbed/joint_pos_target"
    step: float = 0.003      # per keypress increment
    initial: float = 0.70    # first-time value for a touched joint
    min_pos: float = 0.0     # soft clamp (sim still clamps)
    max_pos: float = 1.0
    rate_hz: float = 120.0

HELP = [
    "KorusBed Keyboard Teleop (curses)",
    "---------------------------------",
    "Type digits to choose a joint index, then use Up/Down (or W/S) to move it.",
    "Examples:",
    "  1  ↑  -> PrismaticJoint1  ++",
    "  1  ↓  -> PrismaticJoint1  --",
    "  2 3 ↑ -> PrismaticJoint23 ++",
    "",
    "Keys:",
    "  0..9         append digit to selection",
    "  Backspace    remove last digit",
    "  Space / c    clear selection",
    "  ↑ / w        inflate (increase target)",
    "  ↓ / s        deflate (decrease target)",
    "  q            quit",
]

class Teleop(Node):
    def __init__(self, params: Params):
        super().__init__("korusbed_keyboard_teleop")
        self.p = params
        self.pub = self.create_publisher(JointState, self.p.topic, 10)
        self.buffer = ""          # numeric selection buffer
        self.targets = {}         # name -> last absolute pos (float)

    def _apply_step(self, sign: float, stdscr):
        if not self.buffer:
            return
        try:
            idx = int(self.buffer)
        except ValueError:
            return
        jname = f"PrismaticJoint{idx}"
        prev = self.targets.get(jname, self.p.initial)
        new  = max(self.p.min_pos, min(self.p.max_pos, prev + sign * self.p.step))
        self.targets[jname] = new

        msg = JointState()
        msg.name = [jname]
        msg.position = [float(new)]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

        stdscr.addstr(len(HELP) + 3, 0, f"Commanded {jname:>18} -> {new:0.4f}           ")
        stdscr.clrtoeol()

    def run_curses(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)

        # draw help once
        stdscr.clear()
        for i, line in enumerate(HELP):
            stdscr.addstr(i, 0, line)
        stdscr.addstr(len(HELP) + 1, 0, f"Publishing to: {self.p.topic}   step={self.p.step}  clamp=[{self.p.min_pos},{self.p.max_pos}]")
        stdscr.addstr(len(HELP) + 2, 0, "Selected: (none)")

        period = 1.0 / self.p.rate_hz
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)

                ch = stdscr.getch()  # -1 if nothing
                if ch == -1:
                    time.sleep(period)
                    continue

                # digits
                if ord('0') <= ch <= ord('9'):
                    if len(self.buffer) < 6:
                        self.buffer += chr(ch)

                # backspace variants
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    if self.buffer:
                        self.buffer = self.buffer[:-1]

                # clear
                elif ch in (ord(' '), ord('c'), ord('C')):
                    self.buffer = ""

                # up / w
                elif ch in (curses.KEY_UP, ord('w'), ord('W')):
                    self._apply_step(+1.0, stdscr)

                # down / s
                elif ch in (curses.KEY_DOWN, ord('s'), ord('S')):
                    self._apply_step(-1.0, stdscr)

                # quit
                elif ch in (ord('q'), ord('Q')):
                    break

                # update status line
                sel = self.buffer if self.buffer else "(none)"
                stdscr.addstr(len(HELP) + 2, 0, f"Selected: {sel}          ")
                stdscr.clrtoeol()
                stdscr.refresh()

                time.sleep(period)
        finally:
            # curses.wrapper will restore the terminal
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KorusBed keyboard teleop (curses)")
    parser.add_argument("--topic", default="/korusbed/joint_pos_target")
    parser.add_argument("--step", type=float, default=0.003)
    parser.add_argument("--initial", type=float, default=0.70)
    parser.add_argument("--min", dest="min_pos", type=float, default=0.0)
    parser.add_argument("--max", dest="max_pos", type=float, default=1.2)
    parser.add_argument("--rate", type=float, default=120.0)
    args = parser.parse_args()

    rclpy.init()
    node = Teleop(Params(topic=args.topic, step=args.step, initial=args.initial,
                         min_pos=args.min_pos, max_pos=args.max_pos, rate_hz=args.rate))
    try:
        curses.wrapper(node.run_curses)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
