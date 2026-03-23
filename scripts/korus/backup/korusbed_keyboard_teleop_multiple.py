#!/usr/bin/env python3
# korusbed_keyboard_teleop_multiple.py
#
# Keyboard teleop for KorusBed with combos & ranges.
# Robust to small terminals (no "addwstr ERR"), supports resize.

import time
import locale
from dataclasses import dataclass
from typing import List, Set, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import curses

# Ensure curses sizing & wide chars work with your locale
locale.setlocale(locale.LC_ALL, "")

@dataclass
class Params:
    topic: str = "/korusbed/joint_pos_target"
    step: float = 0.003        # per keypress increment
    initial: float = 0.70      # seed value if a joint hasn't been touched yet
    min_pos: float = 0.0       # soft clamp (sim also clamps)
    max_pos: float = 1.0
    rate_hz: float = 120.0
    max_index: Optional[int] = 31   # highest prismatic index, or None for unlimited
    no_ui: bool = False        # headless mode (prints to stdout)

HELP = [
    "KorusBed Keyboard Teleop — combos & ranges",
    "Type digits/commas/hyphens to choose joints, then Up/Down (or W/S) to move them.",
    "Examples:",
    "  1  ↑              -> PrismaticJoint1  ++",
    "  23 ↓              -> PrismaticJoint23 --",
    "  0,8,16  ↑         -> 0,8,16  ++",
    "  0-3,7,10-12  ↓    -> 0..3,7,10..12  --",
    "",
    "Keys: digits , - | Backspace=del | Space/c=clear | a=all (0..max) | ↑/w=inflate | ↓/s=deflate | q=quit",
]

def compact_ranges(nums: List[int]) -> str:
    if not nums:
        return ""
    nums = sorted(nums)
    out = []
    start = prev = nums[0]
    for x in nums[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = x
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)

class Teleop(Node):
    def __init__(self, p: Params):
        super().__init__("korusbed_keyboard_teleop")
        self.p = p
        self.pub = self.create_publisher(JointState, self.p.topic, 10)
        self.buf = ""          # selection buffer, e.g. "0-3,7,10-12"
        self.targets = {}      # name -> last absolute pos

    # ---- selection parsing ----
    def _parse_selection(self, text: str) -> List[int]:
        text = (text or "").replace(" ", "")
        if not text:
            return []
        picked: Set[int] = set()
        for token in text.split(","):
            if not token:
                continue
            if "-" in token:
                a, _, b = token.partition("-")
                if not (a.isdigit() and b.isdigit()):
                    continue
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                for k in range(lo, hi + 1):
                    if k >= 0 and (self.p.max_index is None or k <= self.p.max_index):
                        picked.add(k)
            else:
                if token.isdigit():
                    k = int(token)
                    if k >= 0 and (self.p.max_index is None or k <= self.p.max_index):
                        picked.add(k)
        return sorted(picked)

    def _publish_targets(self, names: List[str], vals: List[float]):
        msg = JointState()
        msg.name = names
        msg.position = vals
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

    def _apply_step(self, sign: float):
        idxs = self._parse_selection(self.buf)
        if not idxs:
            return ([], [])
        names, vals = [], []
        for i in idxs:
            jn = f"PrismaticJoint{i}"
            prev = self.targets.get(jn, self.p.initial)
            new = max(self.p.min_pos, min(self.p.max_pos, prev + sign * self.p.step))
            self.targets[jn] = new
            names.append(jn)
            vals.append(float(new))
        self._publish_targets(names, vals)
        return (names, vals)

    # ---- helpers for safe drawing ----
    @staticmethod
    def _safe_addstr(stdscr, y: int, x: int, s: str):
        max_y, max_x = stdscr.getmaxyx()
        if y < 0 or y >= max_y or x < 0 or x >= max_x:
            return
        if s is None:
            s = ""
        if x + len(s) > max_x:
            s = s[: max_x - x]
        try:
            stdscr.addstr(y, x, s)
        except curses.error:
            pass

    @staticmethod
    def _safe_clrtoeol(stdscr, y: int):
        max_y, _ = stdscr.getmaxyx()
        if 0 <= y < max_y:
            try:
                stdscr.move(y, 0)
                stdscr.clrtoeol()
            except curses.error:
                pass

    # ------------- curses loop -------------
    def run_curses(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        curses.noecho()
        curses.cbreak()

        period = 1.0 / self.p.rate_hz

        def redraw():
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()

            # Reserve 5 status lines at bottom:
            # meta, buffer, active, last-cmd, hint
            reserve = 5
            usable = max_y - reserve
            help_lines = HELP[: max(0, usable)]

            # Draw help
            for i, line in enumerate(help_lines):
                self._safe_addstr(stdscr, i, 0, line)

            # Status rows
            y_meta   = len(help_lines) + 0
            y_buf    = len(help_lines) + 1
            y_active = len(help_lines) + 2
            y_last   = len(help_lines) + 3
            y_hint   = len(help_lines) + 4

            meta = f"Topic: {self.p.topic} | step={self.p.step} clamp=[{self.p.min_pos},{self.p.max_pos}]"
            if self.p.max_index is not None:
                meta += f" | max-index={self.p.max_index}"
            self._safe_addstr(stdscr, y_meta, 0, meta)
            self._safe_addstr(stdscr, y_buf, 0, "Buffer: " + (self.buf if self.buf else "(empty)"))

            active = self._parse_selection(self.buf)
            act_str = f"Active: {compact_ranges(active)} ({len(active)})" if active else "Active: (none)"
            self._safe_addstr(stdscr, y_active, 0, act_str)

            self._safe_addstr(stdscr, y_last, 0, "Last: (none)")
            self._safe_addstr(stdscr, y_hint, 0, "↑/w inflate | ↓/s deflate | a=all | Space/c=clear | Backspace=del | q=quit")

            stdscr.refresh()
            return (y_meta, y_buf, y_active, y_last, y_hint)

        y_meta, y_buf, y_active, y_last, y_hint = redraw()

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)

                ch = stdscr.getch()
                if ch == -1:
                    time.sleep(period)
                    continue

                if ch == curses.KEY_RESIZE:
                    y_meta, y_buf, y_active, y_last, y_hint = redraw()
                    continue

                # digits, comma, hyphen
                if (ord('0') <= ch <= ord('9')) or ch in (ord(','), ord('-')):
                    if len(self.buf) < 64:
                        self.buf += chr(ch)

                # backspace variants
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    if self.buf:
                        self.buf = self.buf[:-1]

                # clear buffer
                elif ch in (ord(' '), ord('c'), ord('C')):
                    self.buf = ""

                # select ALL (requires max-index)
                elif ch in (ord('a'),):
                    if self.p.max_index is not None and self.p.max_index >= 0:
                        self.buf = f"0-{self.p.max_index}"

                # up/w = inflate
                elif ch in (curses.KEY_UP, ord('w'), ord('W')):
                    names, vals = self._apply_step(+1.0)
                    if names:
                        self._safe_clrtoeol(stdscr, y_last)
                        self._safe_addstr(stdscr, y_last, 0, f"Last: + {len(names)} joint(s), e.g., {names[-1]} -> {vals[-1]:.4f}")

                # down/s = deflate
                elif ch in (curses.KEY_DOWN, ord('s'), ord('S')):
                    names, vals = self._apply_step(-1.0)
                    if names:
                        self._safe_clrtoeol(stdscr, y_last)
                        self._safe_addstr(stdscr, y_last, 0, f"Last: - {len(names)} joint(s), e.g., {names[-1]} -> {vals[-1]:.4f}")

                # quit
                elif ch in (ord('q'), ord('Q')):
                    break

                # refresh status rows
                self._safe_clrtoeol(stdscr, y_buf)
                self._safe_addstr(stdscr, y_buf, 0, "Buffer: " + (self.buf if self.buf else "(empty)"))

                self._safe_clrtoeol(stdscr, y_active)
                active = self._parse_selection(self.buf)
                act_str = f"Active: {compact_ranges(active)} ({len(active)})" if active else "Active: (none)"
                self._safe_addstr(stdscr, y_active, 0, act_str)

                stdscr.refresh()
                time.sleep(period)
        finally:
            pass

    # -------- minimal (no UI) loop, if needed --------
    def run_headless(self):
        self.get_logger().info("Headless mode: type commands like '0-3,7 +', '10 -', 'q' to quit.")
        period = 1.0 / self.p.rate_hz
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    continue
                if line.lower() in ("q", "quit", "exit"):
                    break
                # parse "sel +"/"sel -"
                try:
                    sel, op = line.split()
                    self.buf = sel
                    if op == "+":
                        self._apply_step(+1.0)
                    elif op == "-":
                        self._apply_step(-1.0)
                except Exception:
                    print("Format: '0-3,7 +' or '10 -'")
                time.sleep(period)
        finally:
            pass

def main():
    import argparse
    parser = argparse.ArgumentParser(description="KorusBed keyboard teleop (curses, combos & ranges)")
    parser.add_argument("--topic", default="/korusbed/joint_pos_target")
    parser.add_argument("--step", type=float, default=0.003)
    parser.add_argument("--initial", type=float, default=0.70)
    parser.add_argument("--min", dest="min_pos", type=float, default=0.7)
    parser.add_argument("--max", dest="max_pos", type=float, default=1.2)
    parser.add_argument("--rate", type=float, default=120.0)
    parser.add_argument("--max-index", type=int, default=31, help="highest prismatic joint index (e.g., 31 for 0..31)")
    parser.add_argument("--no-ui", action="store_true", help="run without curses UI (stdin commands)")
    args = parser.parse_args()

    rclpy.init()
    node = Teleop(Params(topic=args.topic, step=args.step, initial=args.initial,
                         min_pos=args.min_pos, max_pos=args.max_pos,
                         rate_hz=args.rate, max_index=args.max_index, no_ui=args.no_ui))
    try:
        if node.p.no_ui:
            node.run_headless()
        else:
            curses.wrapper(node.run_curses)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
