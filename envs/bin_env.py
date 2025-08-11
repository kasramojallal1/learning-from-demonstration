import os
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from utils.geometry import (
    aabb_overlap,
    supported_on_floor_or_stack,
    enumerate_floor_anchors_for_rotation,
    enumerate_all_anchors_for_rotation
)

class BinPackingLfDEnv:
    """
    Interactive LfD environment:
      - Shows current bin + incoming box
      - User cycles rotation ('r') and anchor ('a'/'d'), clicks waypoints, Enter to commit
      - Writes snapshots and exposes compact state + label for logging
    """

    def __init__(self, bin_size=(10,10,10), run_dir=".", rng_seed=0):
        self.bin_w, self.bin_h, self.bin_d = bin_size
        self.run_dir = run_dir
        self.snapshot_dir = os.path.join(run_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.rng = np.random.default_rng(rng_seed)

        self.placed = []   # list of dicts: {"pos":(x,y,z), "size":(w,h,d)}
        self.incoming = None
        self.rotations = []
        self.rotation_idx = 0

        self.anchors_by_rot = []  # list of lists of (x,y,z)
        self.anchor_idx = 0

        self.path_points = []  # user clicks
        self.fig = None
        self.ax = None

        # Last labels/meta saved for logger
        self._last_label = None
        self._last_meta = None

    # ------------- Public API -------------

    def prepare_incoming_box(self, original_size, rotations):
        self.incoming = {
            "original_size": tuple(int(x) for x in original_size),
            "size": tuple(int(x) for x in original_size)  # current rotation size
        }
        self.rotations = [tuple(map(int, r)) for r in rotations]
        self.rotation_idx = 0
        self._apply_rotation(0)
        self._recompute_anchors()
        self.anchor_idx = 0
        self.path_points = []
        self._last_label = None
        self._last_meta = None

    def interactive_place_one(self):
        """
        Returns True if committed, False if skipped (Esc).
        Keybinds:
          - r: next rotation
          - a / d: prev/next anchor
          - left click: add waypoint
          - backspace: remove last waypoint
          - enter: commit
          - esc: skip
        """
        self._ensure_fig()

        committed = False
        while True:
            self._render()
            key = plt.waitforbuttonpress(timeout=-1)  # block for key/mouse
            # key is True for key press, False for mouse click, None for timeout
            if key is None:
                continue

            # Check for keyboard event
            ev = plt.gcf().canvas.manager.key_press_handler_id  # not helpful; use mpl connections instead
            # We instead rely on matplotlib's internal key press event via figure.canvas.mpl_connect
            # But to keep it simple, we detect from event loop handlers set in _ensure_fig.
            # The actual logic is handled via callbacks setting flags.
            if self._event_state["commit"]:
                committed = self._commit_if_valid()
                self._event_state["commit"] = False
                if committed:
                    break
            if self._event_state["skip"]:
                self._event_state["skip"] = False
                break
            if self._event_state["rot_next"]:
                self._event_state["rot_next"] = False
                self._next_rotation()
            if self._event_state["anchor_prev"]:
                self._event_state["anchor_prev"] = False
                self._prev_anchor()
            if self._event_state["anchor_next"]:
                self._event_state["anchor_next"] = False
                self._next_anchor()
            if self._event_state["undo"]:
                self._event_state["undo"] = False
                if self.path_points:
                    self.path_points.pop()

            # mouse click added waypoints in callback

        return committed

    def export_compact_state(self):
        anchors_indexed = []
        for ri, anchors in enumerate(self.anchors_by_rot):
            for j, (x,y,z) in enumerate(anchors):
                anchors_indexed.append({
                    "id": f"r{ri}_a{j}",
                    "rotation_index": ri,
                    "pos": [int(x), int(y), int(z)]
                })
        return {
            "bin": {"w": self.bin_w, "h": self.bin_h, "d": self.bin_d},
            "incoming_box": {
                "original_size": list(self.incoming["original_size"]),
                "rotations": [list(r) for r in self.rotations]
            },
            "anchors_indexed": anchors_indexed
        }

    def export_last_label(self):
        return dict(self._last_label) if self._last_label else None

    def export_last_meta(self):
        return dict(self._last_meta) if self._last_meta else None

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    # ------------- Internal helpers -------------

    def _ensure_fig(self):
        if self.fig is not None:
            return
        self.fig = plt.figure()
        try:
            self.fig.canvas.manager.set_window_title("LfD Bin Packing — Demo Collector")
        except Exception:
            pass
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect([self.bin_w, self.bin_h, self.bin_d])

        # Event state + bindings
        self._event_state = {
            "commit": False, "skip": False, "rot_next": False,
            "anchor_prev": False, "anchor_next": False, "undo": False
        }
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        plt.ion()
        plt.show(block=False)

    def _on_key(self, event):
        if event.key == "enter":
            self._event_state["commit"] = True
        elif event.key == "escape":
            self._event_state["skip"] = True
        elif event.key in ("r",):
            self._event_state["rot_next"] = True
        elif event.key in ("a", "left"):
            self._event_state["anchor_prev"] = True
        elif event.key in ("d", "right"):
            self._event_state["anchor_next"] = True
        elif event.key in ("backspace",):
            self._event_state["undo"] = True

    def _on_click(self, event):
        # Left click to add waypoint in 3D via simple projection on current z of cursor
        if event.button != 1 or event.inaxes != self.ax:
            return
        # Map 2D click into approximate 3D (we’ll snap to grid and keep current path semantics: overhead -> xy -> descend)
        # For simplicity, interpret click as (x,y) at current top Z
        x = int(round(event.xdata)) if event.xdata is not None else 0
        y = int(round(event.ydata)) if event.ydata is not None else 0
        # Use an overhead Z above bin
        z = self.bin_d + 2
        self.path_points.append([max(0, min(self.bin_w, x)),
                                 max(0, min(self.bin_h, y)),
                                 z])

    def _apply_rotation(self, idx):
        self.rotation_idx = int(idx) % len(self.rotations)
        self.incoming["size"] = self.rotations[self.rotation_idx]

    def _recompute_anchors(self):
        self.anchors_by_rot = []
        for rot in self.rotations:
            # Start with simple supported anchors (floor + stack) for speed
            anchors = enumerate_all_anchors_for_rotation(
                bin_size=(self.bin_w, self.bin_h, self.bin_d),
                placed=self.placed,
                box_size=rot
            )
            self.anchors_by_rot.append(anchors)

    def _next_rotation(self):
        self._apply_rotation(self.rotation_idx + 1)
        self._recompute_anchors()
        self.anchor_idx = 0
        self.path_points = []

    def _prev_anchor(self):
        if not self.anchors_by_rot[self.rotation_idx]:
            return
        self.anchor_idx = (self.anchor_idx - 1) % len(self.anchors_by_rot[self.rotation_idx])
        self.path_points = []

    def _next_anchor(self):
        if not self.anchors_by_rot[self.rotation_idx]:
            return
        self.anchor_idx = (self.anchor_idx + 1) % len(self.anchors_by_rot[self.rotation_idx])
        self.path_points = []

    def _commit_if_valid(self):
        anchors = self.anchors_by_rot[self.rotation_idx]
        if not anchors:
            print("⚠️ No anchors for this rotation.")
            return False
        target = anchors[self.anchor_idx]
        # Ensure last waypoint ends at target (append minimal final descent)
        path = self._normalize_path_to_target(self.path_points, target)

        # Validate path: keep inside bounds except overhead segment; final equals target
        if not path or path[-1] != list(target):
            print("❌ Path does not end at target.")
            return False

        # Place box and validate collisions/support
        size = self.incoming["size"]
        if not self._can_place_at(target, size):
            print("❌ Target not placeable (collision/support).")
            return False

        # Commit placement
        self.placed.append({"pos": tuple(map(int, target)), "size": tuple(map(int, size))})

        # Save snapshot
        snap_path = os.path.join(self.snapshot_dir, f"placement_{len(self.placed):04d}.png")
        self._render(save_path=snap_path)

        # Save labels/meta
        rot_idx = self.rotation_idx
        a_id = f"r{rot_idx}_a{self.anchor_idx}"
        self._last_label = {
            "rotation_index": rot_idx,
            "anchor_id": a_id,
            "path": path
        }
        util_gain = int(size[0]*size[1]*size[2])
        self._last_meta = {"util_gain": util_gain}

        # Reset path for next placement
        self.path_points = []
        return True

    def _normalize_path_to_target(self, pts, target):
        # Minimal macro path: overhead -> XY align -> descend to z
        tx, ty, tz = map(int, target)
        overhead_z = self.bin_d + 2
        if not pts:
            # Generate 3-point macro path
            return [[tx, ty, overhead_z], [tx, ty, tz + 1], [tx, ty, tz]]
        # Ensure last three steps end at target with descent
        path = [list(map(int, p)) for p in pts]
        # Align XY at overhead if last point not aligned
        last = path[-1]
        if last[0] != tx or last[1] != ty:
            path.append([tx, ty, overhead_z])
        # Descend
        if path[-1][2] < tz:
            path[-1][2] = overhead_z
        if path[-1][2] != tz + 1:
            path.append([tx, ty, tz + 1])
        path.append([tx, ty, tz])
        # Clamp inside bin on final point (safety)
        path[-1][0] = min(max(0, path[-1][0]), self.bin_w - 1)
        path[-1][1] = min(max(0, path[-1][1]), self.bin_h - 1)
        path[-1][2] = min(max(0, path[-1][2]), self.bin_d - 1)
        return path

    def _can_place_at(self, pos, size):
        x, y, z = map(int, pos)
        w, h, d = map(int, size)
        # Boundary
        if x < 0 or y < 0 or z < 0: return False
        if x + w > self.bin_w or y + h > self.bin_h or z + d > self.bin_d: return False
        # Support
        if not supported_on_floor_or_stack((x,y,z,w,h,d), self.placed):
            return False
        # Collision
        for b in self.placed:
            if aabb_overlap((x,y,z,w,h,d), (*b["pos"], *b["size"])):
                return False
        return True

    def _render(self, save_path=None):
        self.ax.clear()
        self.ax.set_xlim(0, self.bin_w)
        self.ax.set_ylim(0, self.bin_h)
        self.ax.set_zlim(0, self.bin_d)
        self.ax.view_init(elev=30, azim=45)
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.set_zlabel("Z")

        # Bin shell
        self.ax.bar3d(0, 0, 0, self.bin_w, self.bin_h, self.bin_d, color="gray", alpha=0.05, edgecolor="black")

        # Placed boxes
        for b in self.placed:
            px, py, pz = b["pos"]
            sx, sy, sz = b["size"]
            self.ax.bar3d(px, py, pz, sx, sy, sz, color="green", alpha=0.6)

        # Incoming at current anchor (ghost)
        anchors = self.anchors_by_rot[self.rotation_idx] if self.anchors_by_rot else []
        if anchors:
            ax, ay, az = anchors[self.anchor_idx]
            sx, sy, sz = self.incoming["size"]
            self.ax.bar3d(ax, ay, az, sx, sy, sz, color="blue", alpha=0.35)

        # Draw anchors
        if anchors:
            xs = [a[0] + 0.5 for a in anchors]
            ys = [a[1] + 0.5 for a in anchors]
            zs = [a[2] + 0.5 for a in anchors]
            self.ax.scatter(xs, ys, zs, c="red", s=10)
            # highlight current
            cx, cy, cz = anchors[self.anchor_idx]
            self.ax.scatter([cx + 0.5], [cy + 0.5], [cz + 0.5], c="yellow", s=50)

        # Path points
        if self.path_points:
            xs, ys, zs = zip(*self.path_points)
            self.ax.plot(xs, ys, zs, linewidth=2, alpha=0.9)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if save_path:
            self.fig.savefig(save_path, dpi=150)
