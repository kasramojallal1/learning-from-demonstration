"""
Human-expert evaluation (T3.8, D49 C, D80): pack one packer evaluation sequence
by hand under exactly the harness's rules, then score it in the packer.

    python human_eval.py --packer ../llm-robotic-packer --dataset curriculum25 --seed 0

Window: 3D view of the bin (drag to turn it) + top-down height map.  Only the
current box is shown; the offered spots are the harness's top-8 per rotation.
    R / Shift+R   next / previous rotation (rotations without a spot are passed over)
    A / D (or <- / ->)   previous / next spot for this rotation
    Enter         place the box (no undo, no skip)
Progress is saved after every box; rerun the same command to resume.

Writes <packer>/results/human/choices/<dataset>/seed<k>.json (packi-human-choices/1).
Score it in the packer with:  python evaluate.py --method human --dataset <d> --seed <k>

--autoplay greedy|first plays the sequence without a window (self-test only).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from lfd.harness_view import anchors_for_rotation, greedy_choice
from lfd.human_session import HumanSession

ENTER_DEBOUNCE_S = 0.6   # ignore Enter this soon after a new box appears (no accidental double-commit)


def default_packer() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for parent in [here] + [os.path.dirname(here)] + list(_parents(here)):
        cand = os.path.join(parent, "llm-robotic-packer")
        if os.path.exists(os.path.join(cand, "harness", "state.py")):
            return cand
    return os.path.join(os.path.dirname(here), "llm-robotic-packer")


def _parents(p):
    while True:
        q = os.path.dirname(p)
        if q == p:
            return
        yield q
        p = q


# ------------------------------------------------------------------ GUI --

class HumanEvalGUI:
    def __init__(self, session: HumanSession):
        import matplotlib
        matplotlib.use("TkAgg")
        for k in ("keymap.home", "keymap.back", "keymap.forward", "keymap.save", "keymap.quit",
                  "keymap.fullscreen", "keymap.grid", "keymap.pan", "keymap.zoom", "keymap.xscale", "keymap.yscale"):
            matplotlib.rcParams[k] = []
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        self.plt = plt
        self.s = session
        self.fig = plt.figure(figsize=(14, 7.5))
        try:
            self.fig.canvas.manager.set_window_title(
                f"Packi human-expert evaluation — {session.record['dataset']} seed {session.record['seed']}")
        except Exception:
            pass
        self.ax3 = self.fig.add_axes([0.0, 0.06, 0.56, 0.80], projection="3d")
        self.ax2 = self.fig.add_axes([0.60, 0.10, 0.36, 0.70])
        self.ax3.view_init(elev=28, azim=-60)
        self.cbar = None
        self.fig.text(0.01, 0.01, "R / Shift+R rotation    A / D spot    Enter place (no undo)    drag the 3D view to turn it",
                      fontsize=10, color="dimgray")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._new_box()

    # ---------------------------------------------------------- state --

    def _new_box(self):
        self.rot = None
        self.spot = 0
        self.box_shown_wall = time.monotonic()
        if self.s.state is not None:
            self.rot = self._rotations_with_spots()[0]
        self._render()
        self.s.mark_shown()

    def _rotations_with_spots(self):
        st = self.s.state
        return [r for r in range(len(st["incoming_box"]["rotations"])) if anchors_for_rotation(st, r)]

    def _spots(self):
        return anchors_for_rotation(self.s.state, self.rot)

    def _on_key(self, ev):
        if self.s.state is None:
            return
        key = ev.key
        if key in ("r", "R", "shift+r"):
            rots = self._rotations_with_spots()
            step = -1 if key in ("R", "shift+r") else 1
            self.rot = rots[(rots.index(self.rot) + step) % len(rots)]
            self.spot = 0
        elif key in ("a", "left"):
            self.spot = (self.spot - 1) % len(self._spots())
        elif key in ("d", "right"):
            self.spot = (self.spot + 1) % len(self._spots())
        elif key == "enter":
            if time.monotonic() - self.box_shown_wall < ENTER_DEBOUNCE_S:
                return
            a = self._spots()[self.spot]
            d = self.s.choose(a["rotation_index"], a["id"])
            print(f"[{d['index'] + 1:>2}/{len(self.s.boxes)}] {d['size']} -> rotation {d['rotation_index']} "
                  f"{d['chosen_size']} at {d['pos']}  ({d['decision_time_s']:.1f}s)  fill={d['fill_after']:.3f}")
            self._new_box()
            return
        else:
            return
        self._render()

    # ---------------------------------------------------------- drawing --

    def _render(self):
        import numpy as np
        s, ax3, ax2 = self.s, self.ax3, self.ax2
        W, H, D = s.bin_dims
        elev, azim = ax3.elev, ax3.azim
        ax3.clear()
        ax3.set_xlim(0, W); ax3.set_ylim(0, H); ax3.set_zlim(0, D)
        ax3.set_box_aspect([W, H, D])
        ax3.view_init(elev=elev, azim=azim)
        ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z (up)")
        ax3.bar3d(0, 0, 0, W, H, D, color="gray", alpha=0.03, edgecolor="black", linewidth=0.4)
        cmap = self.plt.get_cmap("Pastel2")
        for k, b in enumerate(s.placed):
            ax3.bar3d(*b["position"], *b["size"], color=cmap(k % 8), alpha=0.45, edgecolor="dimgray", linewidth=0.3)

        hm = np.zeros((W, H), dtype=int)
        for b in s.placed:
            (x, y, z), (w, h, d) = b["position"], b["size"]
            hm[x:x + w, y:y + h] = np.maximum(hm[x:x + w, y:y + h], z + d)

        ax2.clear()
        im = ax2.imshow(hm.T, origin="lower", extent=[0, W, 0, H], cmap="Greens", vmin=0, vmax=D)
        for x in range(W):
            for y in range(H):
                ax2.text(x + 0.5, y + 0.5, str(hm[x, y]), ha="center", va="center", fontsize=8,
                         color="white" if hm[x, y] > D * 0.6 else "black")
        ax2.set_xticks(range(W + 1)); ax2.set_yticks(range(H + 1))
        ax2.grid(True, color="gray", linewidth=0.3)
        ax2.set_xlabel("X"); ax2.set_ylabel("Y")
        ax2.set_title("Top-down: stack height in each cell", fontsize=10)
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)

        if s.state is None:
            self.fig.suptitle(f"Sequence finished — final fill {s.fill() * 100:.1f}%.  "
                              f"Saved to {os.path.basename(s.out_path)}. Close the window.", fontsize=13)
            self.fig.canvas.draw_idle()
            return

        st = s.state
        rots = st["incoming_box"]["rotations"]
        spots = self._spots()
        a = spots[self.spot]
        size = rots[self.rot]
        (gx, gy, gz), (gw, gh, gd) = a["pos"], size
        ax3.bar3d(gx, gy, gz, gw, gh, gd, color="magenta", alpha=0.6, edgecolor="black", linewidth=1.2)
        ax3.scatter([p["pos"][0] + 0.5 for p in spots], [p["pos"][1] + 0.5 for p in spots],
                    [p["pos"][2] + 0.05 for p in spots], c="red", s=14, depthshade=False)
        ax3.scatter([gx + 0.5], [gy + 0.5], [gz + 0.05], c="yellow", edgecolors="k", s=60, depthshade=False)

        import matplotlib.patches as mpatches
        ax2.add_patch(mpatches.Rectangle((gx, gy), gw, gh, fill=True, facecolor=(1, 0, 1, 0.3),
                                         edgecolor="magenta", linewidth=2.5))
        ax2.scatter([p["pos"][0] + 0.15 for p in spots], [p["pos"][1] + 0.15 for p in spots], c="red", s=12)
        ax2.scatter([gx + 0.15], [gy + 0.15], c="yellow", edgecolors="k", s=50)

        rot_list = self._rotations_with_spots()
        self.fig.suptitle(
            f"Box {s.index + 1} of {len(s.boxes)}:  {'×'.join(map(str, st['incoming_box']['original_size']))}"
            f"      Bin fill so far: {s.fill() * 100:.1f}%\n"
            f"Rotation {rot_list.index(self.rot) + 1} of {len(rot_list)}: {'×'.join(map(str, size))} (X×Y×Z)"
            f"      Spot {self.spot + 1} of {len(spots)} at x={gx} y={gy} z={gz} (lands on height {gz}, top at {gz + gd})",
            fontsize=12)
        self.fig.canvas.draw_idle()

    def run(self):
        self.plt.show()


# ------------------------------------------------------------------ main --

def autoplay(session: HumanSession, mode: str):
    while session.state is not None:
        session.mark_shown()
        st = session.state
        a = greedy_choice(st) if mode == "greedy" else st["anchors_indexed"][0]
        session.choose(a["rotation_index"], a["id"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packer", default=None, help="llm-robotic-packer checkout (default: found beside this repo)")
    ap.add_argument("--dataset", default="curriculum25")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None, help="choices file (default: <packer>/results/human/choices/<dataset>/seed<k>.json)")
    ap.add_argument("--operator", default="Kasra Mojallal")
    ap.add_argument("--sequence", default=None, help="any sequence file instead of <packer>/data/sequences/<dataset>/seed<k>.json (practice); needs --out")
    ap.add_argument("--autoplay", choices=["greedy", "first"], help="self-test: play without a window")
    args = ap.parse_args(argv)

    packer = os.path.abspath(args.packer or default_packer())
    if args.sequence and not args.out:
        ap.error("--sequence needs --out")
    seq = os.path.abspath(args.sequence) if args.sequence else os.path.join(packer, "data", "sequences", args.dataset, f"seed{args.seed}.json")
    out = os.path.abspath(args.out or os.path.join(packer, "results", "human", "choices", args.dataset, f"seed{args.seed}.json"))
    if os.sep + os.path.join("data", "demos") + os.sep in out:
        sys.exit("refusing to write evaluation choices under data/demos/ (never train on evaluation sequences)")

    view = "autoplay:" + args.autoplay if args.autoplay else "3d+heightmap"
    session = HumanSession(seq, out, operator=("script:" + args.autoplay) if args.autoplay else args.operator, view=view)
    print(f"sequence {seq}\nchoices  {out}\nresuming at box {session.index + 1}" if session.index else
          f"sequence {seq}\nchoices  {out}")
    if args.autoplay:
        autoplay(session, args.autoplay)
    elif session.state is not None:
        HumanEvalGUI(session).run()
    print(f"{'complete' if session.record['complete'] else 'paused'}: {session.index}/{len(session.boxes)} boxes, "
          f"fill {session.fill():.3f}")


if __name__ == "__main__":
    main()
