"""
Human-expert evaluation episode (T3.8, D49 C, D80): the operator packs one
llm-robotic-packer evaluation sequence under the harness's rules.

Rules (D80):
  - boxes arrive in the file's order, one at a time; nothing about later boxes is shown
  - the operator chooses (rotation, anchor) from the harness's top-8-per-rotation
    shortlist only (lfd/harness_view.py); the path is the standard template
  - a box with no offered anchor is skipped automatically, as in the harness
  - no voluntary skip, no undo: a box with anchors must be placed, and stays placed

Output is a choices file (schema packi-human-choices/1), written after every box
so a session can be resumed.  It is NOT a demonstration file: these are
evaluation episodes and must never be appended to data/demos/ or trained on.
The packer scores it with `evaluate.py --method human` (harness/human.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from lfd.harness_view import build_state, lookup

SCHEMA = "packi-human-choices/1"
RULES = {"top_k_per_rotation": 8, "voluntary_skip": False, "undo": False, "lookahead": False,
         "path": "template (overhead -> pre-descend -> target)"}
ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path) -> Dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    dirty = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(dirty) if dirty is not None else None}


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HumanSession:
    def __init__(self, sequence_path: str, out_path: str, operator: str, view: str, clock=time.perf_counter):
        self.sequence_path = os.path.abspath(sequence_path)
        self.out_path = os.path.abspath(out_path)
        self.clock = clock
        with open(self.sequence_path) as f:
            self.sequence = json.load(f)
        self.bin_dims = [int(v) for v in self.sequence["bin_dims"]]
        self.boxes = [list(map(int, b)) for b in self.sequence["boxes"]]
        self.placed: List[Dict] = []
        self.record = {
            "schema": SCHEMA,
            "dataset": self.sequence["dataset"],
            "seed": int(self.sequence["seed"]),
            "sequence_file": f"data/sequences/{self.sequence['dataset']}/seed{self.sequence['seed']}.json",
            "sequence_sha256": sha256_file(self.sequence_path),
            "bin_dims": self.bin_dims,
            "n_items": len(self.boxes),
            "operator": operator,
            "rules": dict(RULES, view=view),
            "recorder_git": _git(ROOT),
            "sessions": [],
            "complete": False,
            "decisions": [],
        }
        if os.path.exists(self.out_path):
            self._resume()
        self.record["sessions"].append({"started_at": _now(), "resumed_at_box": len(self.record["decisions"])})
        self.state: Optional[Dict] = None
        self._shown_at: Optional[float] = None
        self._advance()

    # ------------------------------------------------------------ resume --

    def _resume(self):
        with open(self.out_path) as f:
            old = json.load(f)
        if old.get("schema") != SCHEMA or old.get("sequence_sha256") != self.record["sequence_sha256"]:
            raise RuntimeError(f"{self.out_path} exists but belongs to a different sequence or format")
        for key in ("operator", "rules", "sessions", "complete"):
            self.record[key] = old[key] if key != "rules" else old.get(key, self.record[key])
        self.record["recorder_git_previous"] = old.get("recorder_git_previous", []) + [old["recorder_git"]]
        # replay the stored decisions through the same shortlist; any mismatch means the file was edited
        for d in old["decisions"]:
            i = len(self.record["decisions"])
            state = build_state(self.placed, self.boxes[i], self.bin_dims)
            if d["index"] != i or d["size"] != self.boxes[i]:
                raise RuntimeError(f"stored decision {i} does not match the sequence")
            if d["outcome"] == "placed":
                size, pos = lookup(state, d["rotation_index"], d["anchor_id"])
                if pos != d["pos"] or size != d["chosen_size"]:
                    raise RuntimeError(f"stored decision {i} is not the shortlist entry it names")
                self.placed.append({"position": pos, "size": size})
            elif state["anchors_indexed"]:
                raise RuntimeError(f"stored decision {i} skips a box that has anchors")
            self.record["decisions"].append(d)

    # ------------------------------------------------------------ stepping --

    @property
    def index(self) -> int:
        return len(self.record["decisions"])

    @property
    def done(self) -> bool:
        return self.index >= len(self.boxes)

    def fill(self) -> float:
        return sum(b["size"][0] * b["size"][1] * b["size"][2] for b in self.placed) / (
            self.bin_dims[0] * self.bin_dims[1] * self.bin_dims[2])

    def _advance(self):
        """Auto-skip boxes with no anchor; stop at the next box the operator must place."""
        while not self.done:
            i = self.index
            state = build_state(self.placed, self.boxes[i], self.bin_dims)
            if state["anchors_indexed"]:
                self.state = state
                self._shown_at = None   # set by mark_shown() once the box is on screen
                return
            self.record["decisions"].append({"index": i, "size": self.boxes[i], "outcome": "skipped_no_anchor",
                                             "n_anchors_offered": 0, "fill_after": self.fill()})
        self.state = None
        self.record["complete"] = True
        self.record["sessions"][-1]["finished_at"] = _now()
        self.save()

    def mark_shown(self):
        """Start the decision clock (called when the box is first drawn)."""
        if self._shown_at is None:
            self._shown_at = self.clock()

    def choose(self, rotation_index: int, anchor_id: str) -> Dict:
        if self.state is None:
            raise RuntimeError("sequence finished")
        size, pos = lookup(self.state, rotation_index, anchor_id)
        if pos is None:
            raise ValueError(f"({rotation_index}, {anchor_id}) is not offered for this box")
        t = self.clock()
        i = self.index
        self.placed.append({"position": pos, "size": size})
        d = {"index": i, "size": self.boxes[i], "outcome": "placed", "rotation_index": int(rotation_index),
             "anchor_id": anchor_id, "pos": pos, "chosen_size": size,
             "n_anchors_offered": len(self.state["anchors_indexed"]),
             "decision_time_s": (t - self._shown_at) if self._shown_at is not None else None,
             "decided_at": _now(), "fill_after": self.fill()}
        self.record["decisions"].append(d)
        self.save()
        self._advance()
        return d

    def save(self):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        tmp = self.out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.record, f, indent=1)
        os.replace(tmp, self.out_path)
