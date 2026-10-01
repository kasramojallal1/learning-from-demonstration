"""
The evaluation harness's view of one box (T3.8, D80): what every policy is offered.

Mirror of llm-robotic-packer/harness/state.py::build_state without anchor
shuffling, built on the vendored packer anchor pipeline
(training/packer_vendor/state_manager.py, byte-identical to the packer's):

    all feasible floor+top anchors -> vertical-clearance filter -> Eq. 5 score -> top-8 per rotation

tests/test_human_eval.py checks it against the packer's own build_state.
Placed boxes use the packer's layout: {"position": [x, y, z], "size": [w, h, d]}.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from training.packer_vendor.state_manager import (
    filter_anchors_with_clearance,
    generate_anchor_positions,
    generate_orientations,
    score_anchor,
    topk_anchors,
)

TOP_K = 8   # harness/state.py TOP_K


def build_state(placed_boxes: List[Dict], box_size: List[int], bin_dims: List[int], top_k: int = TOP_K) -> Dict:
    rotations = generate_orientations(box_size)
    anchors: List[Dict] = []
    for r_idx, rot_size in enumerate(rotations):
        cand = generate_anchor_positions(placed_boxes, rot_size, bin_dims)
        cand = filter_anchors_with_clearance(cand, rot_size, placed_boxes, bin_dims)
        cand = topk_anchors(cand, rot_size, bin_dims, k=top_k)
        for j, pos in enumerate(cand):
            anchors.append({"id": f"r{r_idx}_a{j}", "rotation_index": r_idx, "pos": list(map(int, pos))})
    return {
        "bin": {"w": int(bin_dims[0]), "h": int(bin_dims[1]), "d": int(bin_dims[2])},
        "incoming_box": {"original_size": list(map(int, box_size)), "rotations": rotations},
        "anchors_indexed": anchors,
    }


def greedy_choice(state: Dict) -> Optional[Dict]:
    """harness/policies.py GreedyPolicy.pick: highest Eq. 5 score; ties -> lowest z, y, x, rotation_index."""
    bin_dims = [state["bin"]["w"], state["bin"]["h"], state["bin"]["d"]]
    best, best_key = None, None
    for a in state["anchors_indexed"]:
        size = state["incoming_box"]["rotations"][a["rotation_index"]]
        x, y, z = a["pos"]
        key = (-score_anchor(a["pos"], size, bin_dims), z, y, x, a["rotation_index"])
        if best_key is None or key < best_key:
            best, best_key = a, key
    return best


def anchors_for_rotation(state: Dict, r_idx: int) -> List[Dict]:
    return [a for a in state["anchors_indexed"] if a["rotation_index"] == r_idx]


def lookup(state: Dict, r_idx: int, anchor_id: str) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    for a in state["anchors_indexed"]:
        if a["id"] == anchor_id and a["rotation_index"] == r_idx:
            return list(state["incoming_box"]["rotations"][r_idx]), list(a["pos"])
    return None, None
