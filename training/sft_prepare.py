# training/sft_prepare.py
"""
Turn the recorded demonstrations into SFT examples that match the evaluation
harness call-for-call (D24, D31b-D38).

For every kept demo record two chat examples are written:

    pick:  [system SYSTEM_PICK, user pick_user(state), assistant {"rotation_index","anchor_id"}]
    path:  [system SYSTEM_PATH, user path_user(target_pos), assistant {"path": [...]}]

where `state` is the packer's test-time view of the bin: all feasible anchors
-> vertical-clearance filter -> score_anchor -> top-8 per rotation (the same
functions as llm-robotic-packer/harness/state.py, vendored in packer_vendor/),
with anchor order and ids shuffled by a per-record seeded RNG (D34).

Steps
  1. Replay the demo file in order. Records recorded before the 2025-08-11
     generator fix (the first 21, D32) are dropped: their candidate lists contain
     infeasible anchors and the box sampler seed was fixed. Every later record's
     candidate list is recomputed with the recorder's generator from the replayed
     bin and must be reproduced exactly (safety net). A record whose list matches
     an empty bin starts a new episode.
  2. Re-serialize the bin state with the packer pipeline (top-8). If the
     demonstrated anchor is not in the shortlist it is appended to its rotation
     (D33). A record whose anchor the packer would never offer is dropped.
  3. Shuffle ids/order per record (D34); the label follows.
  4. Split 90/10 by episode with random.Random(13) (D38).

Outputs (OUT_DIR, default data/processed_v2/):
    train.jsonl, test.jsonl   one example per line: {"messages": [...], "kind", "record", "episode"}
    manifest.json             counts, dropped records, split, seeds, input hash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.packer_vendor import prompts                                   # noqa: E402
from training.packer_vendor.state_manager import (                           # noqa: E402
    filter_anchors_with_clearance,
    generate_anchor_positions,
    generate_orientations,
    topk_anchors,
)
from utils.geometry import enumerate_all_anchors_for_rotation                # noqa: E402

IN_PATH = ROOT / "data" / "demos" / "binpack_lfd.jsonl"
OUT_DIR = ROOT / "data" / "processed_v2"

TOP_K = 8             # harness.state.TOP_K
SPLIT_SEED = 13       # D38
SHUFFLE_SEED = 2026   # D34
TEST_FRACTION = 0.10
# Records before this UTC timestamp were recorded with the pre-fix anchor generator
# (fixed in commit 36a3f4f, 2025-08-11 11:16 -0400; the fixed code was already in
# the working tree from record 21 on, 15:10:42 UTC).  D30b/D32.
PRE_FIX_CUTOFF_TS = "2025-08-11T15:10:00"


# ---------------------------------------------------------------- replay ----

def recorder_anchors(bin_size, placed, rotations) -> List[Dict]:
    """The recorder's candidate list (envs/bin_env.py export_compact_state)."""
    out = []
    for ri, rot in enumerate(rotations):
        for j, (x, y, z) in enumerate(enumerate_all_anchors_for_rotation(bin_size, placed, rot)):
            out.append({"id": f"r{ri}_a{j}", "rotation_index": ri, "pos": [x, y, z]})
    return out


def chosen_anchor(rec: Dict) -> Tuple[int, List[int], List[int]]:
    """(rotation_index, size, pos) of the demonstrated placement."""
    st, lab = rec["state"], rec["label"]
    ri = int(lab["rotation_index"])
    pos = next(a["pos"] for a in st["anchors_indexed"] if a["id"] == lab["anchor_id"])
    return ri, list(st["incoming_box"]["rotations"][ri]), list(pos)


def replay(records: List[Dict]):
    """
    Yield (index, record, episode_id, placed_before, status) in file order.
    status: "ok" | "unreproduced" (D32).  The placement is applied to the
    replayed bin in every case so that later records stay consistent.
    """
    placed: List[Dict] = []
    episode = -1
    for i, rec in enumerate(records):
        st = rec["state"]
        bin_size = (st["bin"]["w"], st["bin"]["h"], st["bin"]["d"])
        rots = [tuple(r) for r in st["incoming_box"]["rotations"]]
        if recorder_anchors(bin_size, placed, rots) == st["anchors_indexed"] and placed:
            status = "ok"
        elif recorder_anchors(bin_size, [], rots) == st["anchors_indexed"]:
            placed, episode, status = [], episode + 1, "ok"
        else:
            status = "unreproduced"
        yield i, rec, episode, [dict(b) for b in placed], status
        ri, size, pos = chosen_anchor(rec)
        placed.append({"pos": pos, "size": size})


# ----------------------------------------------------- packer's view ----

def packer_view(placed_before: List[Dict], rec: Dict, rng: Optional[random.Random]):
    """
    harness.state.build_state on the replayed bin, plus D33's force-include.
    Returns (state, label_anchor_id) or (None, reason) when the demonstrated
    anchor is not feasible for the packer.
    """
    st = rec["state"]
    bin_dims = [st["bin"]["w"], st["bin"]["h"], st["bin"]["d"]]
    box_size = list(st["incoming_box"]["original_size"])
    pb = [{"position": list(b["pos"]), "size": list(b["size"])} for b in placed_before]
    ri_star, size_star, pos_star = chosen_anchor(rec)

    rotations = generate_orientations(box_size)
    if rotations != [list(r) for r in st["incoming_box"]["rotations"]]:
        return None, "rotation set differs from the recorder's"

    anchors: List[Dict] = []
    forced = False
    for r_idx, rot_size in enumerate(rotations):
        cand = generate_anchor_positions(pb, rot_size, bin_dims)
        cand = filter_anchors_with_clearance(cand, rot_size, pb, bin_dims)
        if r_idx == ri_star and pos_star not in cand:
            return None, "demonstrated anchor is not offered by the packer"
        short = topk_anchors(cand, rot_size, bin_dims, k=TOP_K)
        if r_idx == ri_star and pos_star not in short:
            short = short + [pos_star]          # D33
            forced = True
        ids = list(range(len(short)))
        if rng is not None:
            rng.shuffle(ids)
        for j, pos in zip(ids, short):
            anchors.append({"id": f"r{r_idx}_a{j}", "rotation_index": r_idx, "pos": list(map(int, pos))})
    if rng is not None:
        rng.shuffle(anchors)

    state = {
        "bin": {"w": int(bin_dims[0]), "h": int(bin_dims[1]), "d": int(bin_dims[2])},
        "incoming_box": {"original_size": list(map(int, box_size)), "rotations": rotations},
        "anchors_indexed": anchors,
    }
    label_id = next(a["id"] for a in anchors if a["rotation_index"] == ri_star and a["pos"] == pos_star)
    return state, (label_id, forced)


def build_examples(state: Dict, label_id: str, rec: Dict, idx: int, episode: int) -> List[Dict]:
    ri, _, pos = chosen_anchor(rec)
    pick = {"messages": prompts.pick_messages(state, []) + [
                {"role": "assistant", "content": prompts._dumps({"rotation_index": ri, "anchor_id": label_id})}],
            "kind": "pick", "record": idx, "episode": episode}
    path = {"messages": prompts.path_messages(pos, []) + [
                {"role": "assistant", "content": prompts._dumps({"path": rec["label"]["path"]})}],
            "kind": "path", "record": idx, "episode": episode}
    return [pick, path]


# ------------------------------------------------------------------ main ----

def prepare(in_path: Path = IN_PATH, out_dir: Path = OUT_DIR, shuffle: bool = True,
            split_seed: int = SPLIT_SEED, shuffle_seed: int = SHUFFLE_SEED,
            test_fraction: float = TEST_FRACTION) -> Dict:
    raw = in_path.read_bytes()
    records = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]

    kept: List[Tuple[int, int, Dict, str]] = []   # (idx, episode, state, label_id)
    dropped: Dict[str, List[int]] = {}
    forced_records: List[int] = []
    for idx, rec, episode, placed_before, status in replay(records):
        if rec.get("task") != "pick_and_path":
            dropped.setdefault("not pick_and_path", []).append(idx); continue
        if rec["ts"] < PRE_FIX_CUTOFF_TS:
            dropped.setdefault("recorded before the generator fix (D32)", []).append(idx); continue
        if status != "ok":
            dropped.setdefault("candidate list not reproduced by the recorder (D32)", []).append(idx); continue
        rng = random.Random(f"{shuffle_seed}:{idx}") if shuffle else None
        state, info = packer_view(placed_before, rec, rng)
        if state is None:
            dropped.setdefault(info, []).append(idx); continue
        label_id, forced = info
        if forced:
            forced_records.append(idx)
        kept.append((idx, episode, state, label_id))

    # D38: split by episode
    episodes = sorted({ep for _, ep, _, _ in kept})
    per_ep = Counter(ep for _, ep, _, _ in kept)
    order = list(episodes)
    random.Random(split_seed).shuffle(order)
    target = test_fraction * len(kept)
    test_eps, n_test = set(), 0
    for ep in order:
        if n_test >= target:
            break
        test_eps.add(ep); n_test += per_ep[ep]

    train, test = [], []
    for idx, ep, state, label_id in kept:
        exs = build_examples(state, label_id, records[idx], idx, ep)
        (test if ep in test_eps else train).extend(exs)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    manifest = {
        "input": str(in_path.relative_to(ROOT)) if in_path.is_relative_to(ROOT) else str(in_path),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "input_records": len(records),
        "kept_records": len(kept),
        "dropped_records": {k: v for k, v in dropped.items()},
        "forced_include_records": forced_records,
        "episodes": len(episodes),
        "records_per_episode": {str(ep): per_ep[ep] for ep in episodes},
        "test_episodes": sorted(test_eps),
        "train_records": len(kept) - n_test, "test_records": n_test,
        "train_examples": len(train), "test_examples": len(test),
        "pre_fix_cutoff_ts": PRE_FIX_CUTOFF_TS,
        "top_k": TOP_K, "shuffle_anchors": shuffle, "shuffle_seed": shuffle_seed,
        "split": "by episode", "split_seed": split_seed, "test_fraction": test_fraction,
        "prompt_builder": "training/packer_vendor/prompts.py (verbatim llm-robotic-packer/harness/prompts.py)",
        "anchor_pipeline": "training/packer_vendor/state_manager.py (verbatim llm-robotic-packer/envs/state_manager.py)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=IN_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--no-shuffle", action="store_true", help="keep score order and rank ids (not used for the paper)")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    args = ap.parse_args()
    m = prepare(args.input, args.out, shuffle=not args.no_shuffle,
                split_seed=args.split_seed, shuffle_seed=args.shuffle_seed)
    print(json.dumps({k: v for k, v in m.items() if k not in ("records_per_episode",)}, indent=2))


if __name__ == "__main__":
    main()
