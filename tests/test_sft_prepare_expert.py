"""sft_prepare.py on privileged-expert demonstrations (T3.7, D52): explicit episodes, replay from labels, no force-include."""
import gzip
import json
import sys

import pytest

from tests._paths import ROOT

sys.path.insert(0, str(ROOT))

from training import sft_prepare as sp                                         # noqa: E402
from training.packer_vendor.state_manager import (                             # noqa: E402
    filter_anchors_with_clearance, generate_anchor_positions, generate_orientations, topk_anchors)

BIN = {"w": 10, "h": 10, "d": 10}


def _view(placed, size):
    """Unshuffled top-8 view, the way the packer's build_state serializes it."""
    rots = generate_orientations(size)
    anchors = []
    for r, rs in enumerate(rots):
        c = generate_anchor_positions(placed, rs, [10, 10, 10])
        c = filter_anchors_with_clearance(c, rs, placed, [10, 10, 10])
        for j, pos in enumerate(topk_anchors(c, rs, [10, 10, 10], k=8)):
            anchors.append({"id": f"r{r}_a{j}", "rotation_index": r, "pos": list(pos)})
    return {"bin": dict(BIN), "incoming_box": {"original_size": list(size), "rotations": rots}, "anchors_indexed": anchors}


def _episode(name, sizes, pick):
    """Records for one episode; `pick(state)` returns the anchor dict to label."""
    placed, out = [], []
    for k, size in enumerate(sizes):
        st = _view(placed, size)
        a = pick(st)
        rs = st["incoming_box"]["rotations"][a["rotation_index"]]
        out.append({"ts": "2026-09-21T00:00:00+00:00", "task": "pick_and_path", "episode": name, "state": st,
                    "label": {"rotation_index": a["rotation_index"], "anchor_id": a["id"],
                              "path": [[*a["pos"][:2], 12], [*a["pos"][:2], a["pos"][2] + 1], a["pos"]]},
                    "meta": {"source": "expert", "box_index": k}})
        placed.append({"position": a["pos"], "size": rs})
    return out


def _write(tmp_path, records, gz=True):
    p = tmp_path / ("e.jsonl.gz" if gz else "e.jsonl")
    data = "".join(json.dumps(r) + "\n" for r in records).encode()
    p.write_bytes(gzip.compress(data) if gz else data)
    return p


def test_expert_records_are_kept_split_by_episode_and_never_forced(tmp_path):
    recs = []
    for e in range(12):
        recs += _episode(f"expert/data1/seed{1000 + e}", [[10, 10, 5], [5, 5, 5], [2, 3, 4]],
                         lambda st: st["anchors_indexed"][-1])       # last shortlist entry, never rank 0
    m = sp.prepare(_write(tmp_path, recs), tmp_path / "out")
    assert m["sources"] == {"expert": 36} and m["kept_records"] == 36 and m["dropped_records"] == {}
    assert m["forced_include_records"] == [] and m["episodes"] == 12
    assert m["train_examples"] + m["test_examples"] == 72 and m["test_examples"] >= 0.1 * 72
    train = [json.loads(l) for l in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (tmp_path / "out" / "test.jsonl").read_text().splitlines()]
    assert {r["episode"] for r in train}.isdisjoint({r["episode"] for r in test})
    picks = [r for r in train if r["kind"] == "pick"]
    for pick in picks:
        user = json.loads(pick["messages"][1]["content"])
        ans = json.loads(pick["messages"][2]["content"])
        assert any(a["id"] == ans["anchor_id"] and a["rotation_index"] == ans["rotation_index"] for a in user["anchors_indexed"])
    big = [json.loads(p["messages"][1]["content"])["anchors_indexed"] for p in picks]
    big = [a for a in big if len(a) >= 8]
    assert big and any([x["id"] for x in a] != sorted((x["id"] for x in a), key=lambda i: (int(i[1:i.index("_")]), int(i.split("_a")[1]))) for a in big)   # shuffled (D34)


def test_label_outside_shortlist_or_tampered_state_is_dropped(tmp_path):
    good = _episode("expert/x/seed1", [[3, 3, 3], [3, 3, 3]], lambda st: st["anchors_indexed"][0])
    bad_state = _episode("expert/x/seed2", [[3, 3, 3]], lambda st: st["anchors_indexed"][0])
    bad_state[0]["state"]["anchors_indexed"].append({"id": "r0_a99", "rotation_index": 0, "pos": [5, 5, 0]})
    bad_label = _episode("expert/x/seed3", [[3, 3, 3]], lambda st: st["anchors_indexed"][0])
    bad_label[0]["state"]["anchors_indexed"][0]["pos"] = [4, 4, 0]     # feasible floor spot, but not in the top-8
    m = sp.prepare(_write(tmp_path, good + bad_state + bad_label, gz=False), tmp_path / "out")
    assert m["kept_records"] == 2
    assert sorted(i for v in m["dropped_records"].values() for i in v) == [2, 3]


def test_mixed_human_and_expert_file_is_refused(tmp_path):
    recs = _episode("expert/x/seed1", [[3, 3, 3]], lambda st: st["anchors_indexed"][0])
    human = {k: v for k, v in recs[0].items() if k != "episode"}
    human["meta"] = {"util_gain": 27}
    with pytest.raises(ValueError):
        sp.prepare(_write(tmp_path, recs + [human], gz=False), tmp_path / "out")
