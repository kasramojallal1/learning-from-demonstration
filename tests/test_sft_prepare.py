"""sft_prepare.py: the training view must be the harness's view (D31b-D38)."""
import json
import random
import sys

import pytest

from tests._paths import ROOT, packer_repo   # noqa: E402

sys.path.insert(0, str(ROOT))

from training import sft_prepare as sp   # noqa: E402
from training.packer_vendor import prompts   # noqa: E402

DEMOS = ROOT / "data" / "demos" / "binpack_lfd.jsonl"
PACKER = packer_repo()


@pytest.fixture(scope="module")
def records():
    return [json.loads(l) for l in DEMOS.read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def replayed(records):
    return list(sp.replay(records))


def test_replay_reproduces_every_post_fix_record(replayed):
    bad = [i for i, rec, ep, pb, status in replayed if rec["ts"] >= sp.PRE_FIX_CUTOFF_TS and status != "ok"]
    assert bad == []


def test_episode_count(replayed):
    eps = {ep for i, rec, ep, pb, status in replayed if rec["ts"] >= sp.PRE_FIX_CUTOFF_TS}
    assert len(eps) == 52


def test_label_id_resolves_to_demonstrated_anchor(replayed):
    for i, rec, ep, pb, status in replayed:
        if rec["ts"] < sp.PRE_FIX_CUTOFF_TS:
            continue
        state, (label_id, forced) = sp.packer_view(pb, rec, random.Random(f"x:{i}"))
        ri, size, pos = sp.chosen_anchor(rec)
        hit = [a for a in state["anchors_indexed"] if a["id"] == label_id]
        assert len(hit) == 1 and hit[0]["rotation_index"] == ri and hit[0]["pos"] == pos


def test_top8_and_force_include(replayed):
    n_forced = 0
    for i, rec, ep, pb, status in replayed:
        if rec["ts"] < sp.PRE_FIX_CUTOFF_TS:
            continue
        state, (label_id, forced) = sp.packer_view(pb, rec, None)
        per_rot = {}
        for a in state["anchors_indexed"]:
            per_rot.setdefault(a["rotation_index"], []).append(a)
        ri, size, pos = sp.chosen_anchor(rec)
        for r, lst in per_rot.items():
            assert len(lst) <= sp.TOP_K + (1 if (forced and r == ri) else 0)
            ids = sorted(int(a["id"].split("_a")[1]) for a in lst)
            assert ids == list(range(len(lst)))          # ids are a permutation of 0..n-1
        n_forced += forced
    assert 40 <= n_forced <= 70                             # 53 on the 2026-09-21 data


def test_shuffle_is_deterministic_and_changes_order(replayed):
    i, rec, ep, pb, status = next(x for x in replayed if x[1]["ts"] >= sp.PRE_FIX_CUTOFF_TS and len(x[1]["state"]["anchors_indexed"]) > 20)
    a = sp.packer_view(pb, rec, random.Random("s:1"))[0]["anchors_indexed"]
    b = sp.packer_view(pb, rec, random.Random("s:1"))[0]["anchors_indexed"]
    c = sp.packer_view(pb, rec, None)[0]["anchors_indexed"]
    key = lambda lst: sorted((x["rotation_index"], tuple(x["pos"])) for x in lst)
    assert a == b                      # same seed -> same permutation
    assert a != c                      # order/ids differ from the score order
    assert key(a) == key(c)            # same anchors, only order and ids change


def test_view_matches_packer_harness(replayed):
    """Unshuffled training view == harness.state.build_state on the same bin (when the packer repo is present)."""
    if not (PACKER / "harness" / "state.py").exists():
        pytest.skip("packer repo not found")
    sys.path.insert(0, str(PACKER))
    for mod in list(sys.modules):
        if mod == "envs" or mod.startswith("envs."):
            del sys.modules[mod]
    from harness.state import build_state
    n = 0
    for i, rec, ep, pb, status in replayed:
        if rec["ts"] < sp.PRE_FIX_CUTOFF_TS:
            continue
        state, (label_id, forced) = sp.packer_view(pb, rec, None)
        ref = build_state([{"position": b["pos"], "size": b["size"]} for b in pb],
                          rec["state"]["incoming_box"]["original_size"], [10, 10, 10])
        if forced:
            ri, size, pos = sp.chosen_anchor(rec)
            ours = [a for a in state["anchors_indexed"] if not (a["rotation_index"] == ri and a["pos"] == pos)]
            assert ours == ref["anchors_indexed"]
        else:
            assert state == ref
        n += 1
    assert n == 646


def test_messages_come_from_the_vendored_builder(replayed):
    i, rec, ep, pb, status = next(x for x in replayed if x[1]["ts"] >= sp.PRE_FIX_CUTOFF_TS)
    state, (label_id, forced) = sp.packer_view(pb, rec, None)
    pick, path = sp.build_examples(state, label_id, rec, i, ep)
    assert pick["messages"][0]["content"] == prompts.SYSTEM_PICK
    assert pick["messages"][1]["content"] == prompts.pick_user(state, [])
    assert json.loads(pick["messages"][2]["content"]) == {"rotation_index": rec["label"]["rotation_index"], "anchor_id": label_id}
    assert path["messages"][0]["content"] == prompts.SYSTEM_PATH
    assert json.loads(path["messages"][1]["content"]) == {"target_pos": sp.chosen_anchor(rec)[2]}
    assert json.loads(path["messages"][2]["content"]) == {"path": rec["label"]["path"]}


def test_prepare_end_to_end(tmp_path):
    m = sp.prepare(DEMOS, tmp_path)
    assert m["input_records"] == 667 and m["kept_records"] == 646
    assert m["dropped_records"] == {"recorded before the generator fix (D32)": list(range(21))}
    assert m["train_examples"] == 2 * m["train_records"] and m["test_examples"] == 2 * m["test_records"]
    assert m["test_records"] >= 0.10 * m["kept_records"]
    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (tmp_path / "test.jsonl").read_text().splitlines()]
    assert set(r["episode"] for r in train).isdisjoint(set(r["episode"] for r in test))
    assert (tmp_path / "manifest.json").exists()
