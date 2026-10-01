"""Human-expert evaluation recorder (T3.8, D80): same shortlist as the packer, safe save/resume."""
import json
import random
import subprocess
import sys

import pytest

from lfd.harness_view import build_state, greedy_choice
from lfd.human_session import HumanSession
from tests._paths import packer_repo

PACKER = packer_repo()
SEQ_DIR = PACKER / "data" / "sequences" / "curriculum25"
needs_packer = pytest.mark.skipif(not (SEQ_DIR / "seed0.json").exists(), reason=f"packer repo not found at {PACKER}")


def _random_states(n_episodes=12, seed=7):
    """Bin states reached by random play on the five evaluation sequences and on random boxes."""
    rng = random.Random(seed)
    seqs = [json.loads((SEQ_DIR / f"seed{k}.json").read_text())["boxes"] for k in range(5)]
    seqs += [[[rng.randint(1, 5) for _ in range(3)] for _ in range(30)] for _ in range(n_episodes - 5)]
    out = []
    for boxes in seqs:
        placed = []
        for box in boxes:
            st = build_state(placed, box, [10, 10, 10])
            out.append({"placed": [dict(b) for b in placed], "box": box, "state": st})
            if st["anchors_indexed"]:
                a = rng.choice(st["anchors_indexed"])
                placed.append({"position": a["pos"], "size": st["incoming_box"]["rotations"][a["rotation_index"]]})
    return out


@needs_packer
def test_shortlist_equals_packer_build_state(tmp_path):
    """The recorder offers exactly harness.state.build_state's top-8-per-rotation shortlist (run in the packer)."""
    cases = _random_states()
    assert sum(1 for c in cases if c["state"]["anchors_indexed"]) > 200
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    code = (
        "import json, sys\n"
        "from harness.state import build_state\n"
        "cases = json.load(open(sys.argv[1]))\n"
        "bad = [i for i, c in enumerate(cases) if build_state(c['placed'], c['box'], [10, 10, 10]) != c['state']]\n"
        "print(len(cases), len(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "cases.json")], cwd=PACKER,
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == [str(len(cases)), "0"]


@needs_packer
def test_greedy_autoplay_and_resume(tmp_path):
    seq = SEQ_DIR / "seed1.json"
    full = HumanSession(str(seq), str(tmp_path / "full.json"), operator="t", view="test")
    while full.state is not None:
        a = greedy_choice(full.state)
        full.choose(a["rotation_index"], a["id"])
    rec = json.loads((tmp_path / "full.json").read_text())
    assert rec["complete"] and len(rec["decisions"]) == 25
    assert {d["outcome"] for d in rec["decisions"]} == {"placed", "skipped_no_anchor"}

    # stop after 9 boxes, reopen, finish: identical decisions
    part = HumanSession(str(seq), str(tmp_path / "part.json"), operator="t", view="test")
    while part.index < 9:
        a = greedy_choice(part.state)
        part.choose(a["rotation_index"], a["id"])
    again = HumanSession(str(seq), str(tmp_path / "part.json"), operator="t", view="test")
    assert again.index == 9 and again.placed == part.placed
    while again.state is not None:
        a = greedy_choice(again.state)
        again.choose(a["rotation_index"], a["id"])
    rec2 = json.loads((tmp_path / "part.json").read_text())
    strip = lambda ds: [{k: v for k, v in d.items() if k not in ("decision_time_s", "decided_at")} for d in ds]
    assert strip(rec2["decisions"]) == strip(rec["decisions"])
    assert len(rec2["sessions"]) == 2


@needs_packer
def test_only_offered_anchors_and_tamper_detection(tmp_path):
    seq = SEQ_DIR / "seed0.json"
    s = HumanSession(str(seq), str(tmp_path / "c.json"), operator="t", view="test")
    with pytest.raises(ValueError):
        s.choose(0, "r0_a99")
    a = s.state["anchors_indexed"][0]
    s.choose(a["rotation_index"], a["id"])
    rec = json.loads((tmp_path / "c.json").read_text())
    rec["decisions"][0]["pos"] = [9, 9, 0]
    (tmp_path / "c.json").write_text(json.dumps(rec))
    with pytest.raises(RuntimeError):
        HumanSession(str(seq), str(tmp_path / "c.json"), operator="t", view="test")


def test_refuses_demo_folder(tmp_path):
    import human_eval
    with pytest.raises(SystemExit):
        human_eval.main(["--packer", str(PACKER), "--seed", "0", "--autoplay", "first",
                         "--out", str(tmp_path / "data" / "demos" / "x.json")])
