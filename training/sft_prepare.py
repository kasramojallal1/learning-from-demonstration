# training/sft_prepare.py
import json
from pathlib import Path
from datasets import Dataset

# Resolve project root (this file is training/sft_prepare.py)
ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "demos" / "binpack_lfd.jsonl"
OUT_DIR = ROOT / "data" / "processed"

def build_messages_pick_and_path(state, label):
    # Build compact user/assistant pairs for pick and path
    user_pick = {
        "bin": state["bin"],
        "incoming_box": state["incoming_box"],
        "anchors_indexed": state["anchors_indexed"],
    }
    assistant_pick = {
        "rotation_index": label["rotation_index"],
        "anchor_id": label["anchor_id"],
    }

    # derive target_pos from anchor_id
    aid = label["anchor_id"]
    target_pos = None
    for item in state["anchors_indexed"]:
        if item["id"] == aid:
            target_pos = item["pos"]
            break
    if target_pos is None:
        return None  # skip malformed example

    user_path = {"target_pos": target_pos}
    assistant_path = {"path": label["path"]}

    return (
        json.dumps(user_pick, separators=(",", ":")),
        json.dumps(assistant_pick, separators=(",", ":")),
        json.dumps(user_path, separators=(",", ":")),
        json.dumps(assistant_path, separators=(",", ":")),
    )

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Could not find demos file at: {IN_PATH}\n"
            f"(cwd={Path.cwd()})"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    kept, skipped = 0, 0
    with IN_PATH.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("task") != "pick_and_path":
                skipped += 1
                continue
            state = rec.get("state", {})
            label = rec.get("label", {})
            built = build_messages_pick_and_path(state, label)
            if built is None:
                skipped += 1
                continue
            up, ap, ua, aa = built
            records.append({
                "user_pick": up,
                "assistant_pick": ap,
                "user_path": ua,
                "assistant_path": aa,
            })
            kept += 1

    if not records:
        raise RuntimeError("No valid records found. Check your demos file format.")

    ds = Dataset.from_list(records)
    dsd = ds.train_test_split(test_size=max(1, int(0.1 * len(ds))), seed=13)
    dsd.save_to_disk(str(OUT_DIR))
    print(f"✅ Saved processed dataset to {OUT_DIR}")
    print(f"   kept={kept}  skipped={skipped}")

if __name__ == "__main__":
    main()
