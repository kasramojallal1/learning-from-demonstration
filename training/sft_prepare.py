import os
import json
from datasets import Dataset, DatasetDict

IN_PATH = "data/demos/binpack_lfd.jsonl"
OUT_DIR = "data/processed"

def build_messages_pick_and_path(state, label):
    # Two messages per example (system is implied during SFT)
    # Keep inputs compact; model learns to emit strict JSON labels.
    user_pick = {
        "bin": state["bin"],
        "incoming_box": state["incoming_box"],
        "anchors_indexed": state["anchors_indexed"]
    }
    assistant_pick = {
        "rotation_index": label["rotation_index"],
        "anchor_id": label["anchor_id"]
    }
    user_path = {"target_pos": None}  # we will derive from anchor_id index
    # Find anchor pos from id
    ridx = label["rotation_index"]
    aid = label["anchor_id"]
    # anchors_indexed contains all; find matching id
    pos = None
    for item in state["anchors_indexed"]:
        if item["id"] == aid:
            pos = item["pos"]
            break
    user_path["target_pos"] = pos
    assistant_path = {"path": label["path"]}
    return (json.dumps(user_pick, separators=(",", ":")),
            json.dumps(assistant_pick, separators=(",", ":")),
            json.dumps(user_path, separators=(",", ":")),
            json.dumps(assistant_path, separators=(",", ":")))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    with open(IN_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("task") != "pick_and_path":
                continue
            state = rec["state"]; label = rec["label"]
            up, ap, ua, aa = build_messages_pick_and_path(state, label)
            records.append({
                "user_pick": up,
                "assistant_pick": ap,
                "user_path": ua,
                "assistant_path": aa
            })
    ds = Dataset.from_list(records)
    # Simple split
    dsd = ds.train_test_split(test_size=0.1, seed=13)
    dsd.save_to_disk(OUT_DIR)
    print(f"Saved processed dataset to {OUT_DIR}")

if __name__ == "__main__":
    main()
