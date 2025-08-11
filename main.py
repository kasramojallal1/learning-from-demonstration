import os
import json
import random
from datetime import datetime
from lfd.logger import DemoLogger
from envs.bin_env import BinPackingLfDEnv
from utils.geometry import unique_rotations_3d

SEED = 42
random.seed(SEED)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def new_run_dir():
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join("data", "runs", tag)
    ensure_dir(base)
    ensure_dir(os.path.join(base, "snapshots"))
    return base

def main():
    run_dir = new_run_dir()
    print(f"🎬 LfD run dir: {run_dir}")

    # Env with interactive UI
    env = BinPackingLfDEnv(
        bin_size=(10, 10, 10),
        run_dir=run_dir,
        rng_seed=SEED
    )

    # Where demonstrations get written
    demo_path = os.path.join("data", "demos", "binpack_lfd.jsonl")
    ensure_dir(os.path.dirname(demo_path))
    logger = DemoLogger(demo_path)

    # Simple curriculum: 20 placements per episode
    placements_target = 20
    placed = 0

    # Initial render loop handled by env.render(); it blocks until user acts
    try:
        while placed < placements_target:
            # 1) Sample incoming box (smaller first, gradually bigger)
            max_edge = 5 if placed < 10 else 7
            w = random.randint(1, max_edge)
            h = random.randint(1, max_edge)
            d = random.randint(1, max_edge)
            box = (w, h, d)
            rots = unique_rotations_3d(box)

            # 2) Set new incoming box in env; env computes anchors per rotation
            env.prepare_incoming_box(original_size=box, rotations=rots)

            # 3) Interactive UI: user cycles rotation (R), anchor (A/D), clicks waypoints, Enter to commit
            accepted = env.interactive_place_one()

            if not accepted:
                print("⏭️ Skipped placement.")
                continue

            # 4) Collect compact state + label from env for logging
            state = env.export_compact_state()  # bin + incoming_box + anchors_indexed
            label = env.export_last_label()     # {"rotation_index": i, "anchor_id": "...", "path":[...]}
            meta  = env.export_last_meta()      # {"util_gain":..., ...}

            logger.log_example(
                task="pick_and_path",
                state=state,
                label=label,
                meta=meta
            )

            placed += 1
            print(f"✅ Demo {placed}/{placements_target} recorded.")

        print(f"🎉 Finished {placements_target} demonstrations.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
