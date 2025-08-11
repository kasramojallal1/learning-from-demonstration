# main.py
import os
import random
from datetime import datetime

from lfd.logger import DemoLogger
from envs.bin_env import BinPackingLfDEnv
from utils.geometry import unique_rotations_3d

SEED = random.randint(1, 10000)

# --- Bigger boxes = fewer placements, easier choices ---
PLACEMENTS_TARGET = 8              # how many demos to collect per run
LARGE_BOX_RANGE_DEFAULT = (0.3, 0.6)  # per-axis fraction of bin size


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def new_run_dir() -> str:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join("data", "runs", tag)
    ensure_dir(base)
    ensure_dir(os.path.join(base, "snapshots"))
    return base


def sample_large_box(bin_size, rng, frac_range=LARGE_BOX_RANGE_DEFAULT):
    """Sample a chunky box that still fits the bin (per-axis fraction)."""
    BW, BH, BD = bin_size
    lo, hi = frac_range
    # rng is numpy Generator from env; use its uniform for reproducibility
    w = max(1, int(round(rng.uniform(lo, hi) * BW)))
    h = max(1, int(round(rng.uniform(lo, hi) * BH)))
    d = max(1, int(round(rng.uniform(lo, hi) * BD)))
    return min(w, BW), min(h, BH), min(d, BD)


def has_any_anchor(env) -> bool:
    """True if current env. rotation set yields any valid anchors."""
    arr = getattr(env, "anchors_by_rot", [])
    return any(len(a) > 0 for a in arr)


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

    placed = 0
    placements_target = PLACEMENTS_TARGET

    try:
        while placed < placements_target:
            # --- 1) Sample a large box; if it has no anchors for ANY rotation, auto-shrink and retry
            tried = 0
            frac_hi = LARGE_BOX_RANGE_DEFAULT[1]
            while True:
                box = sample_large_box((env.bin_w, env.bin_h, env.bin_d), env.rng, (LARGE_BOX_RANGE_DEFAULT[0], frac_hi))
                rots = unique_rotations_3d(box)
                env.prepare_incoming_box(original_size=box, rotations=rots)

                if has_any_anchor(env):
                    break  # feasible box found

                tried += 1
                if tried >= 5:
                    print("⛔ No feasible anchors for any rotation after retries — skipping box.")
                    box = None
                    break
                # shrink upper bound and try again
                frac_hi *= 0.85
                print("↩︎ Resampling smaller box (no anchors for any rotation)…")

            if box is None:
                continue  # skip and try another placement

            # --- 2) Interactive UI: R=rotation, A/D=anchor prev/next, click waypoints, Enter=commit, Esc=skip
            accepted = env.interactive_place_one()
            if not accepted:
                print("⏭️ Skipped placement.")
                continue

            # --- 3) Log compact state + label for training
            state = env.export_compact_state()   # bin + incoming_box + anchors_indexed
            label = env.export_last_label()      # {"rotation_index": i, "anchor_id": "...", "path":[...]}
            meta  = env.export_last_meta()       # {"util_gain": ...}

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
