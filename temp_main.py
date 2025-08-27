# main_datasets.py  (Learning-from-Demonstration)
import os
import random
import argparse
from datetime import datetime
from typing import List, Tuple

from lfd.logger import DemoLogger
from envs.bin_env import BinPackingLfDEnv
from utils.geometry import unique_rotations_3d

# ------------------------- Defaults / Seed -------------------------
SEED = random.randint(1, 10_000)
PLACEMENTS_TARGET_DEFAULT = 20
LARGE_BOX_RANGE_DEFAULT = (0.2, 0.5)  # per-axis fraction (our original style)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def new_run_dir() -> str:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join("data", "runs", tag)
    ensure_dir(base)
    ensure_dir(os.path.join(base, "snapshots"))
    return base

# ------------------------- OUR (original) sampler -------------------------
def sample_large_box(bin_size, rng, frac_range=LARGE_BOX_RANGE_DEFAULT):
    BW, BH, BD = bin_size
    lo, hi = frac_range
    w = max(1, int(round(rng.uniform(lo, hi) * BW)))
    h = max(1, int(round(rng.uniform(lo, hi) * BH)))
    d = max(1, int(round(rng.uniform(lo, hi) * BD)))
    return min(w, BW), min(h, BH), min(d, BD)

def gen_sequence_ours(env, n_items: int) -> List[Tuple[int,int,int]]:
    """Generate n_items boxes using our feasibility-aware style (attempt shrink if no anchors)."""
    seq = []
    frac_hi0 = LARGE_BOX_RANGE_DEFAULT[1]
    for _ in range(n_items):
        tried, frac_hi = 0, frac_hi0
        while True:
            box = sample_large_box((env.bin_w, env.bin_h, env.bin_d), env.rng, (LARGE_BOX_RANGE_DEFAULT[0], frac_hi))
            rots = unique_rotations_3d(box)
            env.prepare_incoming_box(original_size=box, rotations=rots)
            # anchors_by_rot is populated by env; feasible if any rotation has anchors
            feasible = any(len(a) > 0 for a in getattr(env, "anchors_by_rot", []))
            if feasible:
                seq.append(box)
                break
            tried += 1
            if tried >= 5:
                # give up on this item; move on (keeps the run flowing)
                break
            frac_hi *= 0.85  # shrink and retry
    return seq

# ------------------------- PAPER dataset samplers -------------------------
def _vol(s: Tuple[int,int,int]) -> int:
    return s[0]*s[1]*s[2]

def _template_sizes_1to5() -> List[Tuple[int,int,int]]:
    sizes = []
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                if [a,b,c].count(1) <= 1:
                    sizes.append((a,b,c))
    return sizes

def _balanced_sample_templates(k: int, rng: random.Random) -> List[Tuple[int,int,int]]:
    sizes = sorted(_template_sizes_1to5(), key=_vol)
    n = len(sizes)
    t1, t2, t3 = sizes[: n//3], sizes[n//3 : 2*n//3], sizes[2*n//3 :]
    k1, k2, k3 = int(round(0.30*k)), int(round(0.40*k)), k - int(round(0.30*k)) - int(round(0.40*k))
    out = []
    if t1: out += rng.sample(t1, min(k1, len(t1)))
    if t2: out += rng.sample(t2, min(k2, len(t2)))
    remain3 = [s for s in t3 if s not in out]
    need = k - len(out)
    if need > 0 and remain3:
        out += rng.sample(remain3, min(need, len(remain3)))
    # top up if short
    remain = [s for s in sizes if s not in out]
    while len(out) < k and remain:
        out.append(remain.pop(0))
    return out[:k]

def gen_sequence_data1(env, n_items: int, seed: int) -> List[Tuple[int,int,int]]:
    """DATA-1: per-axis in [2, L/2], shuffled."""
    rng = random.Random(seed)
    L, W, H = env.bin_w, env.bin_h, env.bin_d
    hiL, hiW, hiH = max(2, L//2), max(2, W//2), max(2, H//2)
    seq = [(rng.randint(2, hiL), rng.randint(2, hiW), rng.randint(2, hiH)) for _ in range(n_items)]
    rng.shuffle(seq)
    return seq

def gen_sequence_data2(env, n_items: int, seed: int) -> List[Tuple[int,int,int]]:
    """DATA-2: draw from 64-size catalog (1..5, ≤1 side==1), random order."""
    rng = random.Random(seed)
    templates = _balanced_sample_templates(64, rng)
    seq = [rng.choice(templates) for _ in range(n_items)]
    rng.shuffle(seq)
    return seq

def gen_sequence_data3(env, n_items: int, seed: int) -> List[Tuple[int,int,int]]:
    """DATA-3: same catalog as DATA-2 but sorted by decreasing volume."""
    rng = random.Random(seed ^ 1337)
    templates = _balanced_sample_templates(64, rng)
    seq = [rng.choice(templates) for _ in range(n_items)]
    seq.sort(key=_vol, reverse=True)
    return seq

# ------------------------- Core LfD loop (unchanged behavior) -------------------------
def collect_demos(env: BinPackingLfDEnv, demo_logger: DemoLogger, sequence: List[Tuple[int,int,int]], placements_target: int):
    placed = 0
    for box in sequence:
        if placed >= placements_target:
            break

        rots = unique_rotations_3d(box)
        env.prepare_incoming_box(original_size=box, rotations=rots)

        # If absolutely no anchors for any rotation (possible for paper boxes), skip this item
        if not any(len(a) > 0 for a in getattr(env, "anchors_by_rot", [])):
            print("⛔ No feasible anchors for any rotation — skipping this item.")
            continue

        accepted = env.interactive_place_one()
        if not accepted:
            print("⏭️ Skipped placement.")
            continue

        state = env.export_compact_state()
        label = env.export_last_label()
        meta  = env.export_last_meta()

        demo_logger.log_example(task="pick_and_path", state=state, label=label, meta=meta)
        placed += 1
        print(f"✅ Demo {placed}/{placements_target} recorded.")

# ------------------------- CLI & main -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LfD demo collector with selectable box generators (ours or paper datasets).")
    p.add_argument("--mode", choices=["ours", "paper:data1", "paper:data2", "paper:data3"], default="ours",
                   help="Box generation scheme")
    p.add_argument("--bin", type=int, default=10, help="Cubic bin edge (10 => 10×10×10)")
    p.add_argument("--placements", type=int, default=PLACEMENTS_TARGET_DEFAULT, help="Target number of demos")
    p.add_argument("--seed", type=int, default=None, help="Random seed (defaults to a new random)")
    return p.parse_args()

def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else SEED
    run_dir = new_run_dir()
    print(f"🎬 LfD run dir: {run_dir}")

    env = BinPackingLfDEnv(bin_size=(args.bin, args.bin, args.bin), run_dir=run_dir, rng_seed=seed)

    demo_path = os.path.join("data", "demos", "binpack_lfd.jsonl")
    ensure_dir(os.path.dirname(demo_path))
    logger = DemoLogger(demo_path)

    # Build the incoming sequence according to the selected mode
    if args.mode == "ours":
        sequence = gen_sequence_ours(env, args.placements * 2)  # oversample a bit to allow skips
    elif args.mode == "paper:data1":
        sequence = gen_sequence_data1(env, args.placements * 2, seed)
    elif args.mode == "paper:data2":
        sequence = gen_sequence_data2(env, args.placements * 2, seed)
    else:  # paper:data3
        sequence = gen_sequence_data3(env, args.placements * 2, seed)

    print(f"📦 Sequence ready: {len(sequence)} items | mode={args.mode} | bin={(args.bin,args.bin,args.bin)} | seed={seed}")

    try:
        collect_demos(env, logger, sequence, placements_target=args.placements)
        print(f"🎉 Finished {args.placements} demonstrations.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
