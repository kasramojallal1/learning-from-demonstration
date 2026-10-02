# Retraining the Packi planner on RunPod (T0.10)

Recipe and run log for the Llama 3.2 3B LoRA adapter used in the revised paper.
Every step below was run in this order; the numbers in the "Run log" section are
filled in from the files the run produced, never typed from memory.

## Pod

| Item | Value |
|---|---|
| Provider / GPU | RunPod, 1 × NVIDIA RTX 4090 (24 GB) — D39 |
| Template | RunPod PyTorch (CUDA 12.x) |
| Disk | ≥ 40 GB volume (base model 6.4 GB, checkpoints ~0.3 GB, packer results) |
| Access | web terminal or SSH; Kasra creates the pod and enters the Hugging Face token himself |

## 0. Check the host CPU before anything else (D43)

RunPod hosts vary; a throttled/oversubscribed CPU makes LLM decoding 10× slower
(kernel-launch bound) even when the GPU is fine. Run this first and stop the pod
if the numbers are bad:

```bash
python -c "
import time; t=time.time(); s=0
for i in range(10_000_000): s+=i
print('python 10M-loop: %.2fs (healthy 0.4-0.7 s)' % (time.time()-t))
import torch; x=torch.randn(1,64,device='cuda'); torch.cuda.synchronize(); t=time.time()
for _ in range(2000): x=x+1
torch.cuda.synchronize(); print('kernel launch: %.0f us (healthy 5-15 us)' % ((time.time()-t)/2000*1e6))"
```

2026-09-21: training pod `dbchl4u4tv7xro` measured 5.95 s / 104 µs (bad);
evaluation pod `7krubot06c0pw3` measured 0.68 s / 14 µs (good).

## 1. Code and data

```bash
cd /workspace
git clone https://github.com/kasramojallal1/learning-from-demonstration.git
cd learning-from-demonstration
git checkout <branch-or-tag>           # the commit is recorded by the trainer in train_config.json
pip install -r training/requirements-train.txt
hf auth login                          # gated meta-llama/Llama-3.2-3B-Instruct (or: export HF_TOKEN=...)
python -m pytest tests/ -q             # vendored prompt/anchor code + data pipeline checks
python training/sft_prepare.py         # -> data/processed_v2/{train,test}.jsonl + manifest.json
```

`sft_prepare.py` is deterministic (split seed 13, shuffle seed 2026); its
`manifest.json` must show `kept_records: 646`, `train_examples: 1156`,
`test_examples: 136` for the 2026-09-21 demo file
(sha256 `1cbdf4216d80…`).

## 2. Train

Smoke test first (2 optimizer steps, ~2 min, throw-away output):

```bash
MAX_STEPS=2 OUTPUT_DIR=/tmp/smoke python training/train_lora_v2.py && ls /tmp/smoke
```

Then the real run:

```bash
mkdir -p checkpoints/lfd-lora-llama32-3b-v2
python training/train_lora_v2.py 2>&1 | tee checkpoints/lfd-lora-llama32-3b-v2/train.log
```

Defaults = the paper recipe (D37, D42): r 16, α 32, dropout 0.05, targets
q/k/v/o/gate/up/down, AdamW lr 2e-4, cosine, warmup 3 %, grad-clip 1.0,
batch 2 × accum 32 (effective 64), 3 epochs, max_len 2048 (no truncation),
bf16, no quantization, seed 42, loss on assistant tokens only, chat-template
date pinned to "26 Jul 2024".  Expected: 1156 examples / 64 → 19 steps per
epoch, 57 optimizer steps.

Outputs in `checkpoints/lfd-lora-llama32-3b-v2/`: `adapter_model.safetensors`,
`adapter_config.json`, tokenizer files, `train_config.json` (all
hyperparameters, data hashes, git commit, step count, wall time, final eval
loss), `versions.json` (Python, torch, transformers, peft, CUDA, driver, GPU),
`trainer_state.json` (loss per step), `train.log`.

## 3. Archive the adapter (D40)

```bash
hf upload kasramojallal/packi-llama32-3b-lora-v2 checkpoints/lfd-lora-llama32-3b-v2 . --private
```

Locally, pull it into both repos (adapters are git-ignored):

```bash
hf download kasramojallal/packi-llama32-3b-lora-v2 --local-dir learning-from-demonstration/checkpoints/lfd-lora-llama32-3b-v2
hf download kasramojallal/packi-llama32-3b-lora-v2 --local-dir llm-robotic-packer/models/llama32-3b-v2
```

## 4. Evaluate with the harness (packer repo, same pod)

```bash
cd /workspace
git clone https://github.com/kasramojallal1/llm-robotic-packer.git
cd llm-robotic-packer
git checkout <branch-or-tag>           # must contain config.LORA_DIR = models/llama32-3b-v2 (D41)
pip install numpy openai python-dotenv
hf download kasramojallal/packi-llama32-3b-lora-v2 --local-dir models/llama32-3b-v2
python -m pytest tests/ -q

python evaluate.py --method packi      --all --quiet
python evaluate.py --method packi      --all --quiet --shuffle-anchors
python evaluate.py --method packi      --all --quiet --no-feedback
python evaluate.py --method base-llama --all --quiet
python evaluate.py --method base-llama --all --quiet --shuffle-anchors
python aggregate.py
```

Each run JSON records the git commit, GPU name, torch version, and per-call
latency (mean / median / p95) — the numbers the paper quotes for hardware and
latency.  Bring `results/` back the same way as the adapter
(`hf upload kasramojallal/packi-llama32-3b-lora-v2 results results --private`),
then commit `results/` from the Mac (D27).

## Run log — 2026-09-21

| Item | Value |
|---|---|
| Pod | RunPod Secure Cloud, id `dbchl4u4tv7xro`, 1 × NVIDIA GeForce RTX 4090 (24 GB, 23.5 GiB usable), 16 vCPU, 41 GB RAM, 50 GB container disk, $0.75/h |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Versions (`versions.json`) | Python 3.12.3, torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.0, datasets 5.0.1, accelerate 1.15.0, CUDA 12.8, cuDNN 91002, driver 595.91.07, Linux 6.8.0-138 |
| LfD commit | `b5fa8a2` (branch `claude/loving-wilbur-121164`), clean tree |
| Data | `sft_prepare.py` on the pod reproduced the Mac output exactly: 646 kept, 1156 / 136 chats, input sha256 `1cbdf421…`; train.jsonl sha256 `dbd80d47…`, test.jsonl `8c0c6466…` |
| Token lengths (pod, Llama tokenizer) | train chats 252–1440 tokens (median 402); label tokens 15–24 |
| Trainable params | 24,313,856 of 3,237,063,680 (0.75 %) |
| Smoke test (`MAX_STEPS=2`) | passed: step losses 0.592, 0.596; eval loss 0.330; 223 s |
| Note | transformers 5 renamed `warmup_ratio` → `warmup_steps` (float = ratio); the script handles both (commit `b5fa8a2`) |
| Full run | 57 optimizer steps (19 / epoch), 1 h 22 min wall (4947.5 s), ~87 s per step; peak GPU memory ~15 GB |
| Loss | train: 0.59 (step 1) → mean 0.062 (epoch 2) → mean 0.052 (epoch 3); validation: 0.0718 (epoch 1), 0.0566 (epoch 2), **0.0557** (epoch 3, final) |
| Adapter | `checkpoints/lfd-lora-llama32-3b-v2/adapter_model.safetensors` (97,307,544 bytes), archived at `https://huggingface.co/kasramojallal/packi-llama32-3b-lora-v2` (private, commit `1507c0a0`) together with `train_config.json`, `versions.json`, `trainer_state.json`, `train.log` |
| Packer | adapter copied to `llm-robotic-packer/models/llama32-3b-v2/`; harness commit `41b548e` (branch `claude/t0.10-retrain`) |
| Evaluation pod (D43) | `7krubot06c0pw3`: 1 × NVIDIA GeForce RTX 5090 (32 GB), driver 570.195.03, AMD EPYC 7543 host, 15 vCPU, 62 GB RAM, $0.99/h; same image and Python stack as the training pod. Chosen because no RTX 4090 was available and the training pod's host CPU was throttled. Pick call 0.77 s median, path 1.2 s |
| Harness runs (evaluation pod, 100 runs, 2026-09-21 16:55–21:53 UTC) | packer branch `claude/t0.10-retrain`: `packi` plain / `--shuffle-anchors` / `--no-feedback` (commit `8503011`), `base-llama` plain / `--shuffle-anchors` (commit `71bb3dc`); run JSONs record harness commit `4e07438` (the five plain curriculum25 packi runs: `41b548e`, identical sequence files) |
| Cost | training pod ≈ $3.5 (incl. the aborted slow evaluation), evaluation pod ≈ $5 |

### Results (utilization, mean ± std over seeds 0–4; `python aggregate.py`)

| method | variant | curriculum25 | data1 | data2 | data3 |
|---|---|---|---|---|---|
| packi (v2 adapter) | plain | 0.731 ± 0.067 | 0.650 ± 0.071 | 0.757 ± 0.042 | 0.686 ± 0.080 |
| packi | shuffled ids | 0.674 ± 0.070 | 0.670 ± 0.026 | 0.731 ± 0.039 | 0.672 ± 0.107 |
| packi | no feedback | = plain (no retry ever occurred) | | | |
| base-llama (no adapter) | plain | 0.360 ± 0.084 | 0.256 ± 0.086 | 0.243 ± 0.055 | 0.290 ± 0.035 |
| base-llama | shuffled ids | 0.442 ± 0.043 | 0.377 ± 0.046 | 0.416 ± 0.057 | 0.376 ± 0.031 |
| greedy (no LLM) | | 0.707 ± 0.037 | 0.727 ± 0.060 | 0.828 ± 0.054 | 0.689 ± 0.017 |
| random | | 0.630 ± 0.030 | 0.600 ± 0.047 | 0.648 ± 0.099 | 0.589 ± 0.036 |

Reliability: packi — first-attempt validity 1.00, 0 invalid JSON, 0 retries, 0
path collisions on all 60 runs; 1.1–1.5 s per box (pick + path) on the RTX 5090.
base-llama — first-attempt validity 0.13–0.19, 4–5 retries per box, 58–97
invalid-JSON outputs and 17–36 path collisions per run, 1.8–2.3 s per box.

---

# Packi-E: the adapter trained on privileged-expert demonstrations (T3.7, D52)

Same recipe as above; only the teacher changes.  The demonstrations come from the
packer repo's beam-search expert (`llm-robotic-packer/harness/expert.py`, width
1000, top-8 shortlist, never worse than greedy) on 1,000 fresh training sequences
(4 datasets × generator seeds 1000–1249; evaluation seeds 0–4 are refused by the
generator).  They are committed here as `data/demos/expert_beam1000.jsonl.gz`
with `data/demos/expert_beam1000.manifest.json` (per-episode expert and greedy
utilization, packer commit, sha256).

## E0. Pod

RTX 5090 (32 GB, $0.99/h) — chosen for step speed (Kasra, 2026-09-21).  Run the
CPU benchmark of step 0 first; stop the pod if it is throttled.

## E1. Data

```bash
python -m pytest tests/ -q
python training/sft_prepare.py --input data/demos/expert_beam1000.jsonl.gz --out data/processed_e
```

`manifest.json` must show `sources: {"expert": N}` with N = the demo file's record
count, `dropped_records: {}`, `forced_include_records: []` (every expert label is
inside the shortlist by construction).  `data/processed_e/` is git-ignored (it is
~60k chats and is reproduced deterministically from the committed demo file).

## E2. Measure the step time before committing to the run (D52 amendment)

```bash
MAX_STEPS=5 DATA_DIR=data/processed_e OUTPUT_DIR=/tmp/smoke_e python training/train_lora_v2.py
```

Read seconds/step from the log; planned steps = 3 × ceil(train_examples / 64).
Kasra approves hours and dollars before E3 starts.

## E3. Train

```bash
mkdir -p checkpoints/lfd-lora-llama32-3b-e
DATA_DIR=data/processed_e OUTPUT_DIR=checkpoints/lfd-lora-llama32-3b-e \
  python training/train_lora_v2.py 2>&1 | tee checkpoints/lfd-lora-llama32-3b-e/train.log
```

## E4. Archive (D40) and evaluate

```bash
hf upload kasramojallal/packi-llama32-3b-lora-e checkpoints/lfd-lora-llama32-3b-e . --private
# packer repo, same pod (config.LORA_DIR_E = models/llama32-3b-e):
hf download kasramojallal/packi-llama32-3b-lora-e --local-dir models/llama32-3b-e
python evaluate.py --method packi-e --all --quiet
python evaluate.py --method packi-e --all --quiet --shuffle-anchors
python aggregate.py
```

## Run log — Packi-E (2026-09-21/22)

| Item | Value |
|---|---|
| Pod | RunPod Secure Cloud (EUR-NO-1), id `vk78ra9hkypn4g`, name `packi-e-5090`, 1 × RTX 5090 (32 GB, 31.4 GiB usable), 15 vCPU (AMD EPYC 7543), 117 GB RAM, 50 GB container disk, $0.99/h; image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| CPU benchmark (step 0) | Python 10M-loop 0.97 s, kernel launch 16 µs — healthy (the throttled 4090 pod of D43 measured 5.95 s / 104 µs) |
| Versions (`versions.json`) | Python 3.12.3, torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.0, datasets 5.0.1, accelerate 1.15.0, CUDA 12.8, cuDNN 91002, driver 570.195.03, Linux 6.14.0-35 |
| LfD commit | `65146b5` (branch `claude/t3.7-packi-e`), clean tree |
| Data | `sft_prepare.py --input data/demos/expert_beam1000.jsonl.gz` on the pod reproduced the Mac output exactly: 24,753 kept, 0 dropped, 0 force-included, 1,000 episodes (101 held out), **44,520 train / 4,986 test chats**; train.jsonl sha256 `bda40ac1…`, test.jsonl `993bfb45…`, demo file `d30dd5e5…` |
| Token lengths (pod) | train chats 252–1,419 tokens (median 402); label tokens 15–24 — same scale as the human set, no truncation at max_len 2048 |
| Trainable params | 24,313,856 of 3,237,063,680 (0.75 %) |
| Timing test (`MAX_STEPS=5`) | 13.6 s/step, eval over 4,986 chats 250 s; losses 0.602 → 0.130, eval 0.114. Basis for Kasra's approval of 3 epochs (2,088 steps ≈ 8 h ≈ $8) |
| Full run | 2,088 optimizer steps (696 / epoch), **8 h 13 min** (29,600 s), 14.2 s/step; started 2026-09-22 01:04 UTC, finished 09:17 UTC; peak GPU memory ~17 GB |
| Loss | train 0.602 (step 1) → 0.0247 (end of epoch 1) → mean 0.0216 (epoch 3) → 0.0174 (last step); **validation 0.0300 / 0.0253 / 0.0243** (epochs 1–3). Packi-H for contrast: final validation 0.0557 on 1,156 chats |
| Adapter | `checkpoints/lfd-lora-llama32-3b-e/adapter_model.safetensors` (97,307,544 bytes — same shape as v2), archived at `https://huggingface.co/kasramojallal/packi-llama32-3b-lora-e` (private) with `train_config.json`, `versions.json`, `trainer_state.json`, `train.log` |
| Evaluation | same pod, packer commit `6160690` (clean), `models/llama32-3b-e/`; 40 runs (4 datasets × seeds 0–4 × {plain, `--shuffle-anchors`}) 09:17–10:06 UTC; `--no-feedback` not run (Packi-E never retried, so it is identical to plain) |
| Unattended chain | `/workspace/after_train.sh` ran train → HF upload → 40 evaluations → HF upload of `results/packi-e` without supervision; log kept in the session scratchpad |
| Cost | ≈ $12.4 for 12.5 h (8.2 h training, 0.8 h evaluation, ~3 h idle before the pod was stopped at 13:15 UTC); balance $15.43 → $2.99 |

### Results (utilization, mean ± std over seeds 0–4; `python aggregate.py` in the packer repo)

| method | teacher | curriculum25 | data1 | data2 | data3 |
|---|---|---|---|---|---|
| random | — | 0.630 | 0.600 | 0.648 | 0.589 |
| greedy (no LLM) | — | 0.707 ± 0.042 | 0.727 ± 0.067 | 0.828 ± 0.060 | 0.689 ± 0.019 |
| base-llama (no adapter) | — | 0.360 | 0.256 | 0.243 | 0.290 |
| Packi-H (v2 adapter) | 646 human demos | 0.731 ± 0.075 | 0.650 ± 0.080 | 0.757 ± 0.047 | 0.686 ± 0.089 |
| **Packi-E (this run)** | **24,753 expert demos** | **0.789 ± 0.103** | **0.709 ± 0.055** | **0.791 ± 0.023** | **0.750 ± 0.058** |
| Packi-E, shuffled ids | | 0.802 ± 0.064 | 0.726 ± 0.056 | 0.802 ± 0.020 | 0.725 ± 0.041 |
| oracle (beam 1000, sees the future) | — | 0.846 ± 0.083 | 0.848 ± 0.061 | 0.898 ± 0.025 | 0.793 ± 0.065 |

Packi-E reliability: first-attempt validity 1.00, 0 invalid JSON, 0 retries, 0 path
collisions on all 40 runs; 1.07–1.58 s per box on the RTX 5090 (p95 1.44–2.29 s).

---

# Second backbone: Qwen3-4B-Instruct-2507 (T9.2; R1.9; D65–D68)

Same recipe, same data files, same prompts, same harness; only `MODEL_NAME`
changes.  Two adapters (human → `qwen3-4b-h`, expert → `qwen3-4b-e`) plus the
unadapted base (`base-qwen3-4b`).  Qwen's chat template has no date, so the
pinned `DATE_STRING` is passed but unused; its turn terminator `<|im_end|>` is
the tokenizer's `eos_token`, so the harness stops on it without any change.
Verified on the Mac before the run: all 1,292 human and 49,506 expert chats
tokenize with the prompt as an exact prefix, 232–1,422 tokens, labels 16–27.

Pods (D66): **A** = RTX 5090 (human training + every evaluation, the card the
Packi-H/E latencies come from); **B** = H100-class (expert training only).
Both: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, step 0 CPU
benchmark first, then the stack pinned to the Packi runs:

```bash
pip install transformers==5.17.0 peft==0.21.0 datasets==5.0.1 accelerate==1.15.0 "huggingface_hub>=0.30" pytest
```

## Q1. Data (both pods)

```bash
python -m pytest tests/ -q
python training/sft_prepare.py                                                   # human  -> data/processed_v2
python training/sft_prepare.py --input data/demos/expert_beam1000.jsonl.gz --out data/processed_e   # expert (pod B)
sha256sum data/processed_v2/*.jsonl data/processed_e/*.jsonl   # train/test: dbd80d47… / 8c0c6466… ; bda40ac1… / 993bfb45…
```

## Q2. Train

```bash
export MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
# pod A, human
MAX_STEPS=2 OUTPUT_DIR=/tmp/smoke python training/train_lora_v2.py
mkdir -p checkpoints/lfd-lora-qwen3-4b-h
OUTPUT_DIR=checkpoints/lfd-lora-qwen3-4b-h python training/train_lora_v2.py 2>&1 | tee checkpoints/lfd-lora-qwen3-4b-h/train.log
# pod B, expert: timing test first, Kasra approves hours/dollars, then the run
MAX_STEPS=5 DATA_DIR=data/processed_e OUTPUT_DIR=/tmp/smoke_e python training/train_lora_v2.py
mkdir -p checkpoints/lfd-lora-qwen3-4b-e
DATA_DIR=data/processed_e OUTPUT_DIR=checkpoints/lfd-lora-qwen3-4b-e python training/train_lora_v2.py 2>&1 | tee checkpoints/lfd-lora-qwen3-4b-e/train.log
```

## Q3. Archive (D40) and evaluate (pod A, packer branch `claude/e2-qwen3`)

```bash
hf upload kasramojallal/packi-qwen3-4b-lora-h checkpoints/lfd-lora-qwen3-4b-h . --private
hf upload kasramojallal/packi-qwen3-4b-lora-e checkpoints/lfd-lora-qwen3-4b-e . --private
# packer repo: adapters live in models/<alias>/ (git-ignored)
hf download kasramojallal/packi-qwen3-4b-lora-h --local-dir models/qwen3-4b-h
hf download kasramojallal/packi-qwen3-4b-lora-e --local-dir models/qwen3-4b-e
for m in base-qwen3-4b qwen3-4b-h qwen3-4b-e; do
  python evaluate.py --method $m --all --quiet
  python evaluate.py --method $m --all --quiet --shuffle-anchors
done
python aggregate.py results/
```

## Run log — Qwen3-4B (2026-10-01)

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507`, snapshot `cdbee75f17c0…`, 4,022,468,096 parameters (Llama 3.2 3B: 3,212,749,824) |
| Pod B (expert training, D69) | RunPod Secure Cloud US-MO-1, id `8nddwrkcz2ui6e`, name `e2-qwen3-h100`, 1 × H100 80GB HBM3 (79.2 GiB usable), Intel Xeon Platinum 8470, $3.49/h; CPU benchmark 0.50 s / 13 µs (healthy) |
| Pod A (human training + every evaluation, D69) | id `vazgsmy8cczz3p`, name `e2-qwen3-pro4500`, 1 × RTX PRO 4500 Blackwell (32 GB, 31.4 GiB usable), AMD EPYC 7713, $0.72/h; CPU benchmark 0.79 s / 16 µs (healthy). RTX 5090 and 4090 could not be deployed that day |
| Versions (`versions.json`, both pods) | Python 3.12.3, torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.0, datasets 5.0.1, accelerate 1.15.0, CUDA 12.8, cuDNN 91002, driver 580.126.09 — the Packi stack, pinned; only GPU and driver differ |
| LfD commit | `7fff494` (branch `claude/charming-khayyam-0bd7ac`), clean tree for both runs |
| Data | regenerated on each pod, sha256 identical to the Packi runs: human train/test `dbd80d47…`/`8c0c6466…` (1,156 / 136 chats), expert `bda40ac1…`/`993bfb45…` (44,520 / 4,986 chats) |
| Token lengths (Qwen tokenizer) | 232–1,422 tokens (median 384); label tokens 16–27 |
| Trainable params | **33,030,144** of 4,055,498,240 (0.81 %) — Llama: 24,313,856 (0.75 %); same seven target modules |
| Qwen-H training | 57 steps, 20 min 39 s (1,239 s, ~21 s/step) on pod A; train loss 2.26 (step 1) → 0.056 (step 57); **validation 0.0560 / 0.0489 / 0.0462** (Packi-H: 0.0718 / 0.0566 / 0.0557) |
| Qwen-E timing test | `MAX_STEPS=5` on pod B: 10.5 s/step, eval over 4,986 chats 183 s → approved as is (D69) |
| Qwen-E training | 2,088 steps, **6 h 9 min** (22,136 s, 10.6 s/step incl. evals), 17:29–23:39 UTC on pod B; train loss 2.31 (step 1) → 0.0173 (last); **validation 0.0275 / 0.0228 / 0.0223** (Packi-E: 0.0300 / 0.0253 / 0.0243) |
| Adapters | `hf.co/kasramojallal/packi-qwen3-4b-lora-h` and `…-lora-e` (private, with `train_config.json`, `versions.json`, `trainer_state.json`, `train.log`); Qwen-E adapter sha256 `560d057f6eea8f2e…` verified equal on HF and pod; local copies in `checkpoints/lfd-lora-qwen3-4b-{h,e}/` (git-ignored) |
| Note | Qwen's starting loss (2.3) is ~4× Llama's (0.6). One visible cause: unadapted Qwen writes JSON with spaces (`{"rotation_index": 2, "anchor_id": "r2_a7"}`, run file `results/base-qwen3-4b/curriculum25/seed0.json`) while the labels are compact (`{"rotation_index":0,"anchor_id":"r0_a0"}`). By the end of epoch 1 its validation loss is below Llama's |
| Evaluation | pod A, packer branch `claude/e2-qwen3`, harness code `41b43f7`; 160 runs committed in packer `1924236` (base + Qwen-H + Packi re-timing) and `b99cffb` (Qwen-E), 17:35 UTC Oct 1 – 00:35 UTC Oct 2; fp16 inference exactly as for Llama (no harness change) — Qwen produced 0 invalid JSON in all 160 runs |
| Packi re-timing (D69) | the published Packi-H / Packi-E adapters re-run plain on pod A → `llm-robotic-packer/retime/rtxpro4500/` (outside `results/`); 19/20 runs each reproduce the original placements; one near-tie each (see D69) |
| Cost | pod B ≈ 6.7 h × $3.49 ≈ $23.4; pod A ≈ 7.1 h × $0.72 ≈ $5.2; **≈ $28.6** (estimate from pod hours; RunPod balance $51.68 at 18:00 UTC → $18.73 at 00:38 UTC, shared with E3) |

### Results — Qwen3-4B vs Llama 3.2 3B (utilization, mean ± std over seeds 0–4; `python aggregate.py results/` in the packer repo)

| method | teacher | curriculum25 | data1 | data2 | data3 | mean |
|---|---|---|---|---|---|---|
| base-llama (no adapter) | — | 0.360 | 0.256 | 0.243 | 0.290 | 0.287 |
| base-qwen3-4b (no adapter) | — | 0.654 ± 0.068 | 0.614 ± 0.067 | 0.659 ± 0.054 | 0.627 ± 0.070 | 0.638 |
| Packi-H (Llama) | 646 human | 0.731 ± 0.067 | 0.650 ± 0.071 | 0.757 ± 0.042 | 0.686 ± 0.080 | 0.706 |
| **Qwen-H** | 646 human | **0.720 ± 0.060** | **0.698 ± 0.054** | **0.751 ± 0.058** | **0.701 ± 0.083** | **0.717** |
| Packi-E (Llama) | 24,753 expert | 0.789 ± 0.092 | 0.709 ± 0.049 | 0.791 ± 0.021 | 0.750 ± 0.052 | 0.760 |
| **Qwen-E** | 24,753 expert | **0.827 ± 0.037** | **0.757 ± 0.068** | **0.795 ± 0.029** | **0.734 ± 0.059** | **0.778** |
| greedy (no LLM) | — | 0.707 ± 0.037 | 0.727 ± 0.060 | 0.828 ± 0.054 | 0.689 ± 0.017 | 0.738 |
| oracle (sees the future) | — | 0.846 ± 0.074 | 0.848 ± 0.055 | 0.898 ± 0.023 | 0.793 ± 0.058 | 0.846 |

Shuffled anchor ids (mean over datasets): base-qwen3-4b 0.626, Qwen-H 0.681 (Packi-H 0.687), Qwen-E 0.771 (Packi-E 0.764).

Per seed (wins of the first over the second, out of 5; ties excluded): Qwen-H vs Packi-H 2 / 3 / 2 / 2 (Δ −0.011 / +0.048 / −0.006 / +0.015); Qwen-E vs Packi-E 3 / 3 / 3 / 2 (Δ +0.038 / +0.048 / +0.004 / −0.016); Qwen-E vs greedy 5 / 3 / 1 / 3 (Δ +0.120 / +0.030 / −0.033 / +0.045); Qwen-E vs Qwen-H 5 / 4 / 4 / 3.

Reliability: Qwen-H and Qwen-E 100 % first-attempt validity, 0 invalid JSON, 0 retries, 0 path collisions on all 80 runs. Unadapted Qwen: 92–99 % first-attempt validity, 0 invalid JSON (unadapted Llama: 13–19 %, 58–97 invalid JSON per run).

Latency, same card (RTX PRO 4500), median per box end to end (pick + path call): Packi-H 2.01 s, Packi-E 2.05 s, Qwen-H 2.98 s, Qwen-E 2.97 s, unadapted Qwen 2.25 s. Note: `aggregate.py`'s `lat` column is the mean **per-call** latency (≈ half of the per-box figure); Packi-E's original RTX 5090 runs have a per-box median of 2.00–2.34 s.
