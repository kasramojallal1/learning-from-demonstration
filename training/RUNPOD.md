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
