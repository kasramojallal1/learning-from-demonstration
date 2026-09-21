# training/train_lora_v2.py
"""
LoRA fine-tuning recipe for the Packi revision (T0.10; D35-D37, D42).

    python training/sft_prepare.py                         # -> data/processed_v2/{train,test}.jsonl
    python training/train_lora_v2.py                       # -> checkpoints/lfd-lora-llama32-3b-v2/

One chat per example (pick or path), tokenized with the model's chat template
and a pinned date string; loss on the assistant turn only; no truncation
(fails if an example exceeds MAX_LEN); bf16; no quantization; fixed seed.
Every hyperparameter is written to <output>/train_config.json, the software
and hardware to <output>/versions.json, the per-step loss to
<output>/trainer_state.json, and the data files' hashes to both.

Knobs (env, all with the paper's defaults):
    MODEL_NAME  meta-llama/Llama-3.2-3B-Instruct     OUTPUT_DIR  checkpoints/lfd-lora-llama32-3b-v2
    DATA_DIR    data/processed_v2                     SEED 42
    LORA_R 16   LORA_ALPHA 32   LORA_DROPOUT 0.05     LORA_TARGET q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
    LR 2e-4     EPOCHS 3        BATCH_SIZE 2          GRAD_ACCUM 32   WARMUP_RATIO 0.03   MAX_GRAD_NORM 1.0
    MAX_LEN 2048                DATE_STRING "26 Jul 2024"      MAX_STEPS -1 (set e.g. 2 with OUTPUT_DIR=/tmp/smoke for a smoke test)

Packi-E (T3.7, D52) uses the same script with only the data and output changed:
    DATA_DIR=data/processed_e OUTPUT_DIR=checkpoints/lfd-lora-llama32-3b-e python training/train_lora_v2.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CFG: Dict = {
    "model_name": os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"),
    "data_dir": os.environ.get("DATA_DIR", os.path.join(REPO_ROOT, "data", "processed_v2")),
    "output_dir": os.environ.get("OUTPUT_DIR", os.path.join(REPO_ROOT, "checkpoints", "lfd-lora-llama32-3b-v2")),
    "seed": int(os.environ.get("SEED", "42")),
    "lora_r": int(os.environ.get("LORA_R", "16")),
    "lora_alpha": int(os.environ.get("LORA_ALPHA", "32")),
    "lora_dropout": float(os.environ.get("LORA_DROPOUT", "0.05")),
    "lora_target": os.environ.get("LORA_TARGET", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(","),
    "lr": float(os.environ.get("LR", "2e-4")),
    "epochs": int(os.environ.get("EPOCHS", "3")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "2")),
    "grad_accum": int(os.environ.get("GRAD_ACCUM", "32")),
    "warmup_ratio": float(os.environ.get("WARMUP_RATIO", "0.03")),
    "lr_scheduler": "cosine",
    "max_grad_norm": float(os.environ.get("MAX_GRAD_NORM", "1.0")),
    "optimizer": "adamw_torch",
    "weight_decay": 0.0,
    "max_len": int(os.environ.get("MAX_LEN", "2048")),
    "max_steps": int(os.environ.get("MAX_STEPS", "-1")),   # >0 only for a smoke test; the paper run uses epochs
    "date_string": os.environ.get("DATE_STRING", "26 Jul 2024"),   # D42, same constant in the packer harness
    "precision": "bf16",
    "quantization": "none",
    "loss_on": "assistant tokens only (D35)",
    "gradient_checkpointing": True,
    "attn_implementation": "sdpa",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> Dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    dirty = run("status", "--porcelain", "--", ".", ":(exclude)data/processed_v2", ":(exclude)data/processed_e",
                ":(exclude)checkpoints")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty) if dirty is not None else None}


def versions() -> Dict:
    import transformers, peft, datasets
    v = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": peft.__version__, "datasets": datasets.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None,
        "driver": None,
    }
    try:
        v["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode().strip()
    except Exception:
        pass
    try:
        import accelerate; v["accelerate"] = accelerate.__version__
    except Exception:
        pass
    return v


def encode(tok, messages: List[Dict[str, str]], max_len: int, date_string: str) -> Dict:
    """input_ids for the whole chat; labels = -100 on everything before the assistant turn."""
    def ids(msgs, gen):
        out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen,
                                     date_string=date_string, return_dict=False)
        return list(out["input_ids"]) if hasattr(out, "keys") else list(out)   # transformers 4.x / 5.x
    prompt = ids(messages[:-1], True)
    full = ids(messages, False)
    if full[:len(prompt)] != prompt:
        raise RuntimeError("chat template: prompt is not a prefix of the full chat")
    if len(full) > max_len:
        raise RuntimeError(f"example has {len(full)} tokens > MAX_LEN={max_len}; raise MAX_LEN, do not truncate")
    labels = [-100] * len(prompt) + full[len(prompt):]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    assert torch.cuda.is_bf16_supported(), "bf16 not supported on this GPU"
    set_seed(CFG["seed"])
    out = CFG["output_dir"]
    os.makedirs(out, exist_ok=True)

    train_file = os.path.join(CFG["data_dir"], "train.jsonl")
    test_file = os.path.join(CFG["data_dir"], "test.jsonl")
    manifest_file = os.path.join(CFG["data_dir"], "manifest.json")
    data_info = {"train_file": os.path.relpath(train_file, REPO_ROOT), "train_sha256": sha256(train_file),
                 "test_file": os.path.relpath(test_file, REPO_ROOT), "test_sha256": sha256(test_file),
                 "manifest": json.load(open(manifest_file)) if os.path.exists(manifest_file) else None}

    tok = AutoTokenizer.from_pretrained(CFG["model_name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    ds = load_dataset("json", data_files={"train": train_file, "test": test_file})
    def _map(row):
        return encode(tok, row["messages"], CFG["max_len"], CFG["date_string"])
    cols = ds["train"].column_names
    train_ds = ds["train"].map(_map, remove_columns=cols)
    test_ds = ds["test"].map(_map, remove_columns=cols)
    lens = [len(x) for x in train_ds["input_ids"]]
    n_label = [sum(1 for t in x if t != -100) for x in train_ds["labels"]]
    data_info.update({"train_examples": len(train_ds), "test_examples": len(test_ds),
                      "train_tokens_min_median_max": [min(lens), sorted(lens)[len(lens) // 2], max(lens)],
                      "train_label_tokens_min_median_max": [min(n_label), sorted(n_label)[len(n_label) // 2], max(n_label)]})
    print("data:", json.dumps({k: v for k, v in data_info.items() if k != "manifest"}))

    model = AutoModelForCausalLM.from_pretrained(
        CFG["model_name"], dtype=torch.bfloat16, attn_implementation=CFG["attn_implementation"],
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["lora_target"], bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    import inspect
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    # transformers < 5: warmup_ratio=<ratio>;  transformers >= 5: warmup_steps=<float ratio in [0,1)>
    warmup_kw = ({"warmup_ratio": CFG["warmup_ratio"]} if "warmup_ratio" in ta_params
                 else {"warmup_steps": CFG["warmup_ratio"]})
    args = TrainingArguments(
        output_dir=out, seed=CFG["seed"], data_seed=CFG["seed"],
        per_device_train_batch_size=CFG["batch_size"], per_device_eval_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["lr"], num_train_epochs=CFG["epochs"], max_steps=CFG["max_steps"],
        lr_scheduler_type=CFG["lr_scheduler"], **warmup_kw,
        max_grad_norm=CFG["max_grad_norm"], optim=CFG["optimizer"], weight_decay=CFG["weight_decay"],
        bf16=True, fp16=False,
        logging_strategy="steps", logging_steps=1,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=3,
        report_to="none", remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, padding="longest", label_pad_token_id=-100),
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    final_eval = trainer.evaluate()
    trainer.save_model(out)
    tok.save_pretrained(out)

    record = {
        "config": CFG, "data": data_info, "git": git_info(),
        "trainable_params": trainable, "total_params": total,
        "optimizer_steps": trainer.state.global_step, "train_wall_time_s": round(train_time, 1),
        "final_eval": final_eval,
    }
    with open(os.path.join(out, "train_config.json"), "w") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(out, "versions.json"), "w") as f:
        json.dump(versions(), f, indent=2)
    with open(os.path.join(out, "trainer_state.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print("saved", out, "| steps", trainer.state.global_step, "| eval", final_eval)


if __name__ == "__main__":
    main()
