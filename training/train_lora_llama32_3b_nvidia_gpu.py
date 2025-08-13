# training/train_lora_llama32_3b_nvidia_gpu.py
# Fine-tune meta-llama/Llama-3.2-3B-Instruct with LoRA on an NVIDIA GPU (e.g., RTX 4070 Ti SUPER).
# Defaults to FP16 on CUDA. If bitsandbytes is available, it will use 4-bit QLoRA automatically.
#
# Env knobs (optional):
#   DATA_DIR=/path/to/data/processed
#   OUTPUT_DIR=/path/to/checkpoints
#   MAX_LEN=512            # try 1024 if VRAM allows
#   BATCH_SIZE=2           # increase if VRAM allows
#   GRAD_ACCUM=32
#   LR=2e-4
#   EPOCHS=3
#   USE_4BIT=true          # force 4-bit if bitsandbytes is installed
#   USE_FLASH=false        # set true if you installed flash-attn and want to try it
#   USE_COMPILE=false      # set true to torch.compile()
#
# Data columns expected (same as your pipeline): user_pick, assistant_pick, user_path, assistant_path

import os
from typing import Dict, List, Optional

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model

# --------------------
# Config
# --------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(REPO_ROOT, "data", "processed"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(REPO_ROOT, "checkpoints", "lfd-lora-llama32-3b-gpu"))

# LoRA (Llama-friendly)
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LORA_TARGET = os.environ.get(
    "LORA_TARGET",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).split(",")

# Training knobs
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "32"))
LR = float(os.environ.get("LR", "2e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "50"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "500"))
SEED = int(os.environ.get("SEED", "42"))

# Optional features
USE_4BIT = os.environ.get("USE_4BIT", "true").lower() in ("1", "true", "yes", "y")
USE_FLASH = os.environ.get("USE_FLASH", "false").lower() in ("1", "true", "yes", "y")
USE_COMPILE = os.environ.get("USE_COMPILE", "false").lower() in ("1", "true", "yes", "y")

# Try to import bitsandbytes/quantization (optional)
bnb_ok = False
BitsAndBytesConfig: Optional[object] = None
prepare_model_for_kbit_training = None
if USE_4BIT:
    try:
        from transformers import BitsAndBytesConfig
        from peft.utils.other import prepare_model_for_kbit_training
        bnb_ok = True
    except Exception:
        bnb_ok = False

def build_messages(row: Dict[str, str]):
    """
    Format one training example as Llama-3.2 chat turns.
    Keeps the same two-pair structure you used for Gemma:
      (user_pick -> assistant_pick) and (user_path -> assistant_path)
    """
    return [
        {"role": "system", "content": "You are a bin-packing assistant. Respond in STRICT JSON only."},
        {"role": "user", "content": row["user_pick"]},
        {"role": "assistant", "content": row["assistant_pick"]},
        {"role": "user", "content": row["user_path"]},
        {"role": "assistant", "content": row["assistant_path"]},
    ]

def main():
    assert torch.cuda.is_available(), "CUDA GPU not detected. Run this on the lab machine with the 4070 Ti SUPER."
    device = torch.device("cuda")
    set_seed(SEED)

    # Performance toggles
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")  # enables TF32 where supported
    except Exception:
        pass

    print("🟢 CUDA device:", torch.cuda.get_device_name(0))

    # Tokenizer (with Llama-3.2 chat template)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Choose attention impl (FlashAttention if installed and requested, else SDPA/eager)
    attn_impl = None
    if USE_FLASH:
        # We'll optimistically try flash-attn v2 via HF flag; if it fails, we fall back silently.
        attn_impl = "flash_attention_2"

    # Load model
    print("🔧 Loading base model...")
    model_kwargs = dict(
        low_cpu_mem_usage=True,
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl  # ignored if not supported by your stack

    quant_config = None
    if bnb_ok:
        # 4-bit QLoRA base
        compute_dtype = torch.float16  # safe default across consumer GPUs
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            **model_kwargs,
        )
        print("✅ Loaded in 4-bit (QLoRA).")
    else:
        # FP16 full-precision weights for forward/backward; only LoRA params train
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            **model_kwargs,
        ).to(device)
        print("✅ Loaded in FP16 (LoRA).")

    # Gradient checkpointing
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    # For k-bit training, prep the model (casts norms, enables input grads)
    if bnb_ok:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA adapters
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Optional: torch.compile for extra speed (PyTorch 2.1+)
    if USE_COMPILE:
        try:
            model = torch.compile(model)
            print("🧪 torch.compile enabled.")
        except Exception as e:
            print(f"⚠️ torch.compile not enabled: {e}")

    # Dataset
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Processed dataset not found at '{DATA_DIR}'. "
            "Run: python training/sft_prepare.py --input data/demos/binpack_lfd.jsonl --out data/processed --train_ratio 0.9"
        )
    ds = load_from_disk(DATA_DIR)
    train_raw = ds["train"]
    test_raw = ds.get("test")

    # Tokenization via Llama-3.2 chat template
    def map_batch(batch: Dict[str, List[str]]):
        n = len(batch["user_pick"])
        rows = [{k: batch[k][i] for k in batch.keys()} for i in range(n)]
        texts = [
            tok.apply_chat_template(
                build_messages(r),
                tokenize=False,
                add_generation_prompt=False,
            )
            for r in rows
        ]
        enc = tok(
            texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    print("📝 Tokenizing train set...")
    train_ds = train_raw.map(map_batch, batched=True, remove_columns=train_raw.column_names)
    test_ds = test_raw.map(map_batch, batched=True, remove_columns=test_raw.column_names) if test_raw else None
    print("🔍 Sample keys:", train_ds[0].keys(), " | seq len:", len(train_ds[0]["input_ids"]))

    # Choose a fast optimizer (fused if available on your PyTorch/CUDA)
    fused_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    optim_name = "adamw_torch_fused" if fused_ok else "adamw_torch"
    if bnb_ok:
        # With 4-bit base, PEFT recommends paged AdamW (bitsandbytes) for stability,
        # but adamw_torch_fused also works well on small adapters. Keep it simple here.
        optim_name = os.environ.get("OPTIM", optim_name)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="no" if test_ds is None else "steps",
        eval_steps=SAVE_STEPS,
        report_to="none",
        fp16=True,                # ✅ Mixed precision on CUDA
        bf16=False,               # keep False for broad compatibility on consumer GPUs
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim=optim_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds if (test_ds and args.eval_strategy != "no") else None,
        processing_class=tok,
        data_collator=default_data_collator,
    )

    print("🚀 Starting training on CUDA...")
    trainer.train()
    print("✅ Training completed.")

    print(f"💾 Saving to {OUTPUT_DIR} ...")
    trainer.save_model()
    tok.save_pretrained(OUTPUT_DIR)
    print("✅ Saved adapter and tokenizer.")
    print("🎉 Done.")


if __name__ == "__main__":
    main()