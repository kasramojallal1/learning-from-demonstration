# training/train_lora.py
import os
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import LoraConfig, get_peft_model

# --- Resolve project root (this file is training/train_lora.py) ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = ROOT / "checkpoints" / "lfd-lora-1b-mps"
MAX_LEN = 512

def build_text(row):
    return (
        "<|system|>You are a bin-packing assistant. Respond in STRICT JSON only.<|end|>\n"
        f"<|user|>{row['user_pick']}<|end|>\n"
        f"<|assistant|>{row['assistant_pick']}<|end|>\n"
        f"<|user|>{row['user_path']}<|end|>\n"
        f"<|assistant|>{row['assistant_path']}<|end|>\n"
    )

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {DATA_DIR}\n(cwd={Path.cwd()})")

    ds = load_from_disk(str(DATA_DIR))
    train_raw = ds["train"]
    eval_raw  = ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def map_batch(batch):
        texts = [build_text(row) for row in batch]
        toks = tokenizer(texts, max_length=MAX_LEN, truncation=True, padding="max_length")
        toks["labels"] = toks["input_ids"].copy()
        return toks

    train = train_raw.map(map_batch, batched=True, remove_columns=train_raw.column_names)
    evald = eval_raw.map(map_batch, batched=True, remove_columns=eval_raw.column_names)

    device_map = {"": "mps"} if torch.backends.mps.is_available() else "auto"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=device_map,
    )

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

    args = TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        num_train_epochs=3,
        max_grad_norm=1.0,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="no",
        report_to="none",
        fp16=False, bf16=False,
        output_dir=str(OUTPUT_DIR),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=evald, data_collator=data_collator)
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"✅ Saved LoRA adapter to {OUTPUT_DIR}")

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    main()
