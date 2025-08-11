import os
import json
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import LoraConfig, get_peft_model

DATA_DIR = "data/processed"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # pick a small local model you can run
OUTPUT_DIR = "checkpoints/lfd-lora"
MAX_LEN = 1024

def build_text(row):
    # Concatenate pick and path as two supervised turns.
    # Use special tags instead of full chat format to keep it simple.
    return (
        "<|system|>You are a bin-packing assistant. Output STRICT JSON when responding.<|end|>\n"
        f"<|user|>{row['user_pick']}<|end|>\n"
        f"<|assistant|>{row['assistant_pick']}<|end|>\n"
        f"<|user|>{row['user_path']}<|end|>\n"
        f"<|assistant|>{row['assistant_path']}<|end|>\n"
    )

def tokenize_fn(examples, tokenizer):
    texts = [build_text(r) for r in examples]
    return tokenizer(
        texts,
        max_length=MAX_LEN,
        truncation=True
    )

def main():
    ds = load_from_disk(DATA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def map_batch(batch):
        texts = [build_text(row) for row in batch]
        toks = tokenizer(texts, max_length=MAX_LEN, truncation=True, padding="max_length")
        # Labels equal to input_ids for LM SFT
        toks["labels"] = toks["input_ids"].copy()
        return toks

    train = ds["train"].map(map_batch, batched=True, remove_columns=ds["train"].column_names)
    evald = ds["test"].map(map_batch, batched=True, remove_columns=ds["test"].column_names)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(base, lora_cfg)

    args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        fp16=torch.cuda.is_available(),
        learning_rate=1e-4,
        num_train_epochs=2,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        logging_steps=100,
        output_dir=OUTPUT_DIR,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=evald,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
