import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

DATA_DIR = "data/processed"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # example
OUTPUT_DIR = "checkpoints/lfd-qlora"
MAX_LEN = 1024

def build_text(row):
    return (
        "<|system|>You are a bin-packing assistant. Output STRICT JSON when responding.<|end|>\n"
        f"<|user|>{row['user_pick']}<|end|>\n"
        f"<|assistant|>{row['assistant_pick']}<|end|>\n"
        f"<|user|>{row['user_path']}<|end|>\n"
        f"<|assistant|>{row['assistant_path']}<|end|>\n"
    )

def main():
    ds = load_from_disk(DATA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def map_batch(batch):
        texts = [build_text(row) for row in batch]
        toks = tokenizer(texts, max_length=MAX_LEN, truncation=True, padding="max_length")
        toks["labels"] = toks["input_ids"].copy()
        return toks

    train = ds["train"].map(map_batch, batched=True, remove_columns=ds["train"].column_names)
    evald = ds["test"].map(map_batch, batched=True, remove_columns=ds["test"].column_names)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto"
    )

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(base, lora_cfg)

    args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_train_epochs=2,
        fp16=torch.cuda.is_available(),
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
    print(f"Saved QLoRA adapter to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
