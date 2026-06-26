# scripts/finetune/train_unsloth.py
"""
QLoRA fine-tune Qwen2.5-3B-Instruct on the router dataset using Unsloth.
Run on Kaggle (free T4 x2) or Google Colab (free T4).

Kaggle setup:
1. New notebook, enable GPU accelerator (T4 x2).
2. First cell: !pip install unsloth
3. Add Kaggle Secret named HF_TOKEN with your HuggingFace token.
4. Set HF_DATASET below to your dataset repo id.
5. Run all cells.
"""

# ── Config ────────────────────────────────────────────────────────────────────
HF_DATASET = "YOUR_HF_USERNAME/cortex-router-dataset"  # <- replace before running
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
OUTPUT_DIR = "./router-lora"
GGUF_DIR = "./router-gguf"
MAX_SEQ_LEN = 512
LORA_RANK = 16
LORA_ALPHA = 32
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 4
LR = 2e-4
# ─────────────────────────────────────────────────────────────────────────────

import os
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    login(token=hf_token)

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Load base model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# Attach LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Load + format dataset
dataset = load_dataset(HF_DATASET)
train_dataset = dataset["train"]


def apply_chat_template(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


train_dataset = train_dataset.map(apply_chat_template, remove_columns=["messages"])

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=10,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=True,
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        report_to="none",
    ),
)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Export to GGUF Q4_K_M for Ollama
model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
print(f"GGUF saved to {GGUF_DIR}")
print("Next: download the .gguf file, place it next to scripts/finetune/Modelfile,")
print("then run: ollama create cortex-router -f scripts/finetune/Modelfile")
