!pip install -q transformers datasets accelerate bitsandbytes peft scikit-learn

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

BASE_MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"

LOGIC_ADAPTER_PATH = "/content/drive/MyDrive/logic_ai2_ar/logic-adapter" #adjust the path to your previous adpater
MATH_ADAPTER_PATH  = "/content/drive/MyDrive/math_gsm8k_from_logic" # adjust the path of the just trained adapter

MAX_SEQ_LEN = 512

# 1) Load GSM8K (math modality)
math_dataset = load_dataset("gsm8k", "main")
math_train = math_dataset["train"]
math_test  = math_dataset["test"]

print(f"Math train size (GSM8K): {len(math_train)}")
print(f"Math test size:          {len(math_test)}")

# For evaluation, we can optionally use a subset, e.g. 50 problems
math_eval = math_test.select(range(50))
print(f"Math eval subset size:   {len(math_eval)}")

def format_math_prompt(example):
    """
    Build a simple math QA prompt.
    """
    question = example["question"]
    prompt = f"Question: {question}\nAnswer:"
    answer = example["answer"]  # full reasoning + final answer line in GSM8K
    return prompt, answer

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization for memory
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Base DeepSeek model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

# Step 1: load LOGIC adapter (previous modality)
print("Loading Logic adapter from:", LOGIC_ADAPTER_PATH)
model = PeftModel.from_pretrained(
    base_model,
    LOGIC_ADAPTER_PATH,
    is_trainable=False,
)

# Step 2: load MATH adapter (trained on top of Logic)
print("Loading Math adapter from:", MATH_ADAPTER_PATH)
model = PeftModel.from_pretrained(
    model,
    MATH_ADAPTER_PATH,
    is_trainable=False,
)

# Activate the math adapter by name (adjust if you used a different name)
try:
    model.set_adapter("math_gsm8k_from_logic")
except Exception:
    # If you aren't sure of the adapter name, you can inspect:
    print("Available adapters:", model.peft_config.keys())
    # Then manually set the correct one.

model.to(DEVICE)
model.eval()

print("Math evaluation model ready")

print("\n=== Evaluating MATH modality ===")

exact_match_count = 0
total_examples = 0
skipped = 0

for i, ex in enumerate(math_eval):
    prompt, answer = format_math_prompt(ex)
    full_text = prompt + " " + answer

    # Tokenize full text and prompt to find answer span
    enc_full = tokenizer(
        full_text,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    enc_prompt = tokenizer(
        prompt,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )

    input_ids = enc_full["input_ids"].to(DEVICE)         # [1, L]
    attention_mask = enc_full["attention_mask"].to(DEVICE)
    L = input_ids.shape[1]
    prompt_len = enc_prompt["input_ids"].shape[1]

    # Need at least one answer token and space for LM shift
    if L <= prompt_len + 1:
        skipped += 1
        continue

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [1, L, vocab]

    # Check whether EVERY answer token is predicted correctly
    all_correct = True
    for t in range(prompt_len, L):
        true_id = input_ids[0, t].item()
        pred_logits = logits[0, t - 1, :]
        pred_id = int(torch.argmax(pred_logits))

        if pred_id != true_id:
            all_correct = False
            break

    total_examples += 1
    if all_correct:
        exact_match_count += 1

    if i < 3:
        print(f"\nExample {i}")
        print("Prompt:\n", prompt)
        print("Answer (truncated):", answer[:120].replace("\n", " "))
        print(f"Answer tokens in eval: {L - prompt_len}")
        print("Exact match for this example:", all_correct)

if total_examples == 0:
    math_exact_match = 0.0
else:
    math_exact_match = exact_match_count / total_examples

print("\n--- Math Results (Base + Logic + Math) ---")
print(f"Total evaluated examples: {total_examples}")
print(f"Exact matches:            {exact_match_count}")
print(f"Exact match accuracy:     {math_exact_match:.4f}")
print(f"Skipped examples (too short/truncated): {skipped}")

