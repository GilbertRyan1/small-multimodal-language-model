!pip install -q "transformers" "datasets" "accelerate" "bitsandbytes" "peft" "scikit-learn"

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

BASE_MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"

# IMPORTANT: paths in *your* notebook
BOOLQ_ADAPTER_PATH = "/content/drive/MyDrive/final-adapter"               # previous (text) adapter   adjust the path if required
LOGIC_ADAPTER_PATH = "/content/drive/MyDrive/logic_ai2_ar/final-adapter"  # logic adapter just trained  adjust the path if required

MAX_SEQ_LEN = 512

# Load AI2 ARC dataset (same split as training)
dataset = load_dataset("ai2_arc", "ARC-Challenge")
split = dataset["train"].train_test_split(test_size=50, seed=42)
logic_train = split["train"]
logic_val   = split["test"]

print(f"Logic train size: {len(logic_train)}")
print(f"Logic val size:   {len(logic_val)}")

def format_prompt(example):
    question = example["question"]
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]

    choice_map = {}
    new_labels = [chr(65 + i) for i in range(len(labels))]  # A,B,C,D,...
    formatted_choices = []

    for i, (orig_label, t) in enumerate(zip(labels, texts)):
        new_label = new_labels[i]
        choice_map[orig_label] = new_label
        formatted_choices.append(f"({new_label}) {t}")

    choices_str = "\n".join(formatted_choices)
    correct = choice_map[example["answerKey"]]

    prompt = (
        f"Question: {question}\n\n"
        f"Choices:\n{choices_str}\n\n"
        f"Answer:"
    )

    return prompt, new_labels, correct

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization
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

# STEP 1: load the previous (text) adapter
print("Loading previous (text) adapter from:", BOOLQ_ADAPTER_PATH)
model = PeftModel.from_pretrained(
    base_model,
    BOOLQ_ADAPTER_PATH,
    is_trainable=False,
)

# STEP 2: load the Logic adapter on top of that model
print("Loading Logic adapter from:", LOGIC_ADAPTER_PATH)
# This call loads an additional adapter into the same PEFT model
model = PeftModel.from_pretrained(
    model,
    LOGIC_ADAPTER_PATH,
    is_trainable=False,
)

# use The logic adapter was created with name used in your training code
model.load_adapter(LOGIC_ADAPTER_PATH, adapter_name="logic_ai2_ar")
model.set_adapter("logic_ai2_ar")


model.to(DEVICE)
model.eval()

print("Model for Logic evaluation")

print("\n=== Evaluating Logic modality ===")

y_true, y_pred = [], []

for i, ex in enumerate(logic_val):
    prompt, label_list, correct = format_prompt(ex)
    y_true.append(correct)

    scores = []
    for letter in label_list:
        candidate = f"{prompt} {letter}"
        enc = tokenizer(
            candidate,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            out = model(**enc)
            logits = out.logits[0, -1, :]  # final position

        letter_id = tokenizer(letter, add_special_tokens=False)["input_ids"][0]
        score = logits[letter_id].item()
        scores.append(score)

    best_idx = int(np.argmax(scores))
    pred = label_list[best_idx]
    y_pred.append(pred)

    if i < 3:
        print(f"\nExample {i}")
        print("Prompt:\n", prompt)
        print("Scores:", {l: s for l, s in zip(label_list, scores)})
        print("True:", correct, "Pred:", pred)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro", labels=["A","B","C","D"])

print("\n--- Logic Results ---")
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1:.4f}")
