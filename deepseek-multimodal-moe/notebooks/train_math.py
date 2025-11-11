!pip install -q transformers datasets accelerate bitsandbytes peft scikit-learn

import re
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Base model
BASE_MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"

# Path to your already-trained LOGIC adapter
LOGIC_ADAPTER_PATH = "/content/drive/MyDrive/logic_ai2_ar/logic-adapter"

# Where to save the new Math (GSM8K) adapter
MATH_ADAPTER_PATH = "/content//drive/MyDrive/math_gsm8k_from_logic"

MAX_SEQ_LEN = 512

# Load GSM8K (main split)
gsm = load_dataset("gsm8k", "main")

train_gsm = gsm["train"]
test_gsm = gsm["test"]   # small val set as test from train to save time

print("GSM8K train size:", len(train_gsm))
print("GSM8K test size:", len(test_gsm))

def extract_final_answer(ans_text: str) -> str:
    """
    GSM8K answers are usually like:
    '... explanation ... #### 42'
    We take the part after '####' and strip spaces and commas.
    """
    if "####" in ans_text:
        ans = ans_text.split("####")[-1].strip()
    else:
        ans = ans_text.strip()
    # Normalize: remove trailing period, commas, spaces
    ans = ans.replace(",", "").strip()
    ans = ans.rstrip(".")
    return ans

# 1500 training examples, 200 validation examples
train_val_split = train_gsm.train_test_split(test_size=50, seed=42)
train_dataset = train_val_split["train"].select(range(1500))
val_dataset   = train_val_split["test"]

print(f"Train subset size: {len(train_dataset)}")
print(f"Val subset size: {len(val_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_gsm8k_example(example):
    question = example["question"]
    full_answer = example["answer"]
    clean_answer = extract_final_answer(full_answer)

    prompt = (
        f"Question: {question}\n\n"
        f"Answer with only the final numeric answer.\n\n"
        f"Answer:"
    )
    full_text = f"{prompt} {clean_answer}"

    return prompt, full_text, clean_answer

def preprocess_gsm8k(examples):
    prompts = []
    full_texts = []
    answers = []

    for i in range(len(examples["question"])):
        ex = {k: v[i] for k, v in examples.items()}
        prompt, full_text, clean = format_gsm8k_example(ex)
        prompts.append(prompt)
        full_texts.append(full_text)
        answers.append(clean)

    # Tokenize prompt+answer for model input
    model_inputs = tokenizer(
        full_texts,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
    )

    # Tokenize prompt alone to figure out which tokens to mask in labels
    prompt_inputs = tokenizer(
        prompts,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
    )
    prompt_lengths = [
        int(np.sum(mask)) for mask in prompt_inputs["attention_mask"]
    ]

    labels = np.array(model_inputs["input_ids"])
    # Mask prompt tokens with -100
    for i, plen in enumerate(prompt_lengths):
        labels[i, :plen] = -100
    # Mask padding tokens
    labels[model_inputs["attention_mask"] == 0] = -100

    model_inputs["labels"] = labels.tolist()
    model_inputs["clean_answer"] = answers  # for later evaluation

    return model_inputs

tokenized_train = train_dataset.map(
    preprocess_gsm8k,
    batched=True,
    remove_columns=train_dataset.column_names,
)

tokenized_val = val_dataset.map(
    preprocess_gsm8k,
    batched=True,
    remove_columns=val_dataset.column_names,
)

print("Tokenized train example keys:", tokenized_train.column_names)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False

# Prepare for k-bit training
base_model = prepare_model_for_kbit_training(base_model)

print("Loading LOGIC adapter from:", LOGIC_ADAPTER_PATH)
model = PeftModel.from_pretrained(
    base_model,
    LOGIC_ADAPTER_PATH,
    is_trainable=True,   # continue training this adapter on GSM8K
)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=MATH_ADAPTER_PATH,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    eval_strategy="no",  # we will eval manually
    bf16=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=None,  # no automatic eval
)

print("--- Starting GSM8K fine-tuning from LOGIC adapter ---")
train_result = trainer.train()
print("--- GSM8K fine-tuning complete ---")

# Save the new adapter
print("Saving Math (GSM8K) adapter...")
trainer.save_model(MATH_ADAPTER_PATH)
tokenizer.save_pretrained(MATH_ADAPTER_PATH)

print("Saved Math adapter to:", MATH_ADAPTER_PATH)

    print("\nCould not find expert_capacity.png")
