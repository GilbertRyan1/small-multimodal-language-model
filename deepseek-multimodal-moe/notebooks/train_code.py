!pip install -q transformers datasets accelerate bitsandbytes peft scikit-learn

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
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from sklearn.metrics import accuracy_score

BASE_MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"

# math-finetuned model (adapter) you already have
MATH_ADAPTER_PATH = "/content/drive/MyDrive/math_gsm8k_from_logic" #change the path to your path

# where to save the new CODE adapter
CODE_ADAPTER_PATH = "/content/drive/MyDrive/code_codeparrot"      #change the path to your path

# CodeParrot dataset (clean valid split has 'content' field with code)
CODE_DATASET_ID = "codeparrot/codeparrot-clean-valid"

TRAIN_SAMPLES = 1500
EVAL_SAMPLES = 50
MAX_SEQ_LEN = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Load CodeParrot clean valid
raw_code = load_dataset(CODE_DATASET_ID, split="train")

print("Total rows in CodeParrot-clean-valid:", len(raw_code))
print("Columns:", raw_code.column_names)

# Subsample to fit your resource constraints
train_code = raw_code.select(range(TRAIN_SAMPLES))
eval_code  = raw_code.select(range(TRAIN_SAMPLES, TRAIN_SAMPLES + EVAL_SAMPLES))

print("Train subset size:", len(train_code))
print("Eval subset size:", len(eval_code))

# Load tokenizer from base DeepSeek model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_code_examples(examples):
    # Code is in the 'content' field
    texts = examples["content"]
    tokenized = tokenizer(
        texts,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
    )
    # For plain LM, labels = input_ids (no masking)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train_code = train_code.map(
    preprocess_code_examples,
    batched=True,
    remove_columns=train_code.column_names,
)

tokenized_eval_code = eval_code.map(
    preprocess_code_examples,
    batched=True,
    remove_columns=eval_code.column_names,
)

print(tokenized_train_code[0].keys())

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base DeepSeek model
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

# Load math adapter (frozen or lightly trainable base)
print(f"Loading math adapter from: {MATH_ADAPTER_PATH}")
model = PeftModel.from_pretrained(
    base_model,
    MATH_ADAPTER_PATH,
    is_trainable=False,  # we freeze math adapter weights
)

# Define LoRA config for CODE
code_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Add a new adapter for CODE on top of math
model.add_adapter("code_codeparrot", code_lora_config)
model.set_adapter("code_codeparrot")

print("Adapters:")
model.print_trainable_parameters()

# Simple Trainer (we don't include custom router loss here to keep it simple;
# you can reuse your MoETrainer if you want, it will also work)

training_args = TrainingArguments(
    output_dir=CODE_ADAPTER_PATH,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="no",   # no eval during training
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
    report_to="tensorboard",           
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_code,
    eval_dataset=tokenized_eval_code,  # just for convenience; we won't call trainer.evaluate()
    tokenizer=tokenizer,
)

print("Trainer for CODE modality is ready.")

print("=== Starting CODE fine-tuning on CodeParrot subset ===")
train_result = trainer.train()
print("=== Training complete ===")

# Save only the CODE adapter
print("Saving CODE adapter to:", CODE_ADAPTER_PATH)
trainer.save_model(CODE_ADAPTER_PATH)          # saves PEFT adapter weights
tokenizer.save_pretrained(CODE_ADAPTER_PATH)   # save tokenizer alongside

print("CODE adapter saved.")
