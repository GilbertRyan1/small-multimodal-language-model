 
# Cell 1: Install Dependencies
 
!pip install -q transformers datasets accelerate
!pip install -q bitsandbytes peft
!pip install -q evaluate scikit-learn

 
# Cell 2: Imports and Environment Info
 
import os
import warnings

import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    pipeline,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training,
)
import evaluate  # HF metrics

warnings.filterwarnings("ignore")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")

 
# Cell 3: Load and Subsample the BoolQ Dataset
 

TRAIN_SAMPLES = 1500
VAL_SAMPLES = 200

# 1. Load the dataset from the hub
full_dataset = load_dataset("boolq")

# 2. Calculate and print the percentage we are using
original_train_size = len(full_dataset["train"])
original_val_size = len(full_dataset["validation"])

train_percentage = (TRAIN_SAMPLES / original_train_size) * 100
val_percentage = (VAL_SAMPLES / original_val_size) * 100

print("--- Dataset Subsampling ---")
print(f"Original train size: {original_train_size}")
print(f"Using {TRAIN_SAMPLES} train samples ({train_percentage:.2f}% of original)")
print(f"Original validation size: {original_val_size}")
print(f"Using {VAL_SAMPLES} validation samples ({val_percentage:.2f}% of original)")
print("---------------------------")

# 3. Create the smaller, balanced dataset
small_train_dataset = full_dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
small_val_dataset = full_dataset["validation"].shuffle(seed=42).select(range(VAL_SAMPLES))

# 4. Combine them into a new DatasetDict
dataset = DatasetDict(
    {
        "train": small_train_dataset,
        "validation": small_val_dataset,
    }
)

# 5. Print the final dataset to verify
print("\nFinal processed dataset:")
print(dataset)

# 6. Print one example to see the structure
print("\nExample data point:")
print(dataset["train"][0])

 
# Cell 4: Load Tokenizer and Quantized Model (4-bit) + Prepare for k-bit Training
 

model_id = "deepseek-ai/deepseek-moe-16b-base"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")

# Model (4-bit)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
print(f"\nModel {model_id} loaded successfully in 4-bit.")

# Prepare model for k-bit (QLoRA) training
model = prepare_model_for_kbit_training(model)
print("Model prepared for k-bit training.")

 
# Cell 5: Preprocess Data and Create Prompt
 

def create_prompt(example):
    prompt_template = (
        f"Passage: {example['passage']}\n"
        f"Question: {example['question']}\n"
        f"Answer: "
    )

    answer = "Yes" if example["answer"] else "No"

    # Tokenize prompt and answer separately
    prompt_tokenized = tokenizer(prompt_template, add_special_tokens=False)
    answer_tokenized = tokenizer(
        answer + tokenizer.eos_token,
        add_special_tokens=False
    )

    input_ids = prompt_tokenized["input_ids"] + answer_tokenized["input_ids"]
    attention_mask = (
        prompt_tokenized["attention_mask"] + answer_tokenized["attention_mask"]
    )

    # Mask out prompt tokens in labels
    labels = [-100] * len(prompt_tokenized["input_ids"]) + answer_tokenized["input_ids"]

    # Truncate to max_length
    max_length = 1024
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_dataset = dataset.map(
    create_prompt,
    remove_columns=["question", "answer", "passage"],
)

print("--- Processed Example ---")
example = tokenized_dataset["train"][0]
print(f"Input IDs (sample): {example['input_ids'][:20]}...")
print(f"Labels (sample): {example['labels'][:20]}...")

print("\n--- Decoded Verification ---")
print("Decoded Input (should be prompt + answer):")
print(tokenizer.decode(example["input_ids"]))

print("\nDecoded Labels (should be only answer):")
decoded_labels = [
    l if l != -100 else tokenizer.pad_token_id for l in example["labels"]
]
print(tokenizer.decode(decoded_labels, skip_special_tokens=True))

 
# Cell 6: Configure PEFT (QLoRA)
 

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

peft_model = get_peft_model(model, lora_config)

print("--- PEFT Model Configured ---")
peft_model.print_trainable_parameters()

# Avoid cache during training
peft_model.config.use_cache = False

 
# Cell 7: Metrics
 

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_preds):
    logits, labels = eval_preds

    # Argmax over vocab dimension
    predictions = np.argmax(logits, axis=-1)

    # Replace -100 in labels for decoding
    labels = labels.copy()
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    cleaned_preds = ["Yes" if "yes" in p.lower() else "No" for p in decoded_preds]
    cleaned_labels = ["Yes" if "yes" in l.lower() else "No" for l in decoded_labels]

    try:
        acc_results = accuracy_metric.compute(
            predictions=cleaned_preds, references=cleaned_labels
        )
        f1_results = f1_metric.compute(
            predictions=cleaned_preds,
            references=cleaned_labels,
            average="macro",
            pos_label="Yes",
        )

        return {
            "accuracy": acc_results["accuracy"],
            "f1": f1_results["f1"],
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        print(f"Preds: {cleaned_preds[:5]}")
        print(f"Labels: {cleaned_labels[:5]}")
        return {"accuracy": 0, "f1": 0}

 
# Cell 8: TrainingArguments, DataCollator, and Trainer
 

training_args = TrainingArguments(
    output_dir="./results-boolq",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,  # works with prepare_model_for_kbit_training
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    do_eval=True,
    eval_steps=50,
    learning_rate=2e-4,
    bf16=True,
    report_to="tensorboard",  # direct TensorBoard logging
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=peft_model,
    padding="longest",
    label_pad_token_id=-100,
)

trainer = Trainer(
    model=peft_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Trainer configured and ready.")


# Cell 9: Start Fine-Tuning and Save Adapter

print("--- Starting 1-Epoch Fine-Tuning for BoolQ ---")
print(f"Logging metrics to TensorBoard in './{training_args.output_dir}'")
trainer.train()
print("\n--- Fine-Tuning Complete ---")

final_adapter_path = f"./{training_args.output_dir}/final-adapter"
trainer.save_model(final_adapter_path)
print(f"Final adapter model saved to {final_adapter_path}")
