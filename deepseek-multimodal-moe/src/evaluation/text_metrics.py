# Reload Model and Saved Adapter

from peft import PeftModel

# 1. Define model paths
model_id = "deepseek-ai/deepseek-moe-16b-base"
adapter_path = "./results-boolq/final-adapter" # adjust the path where the trained adapter is saved

# 2. Configure 4-bit quantization (same as before)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 3. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# 4. Load the PEFT adapter onto the base model
eval_model = PeftModel.from_pretrained(base_model, adapter_path)
eval_model.config.use_cache = False # Apply our earlier fix
eval_model.eval() # Put in evaluation mode

print(f"Successfully loaded base model and adapter from {adapter_path}")

# manual evaluation
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

# 1. We re-use the DataLoader with batch_size=1
# (We need the data collator from the previous session)
eval_dataloader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=1,  # The key fix for OOM
    collate_fn=data_collator
)

# 2. Get the metrics functions
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# 3. Run the new manual loop
all_preds = []
all_labels = []

# (We need the eval_model from the previous session)
eval_model.eval()  # Put model in eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("--- Starting Manual Evaluation (Forward Pass, BS=1) ---")

# Use torch.no_grad() to save VRAM (no gradients)
with torch.no_grad():
    for batch in tqdm(eval_dataloader):
        # Move batch to GPU
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Do a standard forward pass.
        # This is NOT generate() and will not use the buggy cache.
        outputs = eval_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # 2. Get the predictions (logits)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # 3. Decode the predictions and labels
        # We only care about the parts where the label is not -100

        # Set ignored tokens to pad_token_id for safe decoding
        labels[labels == -100] = tokenizer.pad_token_id
        predictions[labels == tokenizer.pad_token_id] = tokenizer.pad_token_id

        # Decode the single example
        decoded_pred = tokenizer.decode(predictions[0], skip_special_tokens=True)
        decoded_label = tokenizer.decode(labels[0], skip_special_tokens=True)

        # 4. Clean and store the text
        all_preds.append("Yes" if "yes" in decoded_pred.lower() else "No")
        all_labels.append("Yes" if "yes" in decoded_label.lower() else "No")

# 5. Compute final metrics
print("\n--- Manual Evaluation Complete ---")
try:
    acc_results = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    f1_results = f1_metric.compute(predictions=all_preds, references=all_labels, average="macro", pos_label="Yes")

    print(f"Eval Accuracy: {acc_results['accuracy']:.4f}")
    print(f"Eval F1: {f1_results['f1']:.4f}")

except Exception as e:
    print(f"Error computing metrics: {e}")
    print("Sample Preds:", all_preds[:10])
    print("Sample Labels:", all_labels[:10])
  
# Re-Calculate Metrics with Integers

# 1. The loop is finished and all_preds/all_labels are in memory.
#    convert them from strings ("Yes"/"No") to integers (1/0).

preds_int = [1 if p == "Yes" else 0 for p in all_preds]
labels_int = [1 if l == "Yes" else 0 for l in all_labels]

print("--- Calculating Metrics (Fixed) ---")

# 2. Get the metrics functions (if they are not in memory, this works)
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# 3. Compute metrics using the integer lists
try:
    acc_results = accuracy_metric.compute(predictions=preds_int, references=labels_int)

    # Use pos_label=1 (for "Yes") and the integer lists
    f1_results = f1_metric.compute(
        predictions=preds_int,
        references=labels_int,
        average="macro",
        pos_label=1
    )

    print("\n--- Final Evaluation Results ---")
    print(f"Eval Accuracy: {acc_results['accuracy']:.4f}")
    print(f"Eval F1: {f1_results['f1']:.4f}")

except Exception as e:
    print(f"Error computing metrics: {e}")
    print("Sample Preds (int):", preds_int[:10])
    print("Sample Labels (int):", labels_int[:10])
