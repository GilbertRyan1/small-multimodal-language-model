import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Paths
BASE_MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"
MATH_ADAPTER_PATH = "/content/drive/MyDrive/math_gsm8k_from_logic"
CODE_ADAPTER_PATH = "/content/drive/MyDrive/code_codeparrot"

CODE_DATASET_ID = "codeparrot/codeparrot-clean-valid"
MAX_SEQ_LEN = 512
EVAL_SAMPLES = 50

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load CodeParrot and take eval subset (same way as during training)
raw_code = load_dataset(CODE_DATASET_ID, split="train")
print("Total CodeParrot rows:", len(raw_code))

# train1500+test 50
eval_code = raw_code.select(range(1500, 1500 + EVAL_SAMPLES))
print("Eval subset size:", len(eval_code))

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model (DeepSeek-MoE) in 4-bit
base_model_eval = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base_model_eval.config.use_cache = False

# Prepare for k-bit (even though we won't train, it's safe)
base_model_eval = prepare_model_for_kbit_training(base_model_eval)

# Load math adapter
print("Loading math adapter from:", MATH_ADAPTER_PATH)
eval_model = PeftModel.from_pretrained(
    base_model_eval,
    MATH_ADAPTER_PATH,
    adapter_name="math_adapter", # Explicitly name the math adapter
    is_trainable=False,
)

# Load CODE adapter on top of math
print("Loading CODE adapter from:", CODE_ADAPTER_PATH)
eval_model.load_adapter(CODE_ADAPTER_PATH, adapter_name="code_codeparrot") # Use load_adapter to add it to the existing PeftModel

# Make sure the CODE adapter is active (name must match what you used in training)
eval_model.set_adapter("code_codeparrot")

eval_model.to(DEVICE)
eval_model.eval()

print("Eval model with math + code adapters is ready on", DEVICE)

from sklearn.metrics import accuracy_score

def compute_pass_at_1_last_token(model, tokenizer, dataset, max_seq_len=MAX_SEQ_LEN):
    model.eval()
    pass_count = 0
    total = 0

    for ex in dataset:
        code = ex["content"]
        enc = tokenizer(
            code,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(DEVICE)      # [1, L]
        attention_mask = enc["attention_mask"].to(DEVICE)

        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue  # need at least one prompt token and one to predict

        # Prompt is all but the last token; target is last token
        prompt_ids = input_ids[:, :-1]
        target_id = input_ids[0, -1]
        prompt_mask = attention_mask[:, :-1]

        with torch.no_grad():
            outputs = model(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
            )
            # logits: [1, L-1, vocab]; prediction for final position
            logits = outputs.logits[0, -1, :]
            pred_id = torch.argmax(logits).item()

        total += 1
        if pred_id == target_id.item():
            pass_count += 1

    if total == 0:
        return 0.0
    return pass_count / total

print("=== Evaluating approximate Pass@1 (last-token completion) on eval set ===")
approx_pass_at_1 = compute_pass_at_1_last_token(
    eval_model,
    tokenizer,
    eval_code,
    max_seq_len=MAX_SEQ_LEN,
)

print(f"Approximate Pass@1 (last-token completion): {approx_pass_at_1:.4f}")
