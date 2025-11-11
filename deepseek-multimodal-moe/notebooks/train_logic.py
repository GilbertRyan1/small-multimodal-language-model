 
# Install dependencies
 
!pip install -q transformers datasets accelerate bitsandbytes peft matplotlib seaborn tensorboard scikit-learn

 
# Imports
 
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from peft import (
    get_peft_model,
    PeftModel,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)

 
# Config
 
BASE_MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"

# Make sure these paths are correct in your Colab / Drive
BOOLQ_ADAPTER_PATH = "/content/drive/MyDrive/Text-adapter"   # path to BoolQ adapter
NEW_ADAPTER_PATH = "/content/results/logic_ai2_ar"           # path to save new adapter

DATASET_ID = "ai2_arc"
DATASET_CONFIG = "ARC-Challenge"

 
# Dataset
 
dataset = load_dataset(DATASET_ID, DATASET_CONFIG)

# Use 200 examples as validation, rest as train
train_val_split = dataset["train"].train_test_split(test_size=200, seed=42)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]

print(f"Training data: {len(train_dataset)} examples")
print(f"Validation data: {len(val_dataset)} examples")

 
# Tokenizer
 
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

 
# Prompt formatting and preprocessing
 
def format_prompt(example):
    """Format one AI2 ARC example as a multiple-choice prompt."""
    question = example["question"]

    labels = example["choices"]["label"]
    texts = example["choices"]["text"]

    choice_map = {}
    new_labels = [chr(65 + i) for i in range(len(labels))]  # A, B, C, ...

    formatted_choices = []
    for i, (original_label, text) in enumerate(zip(labels, texts)):
        new_label = new_labels[i]
        choice_map[original_label] = new_label
        formatted_choices.append(f"({new_label}) {text}")

    choices_str = "\n".join(formatted_choices)
    correct_answer_label = choice_map[example["answerKey"]]

    prompt = f"Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:"
    full_text = f"{prompt} {correct_answer_label}"

    return prompt, full_text, correct_answer_label


def preprocess_function(examples):
    prompts = []
    full_texts = []
    labels_list = []

    for i in range(len(examples["question"])):
        ex = {k: v[i] for k, v in examples.items()}
        prompt, full_text, label = format_prompt(ex)
        prompts.append(prompt)
        full_texts.append(full_text)
        labels_list.append(label)

    model_inputs = tokenizer(
        full_texts,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    prompt_inputs = tokenizer(
        prompts,
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    prompt_lengths = [np.sum(mask) for mask in prompt_inputs["attention_mask"]]

    labels = np.array(model_inputs["input_ids"])
    labels_mask = np.arange(labels.shape[1]) < np.array(prompt_lengths)[:, None]
    labels[labels_mask] = -100
    labels[model_inputs["attention_mask"] == 0] = -100

    model_inputs["labels"] = labels.tolist()
    model_inputs["clean_labels"] = labels_list

    return model_inputs


tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

tokenized_val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
)

 
# Load base model with 4-bit quantization and prepare for k-bit training
 
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

# Needed for QLoRA with gradient checkpointing
model = prepare_model_for_kbit_training(model)

 
# Load BoolQ adapter and add new adapter for logic task
 
print(f"Loading base model complete. Now loading adapter from: {BOOLQ_ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, BOOLQ_ADAPTER_PATH, is_trainable=True)
print("BoolQ adapter loaded and set to trainable.")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
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

model.add_adapter("logic_ai2_ar", lora_config)
model.set_adapter("logic_ai2_ar")
print("New adapter 'logic_ai2_ar' added and set as active.")

model.print_trainable_parameters()

 
# MoE router monitor callback
 
class MoERouterMonitor(TrainerCallback):
    """Capture router statistics (if exposed by the model)."""

    def __init__(self):
        self.stats = []
        self.router_logits_handle = None

    def _capture_router_logits(self, model, args, output):
        if hasattr(output, "router_logits") and output.router_logits:
            self.current_router_logits.append(
                output.router_logits[-1].clone().detach().cpu()
            )

    def on_evaluate_begin(self, args, state, control, model, **kwargs):
        self.current_router_logits = []
        self.router_logits_handle = model.register_forward_hook(
            self._capture_router_logits
        )

    def on_evaluate_end(self, args, state, control, **kwargs):
        if self.router_logits_handle:
            self.router_logits_handle.remove()
            self.router_logits_handle = None

        if not self.current_router_logits:
            print("MoERouterMonitor: No router_logits captured.")
            return

        all_logits = torch.cat(self.current_router_logits, dim=0)
        num_tokens, num_experts = all_logits.shape

        gating_weights, selected_experts = torch.topk(all_logits, 2, dim=-1)
        gating_weights = torch.softmax(gating_weights, dim=-1, dtype=torch.float32)

        expert_counts = torch.zeros(num_experts, dtype=torch.long)
        for i in range(num_experts):
            expert_counts[i] = (selected_experts == i).sum()

        expert_utilization = expert_counts.float() / num_tokens

        p = expert_utilization
        p = p[p > 0]
        routing_entropy = (-torch.sum(p * torch.log(p))).item()
        cv_load = (expert_counts.float().std() / expert_counts.float().mean()).item()

        step_stats = {
            "step": state.global_step,
            "expert_utilization": expert_utilization.numpy(),
            "routing_entropy": routing_entropy,
            "load_balancing_cv": cv_load,
            "avg_gating_weight_top1": gating_weights[:, 0].mean().item(),
            "avg_gating_weight_top2": gating_weights[:, 1].mean().item(),
        }
        self.stats.append(step_stats)

        if not hasattr(state, "moe_monitor_stats"):
            state.moe_monitor_stats = []
        state.moe_monitor_stats.append(step_stats)

        print(
            f"MoERouterMonitor: Step {state.global_step}, "
            f"Entropy: {routing_entropy:.4f}, CV Load: {cv_load:.4f}"
        )

    def get_stats(self):
        return self.stats


def save_router_plots_and_stats(stats_list, output_dir):
    """Save MoE router stats and plots."""
    if not stats_list:
        print("No MoE stats to save.")
        return

    import os
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(stats_list)

    # Scalar metrics over steps
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x="step", y="routing_entropy", marker="o")
    plt.title("Routing Entropy over Training")
    plt.ylabel("Entropy")

    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x="step", y="load_balancing_cv", marker="o")
    plt.title("Load Balancing (CV) over Training")
    plt.ylabel("Coefficient of Variation (Load)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/moe_scalar_metrics.png")
    plt.close()

    # Expert utilization heatmap
    utilization_data = np.array([s["expert_utilization"] for s in stats_list])
    steps = [s["step"] for s in stats_list]
    num_experts = utilization_data.shape[1]

    plt.figure(figsize=(15, 8))
    sns.heatmap(
        utilization_data.T,
        xticklabels=steps,
        yticklabels=[f"E{i}" for i in range(num_experts)],
    )
    plt.title("Expert Utilization (Token Throughput %)")
    plt.xlabel("Training Step")
    plt.ylabel("Expert ID")
    plt.savefig(f"{output_dir}/moe_expert_utilization_heatmap.png")
    plt.close()

    # Flattened CSV
    df_exploded = df.explode("expert_utilization")
    df_exploded["expert_id"] = df_exploded.groupby(level=0).cumcount()
    df_exploded.to_csv(f"{output_dir}/moe_router_stats_full.csv", index=False)

    print(f"MoE router plots and stats saved to {output_dir}")


 
# Custom Trainer including router auxiliary loss
 
class MoETrainer(Trainer):
    """Extend loss with router auxiliary loss if provided by the model."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if hasattr(outputs, "router_losses") and outputs.router_losses:
            router_loss = 0
            for r_loss in outputs.router_losses:
                if r_loss is not None:
                    router_loss += r_loss
            loss += router_loss

        return (loss, outputs) if return_outputs else loss


 
# Metrics
 
def compute_metrics(eval_preds):
    """Compute F1 and accuracy on token-level prediction of the correct label."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    pred_labels = []

    for i in range(len(labels)):
        label_indices = np.where(labels[i] != -100)[0]
        if len(label_indices) > 0:
            idx = label_indices[0]
            true_labels.append(labels[i, idx])
            pred_labels.append(predictions[i, idx])

    if not true_labels:
        return {"accuracy": 0.0, "f1": 0.0}

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro")

    return {"accuracy": accuracy, "f1": f1}


 
# Training arguments
 
training_args = TrainingArguments(
    output_dir=NEW_ADAPTER_PATH,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="no",      # no periodic eval during training
    per_device_eval_batch_size=1,
    eval_accumulation_steps=10,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    bf16=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
    save_total_limit=2,
)

 
# Trainer + callback
 
moe_monitor = MoERouterMonitor()

trainer = MoETrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[moe_monitor],
)

 
# Training
 
trainer.train()
print("--- Full training complete ---")

 
# Save final adapter
 
print("Saving final adapter.")
trainer.save_model(f"{NEW_ADAPTER_PATH}/final-adapter")
tokenizer.save_pretrained(f"{NEW_ADAPTER_PATH}/final-adapter")

print("Model training and saving complete. Ready for evaluation.")
