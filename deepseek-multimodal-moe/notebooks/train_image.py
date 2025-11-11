!pip install -q transformers datasets accelerate bitsandbytes peft pillow torchvision pycocoevalcap

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    CLIPVisionModel,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training
from pycocoevalcap.cider.cider import Cider

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ---- Core model config ----
BASE_MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"

#  path to the adapter you want to stack IMAGE on (e.g. code adapter)
PREV_ADAPTER_PATH = "/content/drive/MyDrive/code_codeparrot"

# New adapter for IMAGE modality
IMAGE_ADAPTER_NAME = "image_coco"
IMAGE_ADAPTER_SAVE_DIR = "/content/drive/MyDrive/image_coco_from_code"

# Vision encoder (frozen CLIP)
VISION_MODEL_ID = "openai/clip-vit-base-patch32"
PREFIX_LENGTH = 10
MAX_CAPTION_LEN = 64

# Dataset config
COCO_DATASET_ID = "lmms-lab/COCO-Caption"
TRAIN_SAMPLES = 1000      
EVAL_SAMPLES  = 50

print("Loading COCO dataset (val split)...")
coco_full = load_dataset(COCO_DATASET_ID, split="val")
print("Total dataset size:", len(coco_full))

# Make small train/eval splits
coco_train = coco_full.select(range(TRAIN_SAMPLES))
coco_eval  = coco_full.select(range(TRAIN_SAMPLES, TRAIN_SAMPLES + EVAL_SAMPLES))

def build_caption_text(example):
    """
    lmms-lab/COCO-Caption:
      - 'image': PIL image
      - 'answer': list of 5 captions
    Use answer[0] as training caption.
    """
    if "answer" in example and example["answer"]:
        cap = example["answer"][0]
    else:
        cap = "[no caption]"
    example["caption_text"] = cap
    return example

coco_train = coco_train.map(build_caption_text)
coco_eval  = coco_eval.map(build_caption_text)

print("Example eval keys:", coco_eval[0].keys())
print("Example eval caption_text:", repr(coco_eval[0]["caption_text"]))

"""inference"""

# Tokenizer for DeepSeek
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Vision processor for CLIP
image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_ID)

def preprocess_coco_batch(batch):
    """Turn (image + caption_text) into tensors for training/eval."""

    # 1) Images -> pixel_values
    images = [img.convert("RGB") for img in batch["image"]]
    vision_out = image_processor(images=images, return_tensors="pt")
    pixel_values = vision_out["pixel_values"]   # [B, C, H, W]

    # 2) Captions -> tokens
    text_out = tokenizer(
        batch["caption_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_CAPTION_LEN,
        return_tensors="pt",
    )
    input_ids = text_out["input_ids"]
    attention_mask = text_out["attention_mask"]

    # 3) Labels = input_ids, but pad_token_id -> -100 for loss masking
    labels = []
    for seq in input_ids:
        labels.append([
            tok if tok != tokenizer.pad_token_id else -100
            for tok in seq
        ])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

print("Preprocessing COCO subsets...")
coco_train_proc = coco_train.map(
    preprocess_coco_batch,
    batched=True,
    remove_columns=coco_train.column_names,
)
coco_eval_proc = coco_eval.map(
    preprocess_coco_batch,
    batched=True,
    remove_columns=coco_eval.column_names,
)

coco_train_proc.set_format(type="torch")
coco_eval_proc.set_format(type="torch")

print("Processed train:", len(coco_train_proc), "eval:", len(coco_eval_proc))
print("Processed keys:", coco_eval_proc[0].keys())

coco_train_proc[0].keys()
# → dict_keys(['pixel_values', 'input_ids', 'attention_mask'])

"""Vision encoder"""

# ---- Load base DeepSeek-MoE in 4-bit ----
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
base_model = prepare_model_for_kbit_training(base_model)

# ---- Load previous adapter (e.g. code) ----
print("Loading previous adapter from:", PREV_ADAPTER_PATH)
lm_with_prev = PeftModel.from_pretrained(
    base_model,
    PREV_ADAPTER_PATH,
    is_trainable=False,
)

# ---- Add new IMAGE adapter on top ----
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

lm_with_prev.add_adapter(IMAGE_ADAPTER_NAME, lora_config)
lm_with_prev.set_adapter(IMAGE_ADAPTER_NAME)
lm_with_prev.to(DEVICE)

print("Trainable parameters in LM (with image adapter):")
lm_with_prev.print_trainable_parameters()

# ---- Frozen CLIP vision encoder ----
vision_model = CLIPVisionModel.from_pretrained(VISION_MODEL_ID).to(DEVICE)
for p in vision_model.parameters():
    p.requires_grad = False

vision_hidden_size = vision_model.config.hidden_size
lm_hidden_size = lm_with_prev.base_model.model.model.embed_tokens.embedding_dim

print("Vision hidden size:", vision_hidden_size)
print("LM hidden size:", lm_hidden_size)

# ---- Prefix projector ----
class ImagePrefixProjector(nn.Module):
    """
    Map CLIP pooled embedding -> sequence of LM prefix vectors.
    """
    def __init__(self, vision_dim, lm_dim, prefix_len):
        super().__init__()
        self.prefix_len = prefix_len
        self.lm_dim = lm_dim
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, lm_dim * prefix_len),
            nn.Tanh(),
        )

    def forward(self, vision_emb):
        x = self.proj(vision_emb)                      # [B, prefix_len * lm_dim]
        x = x.view(-1, self.prefix_len, self.lm_dim)   # [B, prefix_len, lm_dim]
        return x

prefix_projector = ImagePrefixProjector(
    vision_dim=vision_hidden_size,
    lm_dim=lm_hidden_size,
    prefix_len=PREFIX_LENGTH,
).to(DEVICE)

print("Prefix projector trainable params:",
      sum(p.numel() for p in prefix_projector.parameters()))

# ---- VisionLanguageModel wrapper ----
class VisionLanguageModel(nn.Module):
    def __init__(self, vision_model, lm_model, projector, tokenizer, prefix_len):
        super().__init__()
        self.vision_model = vision_model
        self.lm_model = lm_model
        self.projector = projector
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        # 1) Vision encoder
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            vision_emb = vision_outputs.pooler_output
        else:
            vision_emb = vision_outputs.last_hidden_state[:, 0, :]

        visual_prefix = self.projector(vision_emb)   # [B, P, H]

        # 2) Text embeddings from LM
        emb_layer = self.lm_model.base_model.model.model.embed_tokens
        text_embeds = emb_layer(input_ids)           # [B, T, H]

        # 3) Concatenate prefix + text
        inputs_embeds = torch.cat([visual_prefix, text_embeds], dim=1)  # [B, P+T, H]

        # 4) Attention mask
        if attention_mask is not None:
            bsz = attention_mask.shape[0]
            prefix_mask = torch.ones(
                bsz, self.prefix_len,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attn_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            attn_mask = None

        #  Labels: we already have labels for text tokens (padding -> -100)
        #   prepend -100 for prefix positions
        if labels is not None:
            prefix_labels = torch.full(
                (labels.shape[0], self.prefix_len),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            new_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            new_labels = None

        outputs = self.lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=new_labels,
        )
        return outputs

vl_model = VisionLanguageModel(
    vision_model=vision_model,
    lm_model=lm_with_prev,
    projector=prefix_projector,
    tokenizer=tokenizer,
    prefix_len=PREFIX_LENGTH,
).to(DEVICE)

print("Vision-language model ready.")

from torch.utils.data import DataLoader

def image_caption_data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    labels = torch.stack([f["labels"] for f in features])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

training_args = TrainingArguments(
    output_dir=IMAGE_ADAPTER_SAVE_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="no",
    num_train_epochs=1,
    max_steps=300,          
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="steps",
    save_steps=150,
    save_total_limit=2,
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=vl_model,
    args=training_args,
    train_dataset=coco_train_proc,
    eval_dataset=coco_eval_proc,
    data_collator=image_caption_data_collator,
    tokenizer=tokenizer,  # FutureWarning is OK
)

print("Starting IMAGE (COCO) training...")
train_result = trainer.train()
print("IMAGE training complete.")

print("Saving IMAGE adapter and prefix projector...")
vl_model.lm_model.save_pretrained(IMAGE_ADAPTER_SAVE_DIR)
tokenizer.save_pretrained(IMAGE_ADAPTER_SAVE_DIR)
torch.save(
    prefix_projector.state_dict(),
    f"{IMAGE_ADAPTER_SAVE_DIR}/prefix_projector.pt"
)
print("Saved to:", IMAGE_ADAPTER_SAVE_DIR)

    print("\nCould not find expert_capacity.png")
