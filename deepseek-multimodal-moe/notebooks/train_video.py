!pip install -q transformers datasets accelerate bitsandbytes peft pillow torchvision pycocoevalcap opencv-python huggingface_hub

import os
import cv2
import zipfile
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from datasets import load_dataset
from huggingface_hub import hf_hub_download

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
from pycocoevalcap.bleu.bleu import Bleu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Core model 
BASE_MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"

# CHANGE THIS: path to your trained IMAGE adapter (COCO)
PREV_ADAPTER_PATH = "/content/drive/MyDrive/image_coco_from_code"  # <-- UPDATE THIS

# New adapter for VIDEO modality
VIDEO_ADAPTER_NAME = "video_youcook2"
VIDEO_ADAPTER_SAVE_DIR = "/content/drive/MyDrive/video_youcook2_from_image"  # <-- change to yours
# Vision encoder (CLIP)
VISION_MODEL_ID = "openai/clip-vit-base-patch32"
PREFIX_LENGTH = 10
MAX_CAPTION_LEN = 64
NUM_FRAMES = 4   # frames per video segment

# Dataset config
YOUCOOK_DATASET_ID = "lmms-lab/YouCook2"
TRAIN_SAMPLES = 100
EVAL_SAMPLES  = 10

# Local cache for extracted mp4 segments
VIDEO_CACHE_DIR = "/content/youcook2_videos"
os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)

# ---- Download big YouCook2 video zip and build index ----
print("Downloading YouCookIIVideos.zip from lmms-lab/YouCook2 (large file)...")
YCOOK_ZIP_PATH = hf_hub_download(
    repo_id="lmms-lab/YouCook2",
    filename="YouCookIIVideos.zip",
    repo_type="dataset"
)
print("Zip path:", YCOOK_ZIP_PATH)

youcook_zip = zipfile.ZipFile(YCOOK_ZIP_PATH, "r")
all_members = youcook_zip.namelist()
print("Total entries in zip:", len(all_members))

# Build index: 'val/xHr8X2Wpmno_0.mp4' -> full path inside zip
zip_index = {}
for name in all_members:
    if not name.lower().endswith(".mp4"):
        continue
    parts = name.split("/")
    if len(parts) >= 2:
        key = "/".join(parts[-2:])   # e.g. 'val/xHr8X2Wpmno_0.mp4'
    else:
        key = parts[-1]
    zip_index[key] = name

print("Indexed mp4 files:", len(zip_index))
sample_keys = list(zip_index.keys())[:5]
print("Sample keys:", sample_keys)
print("Example:", sample_keys[0], "->", zip_index[sample_keys[0]])

def download_segment_if_needed(video_path: str) -> str:
    """
    Extract a single YouCook2 segment mp4 from the big zip into VIDEO_CACHE_DIR.
    video_path is like 'val/xHr8X2Wpmno_0.mp4'.
    """
    local_name = video_path.replace("/", "_")
    local_path = os.path.join(VIDEO_CACHE_DIR, local_name)

    if os.path.exists(local_path):
        return local_path

    if video_path not in zip_index:
        raise ValueError(
            f"Video path {video_path} not found in zip index. "
            f"Example keys: {list(zip_index.keys())[:5]}"
        )

    zip_member = zip_index[video_path]
    print(f"Extracting {video_path} from zip entry {zip_member} ...")

    with youcook_zip.open(zip_member) as src, open(local_path, "wb") as dst:
        dst.write(src.read())

    return local_path

print("Loading YouCook2 dataset (val split)...")
youcook_full = load_dataset(YOUCOOK_DATASET_ID, split="val")
print("Total dataset size:", len(youcook_full))

# Small train/eval subsets
youcook_train = youcook_full.select(range(TRAIN_SAMPLES))
youcook_eval  = youcook_full.select(range(TRAIN_SAMPLES, TRAIN_SAMPLES + EVAL_SAMPLES))

def build_caption_text(example):
    """
    For lmms-lab/YouCook2:
      - 'sentence' is the segment-level caption.
    """
    cap = example.get("sentence", "")
    if not cap:
        cap = "[no caption]"
    example["caption_text"] = cap
    return example

youcook_train = youcook_train.map(build_caption_text)
youcook_eval  = youcook_eval.map(build_caption_text)

print("Example eval keys:", youcook_eval[0].keys())
print("Example eval caption_text:", repr(youcook_eval[0]["caption_text"]))
print("Example eval video_path:", youcook_eval[0]["video_path"])  # e.g. 'val/xHr8X2Wpmno_0.mp4'

# Tokenizer for DeepSeek
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# CLIP image processor (for frames)
image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_ID)

def preprocess_youcook_text(batch):
    text_out = tokenizer(
        batch["caption_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_CAPTION_LEN,
        return_tensors="pt",
    )
    input_ids = text_out["input_ids"]
    attention_mask = text_out["attention_mask"]

    labels = []
    for seq in input_ids:
        labels.append([
            tok if tok != tokenizer.pad_token_id else -100
            for tok in seq
        ])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "video_path": batch["video_path"],
        "caption_text": batch["caption_text"],
    }

youcook_train_proc = youcook_train.map(
    preprocess_youcook_text,
    batched=True,
    remove_columns=[c for c in youcook_train.column_names if c not in ["video_path", "caption_text"]],
)
youcook_eval_proc = youcook_eval.map(
    preprocess_youcook_text,
    batched=True,
    remove_columns=[c for c in youcook_eval.column_names if c not in ["video_path", "caption_text"]],
)

youcook_train_proc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
youcook_eval_proc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("Processed train:", len(youcook_train_proc), "eval:", len(youcook_eval_proc))
print("Processed example keys:", youcook_eval_proc[0].keys())

from tqdm.auto import tqdm

print("Pre-extracting YouCook2 segment videos for train + eval subsets...")
all_missing = 0

for ex in tqdm(list(youcook_train) + list(youcook_eval)):
    vp = ex["video_path"]  # e.g. 'val/xHr8X2Wpmno_0.mp4'
    try:
        local_path = download_segment_if_needed(vp)
        if not os.path.exists(local_path):
            all_missing += 1
    except Exception as e:
        print("Error for", vp, ":", e)
        all_missing += 1

print("Pre-extraction complete.")
print("Missing files after pass:", all_missing)
print("Cached files in VIDEO_CACHE_DIR:", len(os.listdir(VIDEO_CACHE_DIR)))

def sample_frames_from_video(local_path: str, num_frames: int = NUM_FRAMES):
    """
    Use OpenCV to sample num_frames evenly spaced frames from an mp4.
    Returns list of PIL.Image frames.
    """
    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        # fallback: one black frame repeated
        h, w = 224, 224
        return [Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))] * num_frames

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        h, w = 224, 224
        cap.release()
        return [Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))] * num_frames

    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            h, w = 224, 224
            frames.append(Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames

from torch.utils.data import Dataset

class YouCook2VideoDataset(Dataset):
    def __init__(self, hf_ds_raw, hf_ds_proc):
        assert len(hf_ds_raw) == len(hf_ds_proc)
        self.raw = hf_ds_raw
        self.proc = hf_ds_proc

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        item_raw = self.raw[idx]
        item_proc = self.proc[idx]

        video_path = item_raw["video_path"]          # 'val/xxx.mp4'
        caption_text = item_raw["caption_text"]
        input_ids = torch.tensor(item_proc["input_ids"])
        attention_mask = torch.tensor(item_proc["attention_mask"])
        labels = torch.tensor(item_proc["labels"])

        # Ensure segment is extracted
        local_path = download_segment_if_needed(video_path)
        frames = sample_frames_from_video(local_path, num_frames=NUM_FRAMES)

        vision_inputs = image_processor(images=frames, return_tensors="pt")
        frame_pixel_values = vision_inputs["pixel_values"]  # [T, C, H, W]

        return {
            "frame_pixel_values": frame_pixel_values,  # [T, C, H, W]
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "video_path": video_path,
            "caption_text": caption_text,
        }

def video_collate_fn(batch):
    frame_pixel_values = torch.stack([b["frame_pixel_values"] for b in batch])  # [B, T, C, H, W]
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    return {
        "frame_pixel_values": frame_pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

train_ds_video = YouCook2VideoDataset(youcook_train, youcook_train_proc)
eval_ds_video  = YouCook2VideoDataset(youcook_eval, youcook_eval_proc)

print("Custom video datasets built:",
      len(train_ds_video), "train examples,", len(eval_ds_video), "eval examples")

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

# ---- Load previous IMAGE adapter ----
print("Loading previous IMAGE adapter from:", PREV_ADAPTER_PATH)
lm_with_prev = PeftModel.from_pretrained(
    base_model,
    PREV_ADAPTER_PATH,
    is_trainable=False,
)

# ---- Add new VIDEO adapter on top ----
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

lm_with_prev.add_adapter(VIDEO_ADAPTER_NAME, lora_config)
lm_with_prev.set_adapter(VIDEO_ADAPTER_NAME)
lm_with_prev.to(DEVICE)

print("Trainable parameters in LM (with video adapter):")
lm_with_prev.print_trainable_parameters()

# ---- CLIP vision encoder ----
vision_model = CLIPVisionModel.from_pretrained(VISION_MODEL_ID).to(DEVICE)
for p in vision_model.parameters():
    p.requires_grad = False

vision_hidden_size = vision_model.config.hidden_size
lm_hidden_size = lm_with_prev.base_model.model.model.embed_tokens.embedding_dim

print("Vision hidden size:", vision_hidden_size)
print("LM hidden size:", lm_hidden_size)

# ---- Temporal encoder over frames ----
class TemporalEncoder(nn.Module):
    def __init__(self, vision_dim, num_frames, num_layers=1, num_heads=8):
        super().__init__()
        self.num_frames = num_frames
        self.vision_dim = vision_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vision_dim,
            nhead=num_heads,
            dim_feedforward=vision_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, vision_dim))

    def forward(self, frame_pixel_values):
        # frame_pixel_values: [B, T, C, H, W]
        B, T, C, H, W = frame_pixel_values.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"

        x = frame_pixel_values.view(B * T, C, H, W)
        with torch.no_grad():
            vision_outputs = vision_model(pixel_values=x)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            frame_emb = vision_outputs.pooler_output   # [B*T, D]
        else:
            frame_emb = vision_outputs.last_hidden_state[:, 0, :]
        x = frame_emb.view(B, T, self.vision_dim)
        x = x + self.pos_embed
        x = self.transformer(x)                        # [B, T, D]
        video_emb = x.mean(dim=1)                      # [B, D]
        return video_emb

# ---- Prefix projector ----
class VideoPrefixProjector(nn.Module):
    def __init__(self, vision_dim, lm_dim, prefix_len):
        super().__init__()
        self.prefix_len = prefix_len
        self.lm_dim = lm_dim
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, lm_dim * prefix_len),
            nn.Tanh(),
        )

    def forward(self, video_emb):
        x = self.proj(video_emb)                      # [B, prefix_len * lm_dim]
        x = x.view(-1, self.prefix_len, self.lm_dim)  # [B, prefix_len, lm_dim]
        return x

temporal_encoder = TemporalEncoder(
    vision_dim=vision_hidden_size,
    num_frames=NUM_FRAMES,
    num_layers=1,
    num_heads=8,
).to(DEVICE)

prefix_projector = VideoPrefixProjector(
    vision_dim=vision_hidden_size,
    lm_dim=lm_hidden_size,
    prefix_len=PREFIX_LENGTH,
).to(DEVICE)

print("Temporal encoder trainable params:",
      sum(p.numel() for p in temporal_encoder.parameters()))
print("Prefix projector trainable params:",
      sum(p.numel() for p in prefix_projector.parameters()))

# ---- Video caption model wrapper ----
class VideoCaptionModel(nn.Module):
    def __init__(self, temporal_encoder, lm_model, projector, tokenizer, prefix_len):
        super().__init__()
        self.temporal_encoder = temporal_encoder
        self.lm_model = lm_model
        self.projector = projector
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len

    def forward(
        self,
        frame_pixel_values=None,   # [B, T, C, H, W]
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        video_emb = self.temporal_encoder(frame_pixel_values.to(DEVICE))   # [B, D_vision]
        visual_prefix = self.projector(video_emb)                          # [B, P, H_lm]

        emb_layer = self.lm_model.base_model.model.model.embed_tokens
        text_embeds = emb_layer(input_ids.to(DEVICE))                      # [B, T, H_lm]

        inputs_embeds = torch.cat([visual_prefix, text_embeds], dim=1)     # [B, P+T, H_lm]

        if attention_mask is not None:
            bsz = attention_mask.shape[0]
            prefix_mask = torch.ones(
                bsz, self.prefix_len,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attn_mask = torch.cat([prefix_mask, attention_mask.to(DEVICE)], dim=1)
        else:
            attn_mask = None

        if labels is not None:
            labels = labels.to(DEVICE)
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

video_model = VideoCaptionModel(
    temporal_encoder=temporal_encoder,
    lm_model=lm_with_prev,
    projector=prefix_projector,
    tokenizer=tokenizer,
    prefix_len=PREFIX_LENGTH,
).to(DEVICE)

print("Video caption model ready.")

from torch.utils.data import DataLoader

def trainer_collate_fn(batch):
    # reuse video_collate_fn but ensure tensors
    return video_collate_fn(batch)

training_args = TrainingArguments(
    output_dir=VIDEO_ADAPTER_SAVE_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="no",
    num_train_epochs=1,
    max_steps=100,          # smaller for video
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=video_model,
    args=training_args,
    train_dataset=train_ds_video,
    eval_dataset=eval_ds_video,
    data_collator=trainer_collate_fn,
    tokenizer=tokenizer,
)

print("Starting VIDEO (YouCook2) training...")
train_result = trainer.train()
print("VIDEO training complete.")

print("Saving VIDEO adapter and temporal/prefix modules...")
video_model.lm_model.save_pretrained(VIDEO_ADAPTER_SAVE_DIR)
tokenizer.save_pretrained(VIDEO_ADAPTER_SAVE_DIR)
torch.save(
    temporal_encoder.state_dict(),
    f"{VIDEO_ADAPTER_SAVE_DIR}/temporal_encoder.pt"
)
torch.save(
    prefix_projector.state_dict(),
    f"{VIDEO_ADAPTER_SAVE_DIR}/video_prefix_projector.pt"
)
print("Saved to:", VIDEO_ADAPTER_SAVE_DIR)
    print("\nCould not find expert_capacity.png")
