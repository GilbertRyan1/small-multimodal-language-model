!pip install -q transformers bitsandbytes peft

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"

# Map modality names → your adapter dirs
ADAPTER_PATHS = {
    "text":  "/content/drive/MyDrive/final-adapter",              # BoolQ
    "logic": "/content/drive/MyDrive/logic_ai2_ar/final-adapter", # ARC-Challenge
    "math":  "/content/drive/MyDrive/math_gsm8k_from_logic",      # GSM8K
    "code":  "/content/drive/MyDrive/code_codeparrot",            # CodeParrot subset
    "image": "/content/drive/MyDrive/image_coco_from_code",       # COCO captions
    "video": "/content/drive/MyDrive/video_youcook2_from_image",  # YouCook2 segments
}

multi_model, tokenizer = build_multimodal_deepseek(
    base_model_id=BASE_MODEL_ID,
    adapter_paths=ADAPTER_PATHS,
    device=DEVICE,
)

print(" MultimodalDeepSeek generalist wrapper ready.")

print("\n=== TEXT example ===")
print(multi_model.generate_text(
    "Question: Is the Eiffel Tower in Paris?\n\nAnswer:",
    max_new_tokens=10
))

print("\n=== LOGIC example ===")
print(multi_model.generate_logic(
    "Question: 1, 1, 2, 3, 5, ?\nChoices:\n(A) 7\n(B) 8\n(C) 9\n(D) 11\n\nAnswer:",
    max_new_tokens=4
))

print("\n=== MATH example ===")
print(multi_model.generate_math(
    "Solve: A train travels 120 km in 3 hours. What is its speed in km/h?\n\nAnswer:",
    max_new_tokens=32
))

print("\n=== CODE example ===")
print(multi_model.generate_code(
    "Write a Python function that checks if a number is prime.\n\n```python\n",
    max_new_tokens=64
))

from PIL import Image
import os

image_path = "/content/sample.jpg"  # change to a real image path

if os.path.exists(image_path):
    pil_img = Image.open(image_path)
    print("\n=== IMAGE example ===")
    caption = multi_model.caption_image(pil_img, max_new_tokens=32)
    print("Image caption:", caption)
else:
    print("\n[IMAGE example skipped] Set 'image_path' to a real image file.")

import cv2
import numpy as np

def sample_frames_from_video_file(path, num_frames=4):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from video: {path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames

video_path = "/content/sample_video.mp4"  # change to a real video path

if os.path.exists(video_path):
    frames = sample_frames_from_video_file(video_path, num_frames=4)
    print("\n=== VIDEO example ===")
    v_caption = multi_model.caption_video_frames(frames, max_new_tokens=32)
    print("Video caption:", v_caption)
else:
    print("\n[VIDEO example skipped] Set 'video_path' to a real mp4 file.")
