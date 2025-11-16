# Small Multimodal Mixture of Experts Model Based on DeepSeek MoE

This repository contains the implementation of a multimodal Mixture of Experts system built on the DeepSeek MoE 16B model. The model is extended with LoRA adapters for six modalities: text, logic, math, code, image, and video. A unified generalist inference model is provided to run all modalities within a single interface. All training, evaluation, and routing diagnostic scripts are included in a structured and reproducible format.

Adapters required for inference are provided separately through Google Drive due to size constraints. Instructions for loading them in Colab are included below.

---

## Repository Structure

```
deepseek-multimodal-moe/
    notebooks/
        train_text.py
        train_code.py
        train_logic.py
        train_math.py
        train_image.py
        train_video.py
        generalist_inference.py

    src/
        data/
            inference_sample_image.jpg
            sample_cooking_video.mp4

        model/
            generalist_model.py

        evaluation/
            text_metrics.py
            code_metrics.py
            logic_metrics.py
            math_metrics.py
            image_metrics.py
            video_metrics.py
            moe_router.py
            results/

    datasets/
        Instructions for downloading datasets

    requirements.txt
    LICENSE
```

---

## Installation

The project to be executed on Google Colab

Install requirements:

```python
%pip install -r requirements.txt
```

The model uses 4 bit quantization through bitsandbytes for compatibility with standard Colab GPUs.

---

## Loading Adapters from Google Drive

Six LoRA adapters are required for inference. These are provided in a Google Drive folder:

```
multi-modality_adapter/
    text_adapter/
    logic_adapter/
    math_adapter/
    code_adapter/
    image_adapter/
    video_adapter/
```

Place the folder in your Google Drive.

### Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Set the adapter directory path

```python
ADAPTER_DIR = "/content/drive/MyDrive/multi-modality_adapter"
```

Modify the path if you store the folder elsewhere.

### Step 3: Define adapter paths

```python
ADAPTERS = {
    "text":  f"{ADAPTER_DIR}/text_adapter",
    "logic": f"{ADAPTER_DIR}/logic_adapter",
    "math":  f"{ADAPTER_DIR}/math_adapter",
    "code":  f"{ADAPTER_DIR}/code_adapter",
    "image": f"{ADAPTER_DIR}/image_adapter",
    "video": f"{ADAPTER_DIR}/video_adapter",
}
```

### Step 4: Verify adapter paths

```python
import os
for name, path in ADAPTERS.items():
    print(name, path, "exists:", os.path.exists(path))
```

---

## Generalist Model

Load the unified multimodal model:

```python
from notebooks.generalist_inference import MultiModalGeneralist

generalist = MultiModalGeneralist(adapter_paths=ADAPTERS)
```

Set the active modality before running inference.

---

## Inference Examples

### Text

```python
generalist.set_modality("text")
generalist.generate_text("Question: Is Paris in France. Answer:", max_new_tokens=20)
```

### Logic

```python
generalist.set_modality("logic")
generalist.generate_text("Question: Which is heavier. One kilogram of feathers or one kilogram of steel. Answer:", max_new_tokens=20)
```

### Math

```python
generalist.set_modality("math")
generalist.generate_text("Q: A bag contains 12 apples. If 5 are eaten, how many remain. A:", max_new_tokens=20)
```

### Code

```python
generalist.set_modality("code")
generalist.generate_text("Write a Python function that checks if a string is a palindrome.", max_new_tokens=120)
```

### Image

Sample image available in:

```
src/data/inference_sample_image.jpg
```

```python
generalist.set_modality("image")
generalist.generate_image_caption("deepseek-multimodal-moe/src/data/inference_sample_image.jpg")
```

### Video

Sample video available in:

```
src/data/sample_cooking_video.mp4
```

```python
generalist.set_modality("video")
generalist.generate_video_caption("deepseek-multimodal-moe/src/data/sample_cooking_video.mp4")
```

---
##  Training Dataset as per modality

| Modality | Dataset |
|-----------|----------|
| Text | BoolQ |
| Logic | ARC-Challenge |
| Math | GSM8K |
| Code | CodeParrot subset |
| Image | COCO Captions |
| Video | YouCook2 |

Each modality was fine-tuned as a LoRA adapter on top of the DeepSeek-MoE base.  
Adapters were then combined into a **generalist model** with a unified inference API.

## Training Scripts

Training scripts for all modalities are provided under the notebooks directory. Each script is compatible with Colab and includes data loading, preprocessing, adapter training, and checkpoint saving.

```
train_text.py
train_code.py
train_logic.py
train_math.py
train_image.py
train_video.py
```

---

## Evaluation Metrics

Evaluation scripts for all modalities are located in:

```
src/evaluation/
```

Metrics include:

* Accuracy and macro F1 for text and logic
* Exact Match for math reasoning
* Pass at k for code generation
* CIDEr for image captioning
* BLEU for video captioning

---

## MoE Routing Diagnostics

The MoE router monitor records expert capacity utilization and routing entropy.
Plots and numerical data for each modality are available under:

```
src/evaluation/results/
```

These diagnostics show how tokens are distributed across experts and whether routing is balanced during inference.

---

## Dataset Instructions

Each modality requires publicly available datasets referenced in the datasets directory.
Links and download steps are included in that folder.

---

## License

This project is released under the MIT License.

---

##  Quick Start (Colab Ready)

```bash
!git clone https://github.com/GilbertRyan1/deepseek-multimodal-moe.git
%cd deepseek-multimodal-moe
!pip install -r requirements.txt
