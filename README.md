# DeepSeek Multimodal MoE
**IAISS 2025 Project — Ryan Gilbert**

This project implements a **small multimodal Mixture-of-Experts (MoE)** model based on the `deepseek-ai/deepseek-moe-16b-base` transformer.  
It integrates **six modalities** — text, logic, math, code, image, and video — through modality-specific LoRA adapters and shared experts.

---

##  Modalities

| Modality | Dataset | Metric |
|-----------|----------|--------|
| Text | BoolQ | Accuracy, F1 |
| Logic | ARC-Challenge | Accuracy, F1 |
| Math | GSM8K | Exact Match |
| Code | CodeParrot subset | Pass@k |
| Image | COCO Captions | CIDEr |
| Video | YouCook2 | BLEU |

Each modality was fine-tuned as a LoRA adapter on top of the DeepSeek-MoE base.  
Adapters were then combined into a **generalist model** with a unified inference API.

---

##  Quick Start (Colab Ready)

```bash
!git clone https://github.com/<your-username>/deepseek-multimodal-moe.git
%cd deepseek-multimodal-moe
!pip install -r requirements.txt
