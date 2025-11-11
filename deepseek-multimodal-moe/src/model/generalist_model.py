import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPVisionModel,
    AutoImageProcessor,
)
from peft import PeftModel
from PIL import Image


class ImagePrefixProjector(nn.Module):
    def __init__(self, vision_dim, lm_dim, prefix_len):
        super().__init__()
        self.prefix_len = prefix_len
        self.lm_dim = lm_dim
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, lm_dim * prefix_len),
            nn.Tanh(),
        )

    def forward(self, vision_emb):
        x = self.proj(vision_emb)
        return x.view(-1, self.prefix_len, self.lm_dim)


class TemporalEncoder(nn.Module):
    def __init__(self, vision_dim, num_frames, num_layers=1, num_heads=8):
        super().__init__()
        self.num_frames = num_frames
        self.vision_dim = vision_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=vision_dim,
            nhead=num_heads,
            dim_feedforward=vision_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, vision_dim))

    def forward(self, frame_pixel_values, vision_model):
        """
        frame_pixel_values: [B, T, C, H, W]
        """
        B, T, C, H, W = frame_pixel_values.shape
        x = frame_pixel_values.view(B * T, C, H, W)
        with torch.no_grad():
            vo = vision_model(pixel_values=x)
        if hasattr(vo, "pooler_output") and vo.pooler_output is not None:
            frame_emb = vo.pooler_output
        else:
            frame_emb = vo.last_hidden_state[:, 0, :]
        x = frame_emb.view(B, T, self.vision_dim) + self.pos_embed
        x = self.transformer(x)
        return x.mean(dim=1)  # [B, D]


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
        x = self.proj(video_emb)
        return x.view(-1, self.prefix_len, self.lm_dim)


class MultimodalDeepSeek:
    """
    Generalist wrapper around DeepSeek-MoE + 6 LoRA adapters + vision/video heads.
    Uses manual greedy decoding for text to avoid DynamicCache issues.
    """

    def __init__(
        self,
        model,
        tokenizer,
        vision_model,
        image_processor,
        image_prefix_projector,
        temporal_encoder,
        video_prefix_projector,
        device="cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.vision_model = vision_model
        self.image_processor = image_processor
        self.image_prefix_projector = image_prefix_projector
        self.temporal_encoder = temporal_encoder
        self.video_prefix_projector = video_prefix_projector
        self.device = device

    # --- helper: switch adapter ---
    def _set_adapter(self, name: str):
        self.model.set_adapter(name)

    # --- helper: manual greedy for pure-text prompts ---
    def _greedy_decode_text(self, prompt: str, max_new_tokens: int):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        for _ in range(max_new_tokens):
            attn_mask = torch.ones_like(input_ids, device=self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    use_cache=False,   # IMPORTANT: avoid DynamicCache bug
                )

            next_logits = outputs.logits[:, -1, :]   # [1, vocab]
            next_token_id = int(torch.argmax(next_logits, dim=-1).item())

            if next_token_id == self.tokenizer.eos_token_id:
                break

            next_token = torch.tensor([[next_token_id]], device=self.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # -------- TEXT MODALITIES --------
    def generate_text(self, prompt, max_new_tokens=64):
        self._set_adapter("text")
        return self._greedy_decode_text(prompt, max_new_tokens)

    def generate_logic(self, prompt, max_new_tokens=4):
        self._set_adapter("logic")
        return self._greedy_decode_text(prompt, max_new_tokens)

    def generate_math(self, prompt, max_new_tokens=128):
        self._set_adapter("math")
        return self._greedy_decode_text(prompt, max_new_tokens)

    def generate_code(self, prompt, max_new_tokens=128):
        self._set_adapter("code")
        return self._greedy_decode_text(prompt, max_new_tokens)

    # -------- IMAGE CAPTIONING --------
    def caption_image(self, pil_image: Image.Image, max_new_tokens=32):
        self._set_adapter("image")

        vision_inputs = self.image_processor(
            images=pil_image.convert("RGB"),
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            vo = self.vision_model(**vision_inputs)
        if hasattr(vo, "pooler_output") and vo.pooler_output is not None:
            img_emb = vo.pooler_output
        else:
            img_emb = vo.last_hidden_state[:, 0, :]

        prefix = self.image_prefix_projector(img_emb).to(torch.bfloat16)

        lm = self.model
        start_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else (self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        )
        generated = [start_id]

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated], device=self.device)
            emb_layer = lm.base_model.model.model.embed_tokens
            text_embeds = emb_layer(input_ids).to(torch.bfloat16)

            inputs_embeds = torch.cat([prefix, text_embeds], dim=1).to(torch.bfloat16)
            attn_mask = torch.ones(
                1,
                inputs_embeds.shape[1],
                dtype=torch.long,
                device=self.device,
            )

            with torch.no_grad():
                out = lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
            next_id = int(torch.argmax(out.logits[0, -1, :]))

            if next_id == self.tokenizer.eos_token_id:
                break
            generated.append(next_id)

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    # -------- VIDEO CAPTIONING --------
    def caption_video_frames(self, frames, max_new_tokens=32):
        """
        frames: list of PIL.Image frames from a clip (already sampled).
        """
        self._set_adapter("video")

        vision_inputs = self.image_processor(
            images=[f.convert("RGB") for f in frames],
            return_tensors="pt",
        )
        frame_pixel_values = vision_inputs["pixel_values"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            video_emb = self.temporal_encoder(frame_pixel_values, self.vision_model)
            prefix = self.video_prefix_projector(video_emb).to(torch.bfloat16)

        lm = self.model
        start_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else (self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        )
        generated = [start_id]

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated], device=self.device)
            emb_layer = lm.base_model.model.model.embed_tokens
            text_embeds = emb_layer(input_ids).to(torch.bfloat16)

            inputs_embeds = torch.cat([prefix, text_embeds], dim=1).to(torch.bfloat16)
            attn_mask = torch.ones(
                1,
                inputs_embeds.shape[1],
                dtype=torch.long,
                device=self.device,
            )

            with torch.no_grad():
                out = lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
            next_id = int(torch.argmax(out.logits[0, -1, :]))

            if next_id == self.tokenizer.eos_token_id:
                break
            generated.append(next_id)

        return self.tokenizer.decode(generated, skip_special_tokens=True)


def build_multimodal_deepseek(
    base_model_id: str,
    adapter_paths: dict,
    device: str = "cuda",
    prefix_length: int = 10,
    num_frames: int = 4,
):
    """
    High-level builder that:
      - loads DeepSeek in 4bit
      - loads all LoRA adapters
      - loads CLIP vision
      - loads image & video prefix projectors from adapter dirs
      - returns (multi_model, tokenizer)
    """
    DEVICE = device

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- base model in 4-bit ---
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    # --- first adapter to wrap as PeftModel ---
    first_name, first_path = next(iter(adapter_paths.items()))
    model = PeftModel.from_pretrained(
        base_model,
        first_path,
        adapter_name=first_name,
        is_trainable=False,
    )

    # --- load remaining adapters ---
    for name, path in list(adapter_paths.items())[1:]:
        model.load_adapter(path, adapter_name=name)

    model.to(DEVICE)

    # --- shared CLIP vision backbone ---
    VISION_MODEL_ID = "openai/clip-vit-base-patch32"
    image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_ID)
    vision_model = CLIPVisionModel.from_pretrained(VISION_MODEL_ID).to(DEVICE)
    for p in vision_model.parameters():
        p.requires_grad = False

    vision_hidden_size = vision_model.config.hidden_size
    lm_hidden_size = model.base_model.model.model.embed_tokens.embedding_dim

    # --- image prefix projector ---
    image_prefix_projector = ImagePrefixProjector(
        vision_dim=vision_hidden_size,
        lm_dim=lm_hidden_size,
        prefix_len=prefix_length,
    ).to(DEVICE)

    image_prefix_ckpt = os.path.join(adapter_paths["image"], "prefix_projector.pt")
    if not os.path.exists(image_prefix_ckpt):
        raise FileNotFoundError(f"Missing image prefix projector: {image_prefix_ckpt}")
    image_prefix_projector.load_state_dict(
        torch.load(image_prefix_ckpt, map_location=DEVICE)
    )

    # --- video temporal encoder + prefix projector ---
    temporal_encoder = TemporalEncoder(
        vision_dim=vision_hidden_size,
        num_frames=num_frames,
    ).to(DEVICE)

    video_prefix_projector = VideoPrefixProjector(
        vision_dim=vision_hidden_size,
        lm_dim=lm_hidden_size,
        prefix_len=prefix_length,
    ).to(DEVICE)

    video_temporal_ckpt = os.path.join(adapter_paths["video"], "temporal_encoder.pt")
    video_prefix_ckpt = os.path.join(adapter_paths["video"], "video_prefix_projector.pt")

    if not os.path.exists(video_temporal_ckpt):
        raise FileNotFoundError(f"Missing video temporal encoder: {video_temporal_ckpt}")
    if not os.path.exists(video_prefix_ckpt):
        raise FileNotFoundError(f"Missing video prefix projector: {video_prefix_ckpt}")

    temporal_encoder.load_state_dict(
        torch.load(video_temporal_ckpt, map_location=DEVICE)
    )
    video_prefix_projector.load_state_dict(
        torch.load(video_prefix_ckpt, map_location=DEVICE)
    )

    # --- final wrapper ---
    multi_model = MultimodalDeepSeek(
        model=model,
        tokenizer=tokenizer,
        vision_model=vision_model,
        image_processor=image_processor,
        image_prefix_projector=image_prefix_projector,
        temporal_encoder=temporal_encoder,
        video_prefix_projector=video_prefix_projector,
        device=DEVICE,
    )

    return multi_model, tokenizer
