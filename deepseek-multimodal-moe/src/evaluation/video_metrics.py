def generate_video_caption_greedy(example_raw, max_new_tokens=MAX_CAPTION_LEN):
    video_model.eval()
    with torch.no_grad():
        video_path = example_raw["video_path"]
        # 1) Load frames
        local_path = download_segment_if_needed(video_path)
        frames = sample_frames_from_video(local_path, num_frames=NUM_FRAMES)
        vision_inputs = image_processor(images=frames, return_tensors="pt")
        frame_pixel_values = vision_inputs["pixel_values"].unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]

        # 2) Temporal encoder + prefix
        video_emb = temporal_encoder(frame_pixel_values)          # [1, D_vision]
        visual_prefix = prefix_projector(video_emb)               # [1, P, H_lm]

        # 3) Greedy decode
        lm_for_gen = video_model.lm_model
        start_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else (tokenizer.pad_token_id or tokenizer.eos_token_id)
        )
        generated = [start_id]

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated], device=DEVICE)  # [1, T]
            emb_layer = lm_for_gen.base_model.model.model.embed_tokens
            text_embeds = emb_layer(input_ids)                    # [1, T, H_lm]

            inputs_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
            attn_mask = torch.ones(
                1,
                visual_prefix.shape[1] + text_embeds.shape[1],
                dtype=torch.long,
                device=DEVICE,
            )

            outputs = lm_for_gen(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
            )
            next_logits = outputs.logits[0, -1, :]
            next_id = int(torch.argmax(next_logits))

            if next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)

        caption = tokenizer.decode(generated, skip_special_tokens=True)
        return caption.strip()

print("\n=== Evaluating VIDEO modality with BLEU (YouCook2 subset) ===")

refs = {}
hyps = {}

for i, ex in enumerate(youcook_eval):
    gen_caption = generate_video_caption_greedy(ex, max_new_tokens=32)
    ref_caption = ex["caption_text"]

    refs[i] = [ref_caption]
    hyps[i] = [gen_caption]

    if i < 3:
        print(f"\nExample {i}")
        print("REF:", ref_caption)
        print("GEN:", gen_caption)

bleu_scorer = Bleu(n=4)
bleu_scores, bleu_per_sentence = bleu_scorer.compute_score(refs, hyps)

print("\n--- YouCook2 BLEU Scores (subset) ---")
print(f"BLEU-1: {bleu_scores[0]:.4f}")
print(f"BLEU-2: {bleu_scores[1]:.4f}")
print(f"BLEU-3: {bleu_scores[2]:.4f}")
print(f"BLEU-4: {bleu_scores[3]:.4f}")
