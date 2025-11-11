def generate_caption_greedy(image_pil, max_new_tokens=MAX_CAPTION_LEN):
    vl_model.eval()
    with torch.no_grad():
        # 1) Image -> pixel_values
        vision_inputs = image_processor(images=image_pil.convert("RGB"), return_tensors="pt")
        pixel_values = vision_inputs["pixel_values"].to(DEVICE)

        # 2) Vision -> prefix
        vision_outputs = vision_model(pixel_values=pixel_values)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            vision_emb = vision_outputs.pooler_output
        else:
            vision_emb = vision_outputs.last_hidden_state[:, 0, :]

        visual_prefix = prefix_projector(vision_emb)   # [1, P, H]

        # 3) Greedy decode using LM token embeddings
        start_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else (tokenizer.pad_token_id or tokenizer.eos_token_id)
        )
        generated = [start_id]

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated], device=DEVICE)   # [1, T]
            emb_layer = lm_with_prev.base_model.model.model.embed_tokens
            text_embeds = emb_layer(input_ids)                    # [1, T, H]

            inputs_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
            attn_mask = torch.ones(
                1,
                visual_prefix.shape[1] + text_embeds.shape[1],
                dtype=torch.long,
                device=DEVICE,
            )

            outputs = lm_with_prev(
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

print("\n=== Evaluating IMAGE modality with CIDEr (COCO subset) ===")

refs = {}
hyps = {}

for i, ex in enumerate(coco_eval):
    img = ex["image"]            # PIL image
    answers = ex["answer"]       # list of 5 reference captions
    gen_caption = generate_caption_greedy(img, max_new_tokens=32)

    refs[i] = answers
    hyps[i] = [gen_caption]

    if i < 3:
        print(f"\nExample {i}")
        print("REFS:")
        for r in answers:
            print(" -", r)
        print("GEN:", gen_caption)

cider_scorer = Cider()
cider_score, cider_scores = cider_scorer.compute_score(refs, hyps)

print(f"\n--- COCO CIDEr Score (subset) ---")
print(f"CIDEr: {cider_score:.4f}")

