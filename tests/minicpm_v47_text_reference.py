#!/usr/bin/env python3
"""Official CPU VQA reference using separately verified HF visual features.

This is a test oracle, never a runtime backend. Splitting the official model at
its public mm_encoder_outputs interface avoids holding both FP32 towers in RAM.
The feature file must come from the official vision reference for this image.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7ForConditionalGeneration
from transformers.models.minicpmv4_6.image_processing_pil_minicpmv4_6 import MiniCPMV4_6ImageProcessorPil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from minicpm_v47_export import load_language_only
from minicpm_streamed_reference import bind_streamed_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("hf_features")
    parser.add_argument("output")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-mode", choices=("streamed", "full"), default="streamed",
                        help="streamed retains only decoder weights, reads embedding rows/head chunks on demand")
    parser.add_argument("--head-chunk-rows", type=int, default=16384)
    parser.add_argument("--prompt", default="Describe this image in one short sentence.")
    args = parser.parse_args()
    if args.tokens < 1 or args.threads < 1 or args.head_chunk_rows < 1:
        parser.error("--tokens, --threads and --head-chunk-rows must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor = MiniCPMV4_6ImageProcessorPil.from_pretrained(args.model, local_files_only=True)
    # The checkpoint's PAD is also an EOS token. Do not infer attention masks
    # by comparing token IDs to PAD, or silently substitute a different token.
    # Single-image oracle deliberately uses no padding; decode below relies on it.
    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image", "image": image},
                                                {"type": "text", "text": args.prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", downsample_mode="16x", max_slice_nums=1,
        use_image_id=False, enable_thinking=False, padding=False)
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    required = ("input_ids", "pixel_values", "target_sizes", "mm_token_type_ids")
    if any(name not in inputs for name in required):
        raise ValueError(f"processor is missing required fields: {set(required) - set(inputs)}")
    if inputs["input_ids"].shape[0] != 1 or inputs["target_sizes"].shape != (1, 2):
        raise ValueError("reference supports exactly one unpadded image thumbnail")
    if "attention_mask" in inputs and not bool(inputs["attention_mask"].all()):
        raise ValueError("padded reference prompts are unsupported")
    mask = inputs["input_ids"] == cfg.image_token_id
    if not torch.equal(inputs["mm_token_type_ids"] == 1, mask):
        raise ValueError("processor image modality labels disagree with image tokens")
    grid = inputs["target_sizes"][0]
    if bool((grid % 4 != 0).any()) or int(mask.sum()) != int(grid.prod()) // 16:
        raise ValueError("processor visual token count does not match 16x target grid")
    for marker in (cfg.image_start_id, cfg.image_end_id):
        if int((inputs["input_ids"] == marker).sum()) != 1:
            raise ValueError("expected one image start and end marker")
    if any(bool((inputs["input_ids"] == marker).any()) for marker in
           (cfg.slice_start_id, cfg.slice_end_id)):
        raise ValueError("detail slices are unsupported in this reference")
    (output / "prompt.txt").write_text(processor.decode(inputs["input_ids"][0],
                                                       skip_special_tokens=False))
    for name in ("input_ids", "pixel_values", "target_sizes", "mm_token_type_ids"):
        if name in inputs:
            np.save(output / (name + ".npy"), inputs[name].numpy())
    features = torch.from_numpy(np.load(args.hf_features, allow_pickle=False)).float()
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    expected_shape = (int(mask.sum()), cfg.text_config.hidden_size)
    if tuple(features.shape) != expected_shape or not bool(torch.isfinite(features).all()):
        raise ValueError(f"invalid official visual features: {tuple(features.shape)}, expected {expected_shape}")
    # from_config/meta construction does NOT load generation_config.json.
    # Preserve both checkpoint stop IDs, not merely tokenizer.eos_token_id.
    generation_config = GenerationConfig.from_pretrained(args.model, local_files_only=True)
    eos_value = generation_config.eos_token_id
    if eos_value is None:
        eos_value = processor.tokenizer.eos_token_id
    eos = {eos_value} if isinstance(eos_value, int) else set(eos_value or [])
    if not eos:
        raise ValueError("no checkpoint EOS IDs available")
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = processor.tokenizer.pad_token_id
    print(f"Prepared {inputs['input_ids'].shape[-1]} prompt tokens, "
          f"{features.shape[0]} official visual tokens; EOS={sorted(eos)}", flush=True)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = MiniCPMV4_7ForConditionalGeneration(cfg)
    language = load_language_only(args.model, cfg,
                                  component="decoder" if args.memory_mode == "streamed" else "all")
    model.model.language_model = language.model.language_model
    model.lm_head = language.lm_head
    del language
    if args.memory_mode == "streamed":
        bind_streamed_tables(model, args.model, args.head_chunk_rows)
    model.generation_config = generation_config
    model.eval()
    assert not any(p.is_meta for p in model.model.language_model.parameters())
    assert all(p.is_meta for p in model.model.vision_tower.parameters())
    assert all(p.is_meta for p in model.model.merger.parameters())
    assert not any(p.is_meta for p in model.lm_head.parameters())
    # Fail visibly if supplied features accidentally stop bypassing either tower.
    def reject_vision_call(module, args):
        raise RuntimeError("precomputed visual reference must not execute meta vision modules")
    vision_hooks = [model.model.vision_tower.register_forward_pre_hook(reject_vision_call),
                    model.model.merger.register_forward_pre_hook(reject_vision_call)]
    print("Loaded real language weights; using official precomputed visual features", flush=True)
    inputs.pop("pixel_values", None)
    inputs["mm_encoder_outputs"] = {"image": BaseModelOutputWithPooling(pooler_output=(features,))}
    inputs["downsample_mode"] = "16x"
    positions, deltas = model.model.get_rope_index(
        inputs["input_ids"], mm_token_type_ids=inputs.get("mm_token_type_ids"),
        target_sizes=inputs["target_sizes"], downsample_mode="16x",
        attention_mask=inputs.get("attention_mask"))
    np.save(output / "positions.npy", positions.numpy())
    generated = []
    started = time.time()
    with torch.no_grad():
        result = model(**inputs, use_cache=True, logits_to_keep=1)
        if result.past_key_values is None:
            raise RuntimeError("official prefill returned no cache")
        if result.past_key_values.get_seq_length() != inputs["input_ids"].shape[-1]:
            raise RuntimeError("unexpected prefill cache sequence length")
        torch.testing.assert_close(model.model.rope_deltas, deltas, rtol=0, atol=0)
        if not bool(torch.isfinite(result.logits).all()):
            raise RuntimeError("non-finite prefill logits")
        np.save(output / "prefill_logits.npy", result.logits[0, -1].float().numpy())
        for step in range(args.tokens):
            token = int(result.logits[0, -1].argmax())
            generated.append(token)
            print("token", step, token, repr(processor.decode([token])), flush=True)
            if token in eos or step + 1 == args.tokens:
                break
            previous_length = result.past_key_values.get_seq_length()
            # No attention_mask on decode is intentional: this oracle rejects
            # padding. The official model derives one new canvas position from
            # cache length + its saved rope_delta. Passing a full-length mask
            # directly would produce full-length position IDs in this HF method.
            result = model(input_ids=torch.tensor([[token]], dtype=torch.long),
                           past_key_values=result.past_key_values,
                           use_cache=True, logits_to_keep=1)
            if result.past_key_values.get_seq_length() != previous_length + 1:
                raise RuntimeError("official decode did not advance cache by one")
            if not bool(torch.isfinite(result.logits).all()):
                raise RuntimeError("non-finite decode logits")
    for hook in vision_hooks:
        hook.remove()
    report = {"generated_ids": generated,
              "text": processor.decode(generated, skip_special_tokens=True),
              "seconds": time.time() - started, "image": str(args.image),
              "input_ids": inputs["input_ids"][0].tolist(),
              "image_processor": type(processor.image_processor).__name__,
              "reference": "official HF forward with official precomputed vision features",
              "dtype": "float32", "device": "cpu", "rope_deltas": deltas.tolist(),
              "eos_token_ids": sorted(eos), "pad_token_id": generation_config.pad_token_id,
              "stop_reason": "eos" if generated[-1] in eos else "max_tokens",
              "hf_features": str(args.hf_features), "visual_feature_shape": list(features.shape),
              "memory_mode": args.memory_mode, "head_chunk_rows": args.head_chunk_rows,
              "streamed_operations": ["checkpoint embedding rows", "FP32 chunked linear head"]
                 if args.memory_mode == "streamed" else []}
    (output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(report["text"], flush=True)


if __name__ == "__main__":
    main()
