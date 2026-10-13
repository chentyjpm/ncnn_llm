# Copyright (c) 2026 ncnn_llm authors. Apache-2.0.
"""Experimental MiniCPM-V 4.7 export from the original BF16 checkpoint.

The language tower is loaded independently so vision weights are never
duplicated in memory during tracing. The visual exporter streams tensors.
Outputs are FP32 until real-model parity has been established; quantization
must be evaluated separately. This exporter does not read GGUF files.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from transformers import AutoModel

from qwen35_export import export
from minicpm_v47_vision_export import export_minicpm_v47_vision


class LanguageOnly(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = AutoModel.from_config(cfg.text_config,
                                                         attn_implementation="eager")
        self.lm_head = torch.nn.Linear(cfg.text_config.hidden_size,
                                      cfg.text_config.vocab_size, bias=False)


def load_language_only(model_dir, cfg, component="all"):
    if cfg.model_type != "minicpmv4_7" or cfg.text_config.model_type != "qwen3_5_text":
        raise ValueError("expected MiniCPMV4_7 with dense Qwen3.5 language tower")
    if cfg.tie_word_embeddings or cfg.text_config.tie_word_embeddings:
        raise ValueError("this experimental exporter requires untied embeddings")
    from accelerate.utils import set_module_tensor_to_device
    with torch.device("meta"):
        model = LanguageOnly(cfg)
    expected = dict(model.named_parameters())
    if component == "decoder":
        expected = {name: parameter for name, parameter in expected.items()
                    if name not in ("model.language_model.embed_tokens.weight", "lm_head.weight")}
    elif component != "all":
        raise ValueError("component must be all or decoder")
    loaded = set()
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError("original safetensors weights are required")
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name not in expected:
                    continue
                if name in loaded:
                    raise ValueError(f"duplicate checkpoint tensor: {name}")
                tensor = checkpoint.get_tensor(name)
                if tuple(tensor.shape) != tuple(expected[name].shape):
                    raise ValueError(f"checkpoint shape mismatch for {name}")
                set_module_tensor_to_device(model, name, "cpu", value=tensor, dtype=torch.float32)
                loaded.add(name)
                del tensor
    missing = set(expected) - loaded
    if missing:
        raise ValueError(f"missing language tensors: {sorted(missing)[:10]}")
    # Non-persistent rotary buffers were built on meta along with parameters.
    # Recreate their module on CPU using the exact checkpoint config.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    model.model.language_model.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg.text_config, device="cpu")
    if component == "decoder":
        # The decoder consumes inputs_embeds. Do not serialize an unused 1 GB
        # embedding table into its TorchScript container.
        model.model.language_model.embed_tokens = torch.nn.Embedding(1, cfg.text_config.hidden_size)
    model.requires_grad_(False)
    return model.eval()


def export_vision_settings(model_dir, output_dir, prefix="", do_bf16=False):
    if do_bf16:
        raise ValueError("MiniCPM vision parity must be established before quantization")
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    export_minicpm_v47_vision(model_dir, output_dir, prefix=prefix)
    return {
        "type": "minicpm_v4_7",
        "manifest": prefix + "minicpm_v47_vision.json",
        "scale_resolution": 448,
        "slice_mode": True,
        "max_slice_nums": 9,
        "marker_ids": {name: cfg[key] for name, key in (
            ("image", "image_token_id"), ("image_start", "image_start_id"),
            ("image_end", "image_end_id"), ("slice_start", "slice_start_id"),
            ("slice_end", "slice_end_id"), ("newline", "newline_id"))},
        "rope": {"type": "mRoPE",
                 "mrope_section": cfg["text_config"]["rope_parameters"]["mrope_section"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", "-m", required=True)
    parser.add_argument("--output-dir", "-o", required=True)
    args = parser.parse_args()
    destination = Path(args.output_dir).resolve()
    if destination.exists():
        raise ValueError("Output directory already exists; choose a new destination to preserve existing files")
    with open(os.path.join(args.model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "minicpmv4_7":
        raise ValueError("checkpoint is not MiniCPM-V 4.7")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    print(f"[staging] {staging}; retained for diagnosis if conversion fails", flush=True)
    from minicpm_v47_language_export import export_language_components
    export_language_components(args.model_dir, staging, prefix=destination.name)
    export(args.model_dir, str(staging), prefix=destination.name, keep_fp32=False,
           do_bf16=False, do_int8=False, cross_check=True,
           vision_exporter=export_vision_settings)
    finalize_runtime_config(args.model_dir, staging)
    finalize_tokenizer(args.model_dir, staging)
    if destination.exists():
        raise FileExistsError("Destination appeared during conversion; staging was preserved")
    staging.rename(destination)
    print(f"[complete] {destination}", flush=True)


def finalize_tokenizer(model_dir, output_dir):
    from minicpm_v47_tokenizer import export_minicpm_v47_tokenizer
    metadata = export_minicpm_v47_tokenizer(model_dir, output_dir)
    destination = os.path.join(output_dir, "model.json")
    with open(destination, encoding="utf-8") as f:
        runtime = json.load(f)
    runtime["tokenizer"]["pretokenizer"] = metadata
    with open(destination, "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2, ensure_ascii=False)


def finalize_runtime_config(model_dir, output_dir):
    destination = os.path.join(output_dir, "model.json")
    with open(destination, encoding="utf-8") as f:
        runtime = json.load(f)
    # The official ByteLevel tokenizer has no BOS postprocessor. im_start is
    # already in the chat template; prepending its configured bos_token would
    # duplicate the opening marker and shift every image/decode coordinate.
    runtime["tokenizer"]["bos"] = ""
    # This first integration deliberately exposes chat/image questions only.
    # Advertising the base Qwen tool-call protocol would select an unsupported
    # template and cause normal CLI startup to fail.
    runtime.get("setting", {}).pop("functions", None)
    generation_path = os.path.join(model_dir, "generation_config.json")
    if os.path.isfile(generation_path):
        with open(generation_path, encoding="utf-8") as f:
            generation = json.load(f)
        eos = generation.get("eos_token_id", [])
        eos = [eos] if isinstance(eos, int) else eos
        if eos:
            runtime["tokenizer"]["eos_ids"] = eos
    with open(destination, "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
