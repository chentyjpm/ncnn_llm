"""Synthetic graph/serialization tests. No model weights or network required.

The NumPy interpreter is a format-level check, not a substitute for ncnn/HF
real-checkpoint parity. Execute with python -m unittest discover -s tests -p
 'test_minicpm_v47_vision_export.py'.
"""
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("vision_export", Path(__file__).parents[1] / "export/minicpm_v47_vision_export.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config():
    return {"model_type": "minicpmv4_7", "insert_layer_id": 0,
            "text_config": {"hidden_size": 6},
            "vision_config": {"hidden_size": 8, "intermediate_size": 12,
                              "num_attention_heads": 2, "num_hidden_layers": 2,
                              "image_size": 8, "patch_size": 2, "num_channels": 3,
                              "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6}}


class Source:
    def __init__(self, graphs):
        rng = np.random.default_rng(413)
        self.values = {}
        self.calls = []
        for g in graphs.values():
            for key, shape, _ in g.weights:
                if key not in self.values:
                    self.values[key] = (rng.standard_normal(shape) * .1).astype(np.float32)
                    if "norm" in key and key.endswith("weight"):
                        self.values[key] += 1
        self.values[module.VISION + ".embeddings.position_embedding.weight"] = rng.standard_normal((16, 8)).astype(np.float32)

    def get(self, key):
        self.calls.append(key)
        return self.values[key]


def norm(x, weight, bias, eps):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps) * weight + bias


def gelu(x, tanh):
    if tanh:
        return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x**3)))
    return .5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))


def interpret(param, binary, value):
    """Evaluate exported bytes using ncnn's dimension and weight-tag conventions."""
    data = memoryview(binary)
    offset = 0
    blobs = {"in0": value}

    def read(shape, tagged=False):
        nonlocal offset
        if tagged:
            assert bytes(data[offset:offset + 4]) == b"\0\0\0\0"
            offset += 4
        size = math.prod(shape)
        result = np.frombuffer(data[offset:offset + size * 4], dtype="<f4").reshape(shape)
        offset += size * 4
        return result

    lines = param.splitlines()
    assert int(lines[1].split()[0]) == len(lines) - 2
    for line in lines[2:]:
        fields = line.split()
        kind, _, ni, no = fields[:4]
        ni, no = int(ni), int(no)
        inputs = fields[4:4 + ni]
        outputs = fields[4 + ni:4 + ni + no]
        pd = dict(p.split("=", 1) for p in fields[4 + ni + no:])
        if kind == "Input":
            continue
        x = blobs[inputs[0]]
        if kind == "Split":
            for out in outputs:
                blobs[out] = x
            continue
        if kind == "LayerNorm":
            n = int(pd["0"])
            y = norm(x, read((n,)), read((n,)), float(pd["1"]))
        elif kind == "Gemm":
            n, k = int(pd["8"]), int(pd["9"])
            y = x @ read((n, k), True).T + read((n,), True)
        elif kind == "Reshape":
            dims = [int(pd[k]) for k in ("2", "1", "0") if k in pd]
            y = x.reshape(dims)
        elif kind == "Permute":
            order = int(pd["0"])
            y = x.T if x.ndim == 2 and order == 1 else x.transpose(1, 0, 2)
        elif kind == "SDPA":
            q, k, v = (blobs[name] for name in inputs)
            scores = q @ k.transpose(0, 2, 1) * float(pd["6"])
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            y = (scores / scores.sum(-1, keepdims=True)) @ v
        elif kind == "BinaryOp":
            y = x + blobs[inputs[1]]
        elif kind == "GELU":
            y = gelu(x, bool(int(pd.get("0", "0"))))
        elif kind == "Noop":
            y = x
        elif kind == "Convolution":
            n, p = int(pd["0"]), int(pd["1"])
            w = read((n, x.shape[0], p, p), True)
            b = read((n,))
            patches = np.lib.stride_tricks.sliding_window_view(x, (p, p), axis=(1, 2))[:, ::p, ::p]
            y = np.einsum("chwkl,ockl->ohw", patches, w) + b[:, None, None]
        else:
            raise AssertionError(kind)
        blobs[outputs[0]] = y
    assert offset == len(binary), (offset, len(binary))
    return blobs["out0"]


def attention_reference(x, state, base, heads=2):
    ln = norm(x, state[base + ".layer_norm1.weight"], state[base + ".layer_norm1.bias"], 1e-6)
    qkv = []
    for c in "qkv":
        key = base + ".self_attn." + c + "_proj"
        qkv.append((ln @ state[key + ".weight"].T + state[key + ".bias"]).reshape(len(x), heads, -1))
    q, k, v = qkv
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(q.shape[-1])
    prob = np.exp(scores - scores.max(-1, keepdims=True))
    prob /= prob.sum(-1, keepdims=True)
    y = np.einsum("hts,shd->thd", prob, v).reshape(x.shape)
    base += ".self_attn.out_proj"
    return x + y @ state[base + ".weight"].T + state[base + ".bias"]


def block_reference(x, state, base):
    x = attention_reference(x, state, base)
    y = norm(x, state[base + ".layer_norm2.weight"], state[base + ".layer_norm2.bias"], 1e-6)
    y = gelu(y @ state[base + ".mlp.fc1.weight"].T + state[base + ".mlp.fc1.bias"], True)
    return x + y @ state[base + ".mlp.fc2.weight"].T + state[base + ".mlp.fc2.bias"]


class VisionExportTests(unittest.TestCase):
    def setUp(self):
        self.graphs = module.build_graphs(config())
        self.source = Source(self.graphs)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def run_graph(self, name, x):
        files = self.graphs[name].write(self.source, self.path, name)
        return interpret((self.path / files["param"]).read_text(), (self.path / files["bin"]).read_bytes(), x)

    def test_encoder_components_numerical(self):
        x = np.random.default_rng(1).standard_normal((5, 8)).astype(np.float32)
        expected = block_reference(x, self.source.values, module.VISION + ".encoder.layers.0")
        np.testing.assert_allclose(self.run_graph("encoder_pre", x), expected, atol=2e-6, rtol=2e-6)
        y = block_reference(x, self.source.values, module.VISION + ".encoder.layers.1")
        key = module.VISION + ".post_layernorm"
        expected = norm(y, self.source.values[key + ".weight"], self.source.values[key + ".bias"], 1e-6)
        np.testing.assert_allclose(self.run_graph("encoder_post", x), expected, atol=2e-6, rtol=2e-6)

    def test_window_attention_numerical(self):
        x = np.random.default_rng(3).standard_normal((4, 8)).astype(np.float32)
        expected = attention_reference(x, self.source.values, module.VISION + ".vit_merger")
        np.testing.assert_allclose(self.run_graph("window_attn", x), expected, atol=2e-6, rtol=2e-6)

    def test_projectors_numerical_and_activation(self):
        x = np.random.default_rng(5).standard_normal((3, 32)).astype(np.float32)
        for name, base, tanh in (("window_mlp", module.VISION + ".vit_merger", True),
                                 ("projector", "model.merger.mlp.0", False)):
            s = self.source.values
            y = norm(x, s[base + ".pre_norm.weight"], s[base + ".pre_norm.bias"], 1e-6)
            y = gelu(y @ s[base + ".linear_1.weight"].T + s[base + ".linear_1.bias"], tanh)
            expected = y @ s[base + ".linear_2.weight"].T + s[base + ".linear_2.bias"]
            np.testing.assert_allclose(self.run_graph(name, x), expected, atol=2e-6, rtol=2e-6)

    def test_patch_convolution_row_order(self):
        x = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8) / 100
        s = self.source.values
        key = module.VISION + ".embeddings.patch_embedding"
        rows = []
        for y in range(0, 8, 2):
            for z in range(0, 8, 2):
                rows.append((s[key + ".weight"] * x[None, :, y:y + 2, z:z + 2]).sum((1, 2, 3)) + s[key + ".bias"])
        np.testing.assert_allclose(self.run_graph("patch_embed", x), rows, atol=2e-6)

    def test_windows_and_positions(self):
        np.testing.assert_array_equal(module.window_indices(4, 4), [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
        np.testing.assert_array_equal(module.nearest_position_ids(2, 2, 4), [0, 2, 8, 10])
        with self.assertRaises(ValueError):
            module.window_indices(3, 4)

    def test_manifest_and_weight_shape_guard(self):
        (self.path / "config.json").write_text(json.dumps(config()))
        out = self.path / "out"
        manifest = module.export_minicpm_v47_vision(self.path, out, source=self.source)
        self.assertEqual(set(manifest["components"]), set(self.graphs))
        self.assertEqual((out / manifest["position_embeddings"]["file"]).stat().st_size, 16 * 8 * 4)
        self.assertEqual(manifest["output_hidden_size"], 6)
        first = self.graphs["window_mlp"].weights[0][0]
        self.source.values[first] = np.zeros(1)
        with self.assertRaisesRegex(ValueError, "expected"):
            self.graphs["window_mlp"].write(self.source, self.path, "broken")
        self.assertFalse((self.path / "broken.ncnn.bin").exists())
        self.assertFalse((self.path / "broken.ncnn.bin.tmp").exists())

    def test_native_ncnn_component_parity_when_probe_available(self):
        probe = os.environ.get("NCNN_COMPONENT_PROBE")
        if not probe:
            self.skipTest("Set NCNN_COMPONENT_PROBE to the compiled C++ probe")
        rng = np.random.default_rng(19)
        shapes = {"patch_embed": (3, 8, 8), "encoder_pre": (16, 8),
                  "encoder_post": (4, 8), "window_attn": (4, 8),
                  "window_mlp": (4, 32), "projector": (1, 32)}
        for name, shape in shapes.items():
            with self.subTest(component=name):
                x = rng.standard_normal(shape).astype(np.float32)
                expected = self.run_graph(name, x)
                input_path, output_path = self.path / "input.f32", self.path / "output.f32"
                x.tofile(input_path)
                result = subprocess.run([probe, str(self.path / (name + ".ncnn.param")),
                    str(self.path / (name + ".ncnn.bin")), str(input_path), str(output_path),
                    str(len(shape)), str(shape[-1]), str(shape[-2]),
                    str(shape[0] if len(shape) == 3 else 1)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual_shape = tuple(map(int, result.stdout.strip().split()))
                self.assertEqual(actual_shape, expected.shape)
                actual = np.fromfile(output_path, dtype=np.float32).reshape(actual_shape)
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_ncnn_component_parity_when_bindings_available(self):
        try:
            import ncnn
        except ImportError:
            self.skipTest("ncnn Python binding is not installed")
        if not hasattr(ncnn, "Net"):
            self.skipTest("ncnn source namespace is present, but Python binding is not installed")
        rng = np.random.default_rng(9)
        shapes = {"patch_embed": (3, 8, 8), "encoder_pre": (16, 8),
                  "encoder_post": (4, 8), "window_attn": (4, 8),
                  "window_mlp": (4, 32), "projector": (1, 32)}
        for name, shape in shapes.items():
            with self.subTest(component=name):
                x = rng.standard_normal(shape).astype(np.float32)
                expected = self.run_graph(name, x)
                net = ncnn.Net()
                net.opt.use_vulkan_compute = False
                net.opt.use_fp16_storage = False
                net.opt.use_fp16_arithmetic = False
                net.opt.use_bf16_storage = False
                net.opt.use_packing_layout = False
                net.opt.num_threads = 1
                self.assertEqual(net.load_param(str(self.path / (name + ".ncnn.param"))), 0)
                self.assertEqual(net.load_model(str(self.path / (name + ".ncnn.bin"))), 0)
                ex = net.create_extractor()
                owned_input = x.copy()
                input_mat = ncnn.Mat(owned_input).clone()
                self.assertEqual(ex.input("in0", input_mat), 0)
                status, out = ex.extract("out0")
                self.assertEqual(status, 0)
                np.testing.assert_allclose(np.array(out), expected, atol=2e-5, rtol=2e-5)

    def test_bfloat16_lazy_source_when_torch_available(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch/safetensors are not installed")
        tensor = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
        save_file({"sample": tensor}, str(self.path / "model.safetensors"))
        source = module.SafeTensorSource(self.path)
        self.assertEqual(source.get("sample").dtype, np.float32)
        np.testing.assert_array_equal(source.get("sample"), tensor.float().numpy())
        with self.assertRaisesRegex(KeyError, "Missing"):
            source.get("not_present")

    def test_full_architecture_and_streaming_descriptors(self):
        cfg = config()
        cfg["insert_layer_id"] = 6
        cfg["vision_config"]["num_hidden_layers"] = 27
        graphs = module.build_graphs(cfg)
        self.assertEqual(graphs["encoder_pre"].param().count("SDPA "), 7)
        self.assertEqual(graphs["encoder_post"].param().count("SDPA "), 20)
        self.assertNotIn("RotaryEmbed", graphs["encoder_pre"].param())
        for g in graphs.values():
            for key, shape, tagged in g.weights:
                self.assertIsInstance(key, str)
                self.assertIsInstance(shape, tuple)
                self.assertIsInstance(tagged, bool)
        cfg["merger_times"] = 2
        with self.assertRaises(ValueError):
            module.build_graphs(cfg)

    def test_ncnn_float_parser_precision(self):
        # Mirror ncnn ParamDict's unsigned 32-bit fraction accumulators.
        def ncnn_float(value):
            mantissa, _, exponent = value.lower().partition("e")
            integer, _, fraction = mantissa.partition(".")
            numerator, denominator = 0, 1
            for digit in fraction:
                numerator = (numerator * 10 + int(digit)) & 0xffffffff
                denominator = (denominator * 10) & 0xffffffff
            return np.float32((int(integer) + numerator / denominator)
                              * 10 ** int(exponent or "0"))

        self.assertGreater(abs(float(ncnn_float("0.117851130198"))
                               - 1 / math.sqrt(72)), .4)
        cfg = config()
        cfg["vision_config"].update(hidden_size=144, intermediate_size=192)
        for graph in module.build_graphs(cfg).values():
            for line in graph.layers:
                for field in line.split():
                    if "=" not in field:
                        continue
                    value = field.split("=", 1)[1]
                    if "." not in value and "e" not in value.lower():
                        continue
                    self.assertLessEqual(len(value), 15)
                    self.assertLessEqual(len(value.split("e")[0].split(".")[-1]), 8)
                    self.assertEqual(ncnn_float(value), np.float32(float(value)))
                if line.startswith("SDPA "):
                    value = line.split("6=", 1)[1]
                    self.assertEqual(ncnn_float(value), np.float32(1 / math.sqrt(72)))

    def test_ncnn_head_dim_72_regression(self):
        try:
            import ncnn
        except ImportError:
            self.skipTest("ncnn Python binding is not installed")
        if not hasattr(ncnn, "Net"):
            self.skipTest("ncnn Python binding is not installed")
        cfg = config()
        cfg["vision_config"].update(hidden_size=144, intermediate_size=192)
        self.graphs = module.build_graphs(cfg)
        self.source = Source(self.graphs)
        x = np.random.default_rng(72).standard_normal((7, 144)).astype(np.float32)
        expected = self.run_graph("encoder_pre", x)
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        net.opt.use_bf16_storage = False
        net.opt.use_packing_layout = False
        net.opt.num_threads = 1
        self.assertEqual(net.load_param(str(self.path / "encoder_pre.ncnn.param")), 0)
        self.assertEqual(net.load_model(str(self.path / "encoder_pre.ncnn.bin")), 0)
        ex = net.create_extractor()
        owned_input = x.copy()
        input_mat = ncnn.Mat(owned_input).clone()
        self.assertEqual(ex.input("in0", input_mat), 0)
        status, output = ex.extract("out0")
        self.assertEqual(status, 0)
        np.testing.assert_allclose(np.array(output), expected, atol=2e-5, rtol=2e-5)

    def test_native_ncnn_head_dim_72_regression(self):
        probe = os.environ.get("NCNN_COMPONENT_PROBE")
        if not probe:
            self.skipTest("Set NCNN_COMPONENT_PROBE to the compiled C++ probe")
        cfg = config()
        cfg["vision_config"].update(hidden_size=144, intermediate_size=192)
        self.graphs = module.build_graphs(cfg)
        self.source = Source(self.graphs)
        x = np.random.default_rng(72).standard_normal((7, 144)).astype(np.float32)
        expected = self.run_graph("encoder_pre", x)
        input_path, output_path = self.path / "input.f32", self.path / "output.f32"
        x.tofile(input_path)
        result = subprocess.run([probe, str(self.path / "encoder_pre.ncnn.param"),
            str(self.path / "encoder_pre.ncnn.bin"), str(input_path), str(output_path),
            "2", "144", "7", "1"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(tuple(map(int, result.stdout.strip().split())), expected.shape)
        actual = np.fromfile(output_path, dtype=np.float32).reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_native_full_runtime_owned_window_regression(self):
        probe = os.environ.get("NCNN_VISION_PROBE")
        if not probe:
            self.skipTest("Set NCNN_VISION_PROBE to the vision-only C++ probe")
        from PIL import Image
        cfg = config()
        cfg["vision_config"].update(patch_size=14, image_size=56)
        self.graphs = module.build_graphs(cfg)
        self.source = Source(self.graphs)
        (self.path / "config.json").write_text(json.dumps(cfg))
        model_dir = self.path / "model"
        module.export_minicpm_v47_vision(self.path, model_dir, source=self.source)
        vision = {"manifest": "minicpm_v47_vision.json", "scale_resolution": 56,
                  "slice_mode": False, "max_slice_nums": 1,
                  "marker_ids": dict(zip(("image", "image_start", "image_end",
                                           "slice_start", "slice_end", "newline"), range(1, 7)))}
        (model_dir / "model.json").write_text(json.dumps({"setting": {"vision": vision}}))
        rgb = (np.arange(56 * 56 * 3, dtype=np.uint16) % 256).astype(np.uint8).reshape(56, 56, 3)
        image_path, output_path = self.path / "fixture.png", self.path / "features.f32"
        Image.fromarray(rgb).save(image_path)
        pixels = (rgb.transpose(2, 0, 1).astype(np.float32) / 255 - .5) / .5
        x = self.run_graph("patch_embed", pixels)
        x += self.source.values[module.VISION + ".embeddings.position_embedding.weight"][
            module.nearest_position_ids(4, 4, 4)]
        x = self.run_graph("encoder_pre", x)
        windows = x[module.window_indices(4, 4)].reshape(-1, 4, 8)
        attended = np.stack([self.run_graph("window_attn", window) for window in windows])
        x = self.run_graph("window_mlp", attended.reshape(-1, 32)) + attended.mean(axis=1)
        x = self.run_graph("encoder_post", x)
        expected = self.run_graph("projector", x[module.window_indices(2, 2)].reshape(-1, 32))
        result = subprocess.run([probe, str(model_dir), str(image_path), str(output_path), "1"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        shape = json.loads(result.stdout)
        self.assertEqual((shape["rows"], shape["cols"]), expected.shape)
        self.assertFalse(shape["language_model_loaded"])
        actual = np.fromfile(output_path, dtype=np.float32).reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
