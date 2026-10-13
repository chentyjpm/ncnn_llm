# MiniCPM-V 4.7 vision validation

## Verified result

The complete exported FP32 vision path was compared with the official
Transformers implementation using the real `openbmb/MiniCPM-V-4.7-1B` checkpoint
and a 448 × 448 photo. The final 64 × 1024 visual embedding matrix passed
absolute/relative checks (`atol=1e-3`, `rtol=1e-4`) and cosine > 0.999999.

| Stage | Shape | Maximum absolute error | Mean absolute error | Cosine similarity |
|---|---|---:|---:|---:|
| Patch + learned position | 1024 × 1152 | 7.39098e-6 | 1.48013e-7 | 0.999999999999907 |
| Encoder layers 0–6 | 1024 × 1152 | 6.91414e-5 | 1.50244e-6 | 0.999999999985954 |
| Inserted window merger | 256 × 1152 | 2.70844e-4 | 1.46963e-6 | 0.999999999928677 |
| Encoder layers 7–26 + final norm | 256 × 1152 | 3.41415e-4 | 2.59670e-6 | 0.999999999984749 |
| Final projector (Python ncnn) | 64 × 1024 | 8.05855e-5 | 1.53372e-6 | 0.999999999903977 |
| Complete native C++ runtime | 64 × 1024 | 6.93798e-5 | 1.46256e-6 | 0.999999999906202 |

The attention-head dimension is 72, with 16 heads. We independently checked
encoder layers 0–6 using the exact HF input embedding matrix: maximum error
7.00951e-5. This separates attention-graph validation from patch-convolution
rounding.

Both ncnn Python 1.0.20260526 and the actual C++
`MiniCPMVisionRuntime::encode` passed. The native runtime used pinned ncnn
`b0f97f43f266617ff5e5096d440dbe3f9a4c35b8`, the final converted model directory,
and the same saved photo. Load plus inference took 89.07 seconds with two CPU
threads and peak RSS 2,305,772 KiB (about 2.20 GiB). No language model was loaded.

This report covers one real thumbnail and CPU FP32 execution. It does not
establish language-generation quality, multislice/video support, Vulkan
results, or reduced-precision accuracy.

## Provenance and artifacts

`minicpm_v47_vision_validation.json` records the exact model revision, checkpoint
SHA-256, source image URL and fixture hash, framework versions, official model
source hashes, graph-parameter hashes, saved-array hashes, and measured errors.
Large model weights and arrays are deliberately not committed.

The public input is the MiniCPM demo photo:
https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png

The saved test fixture was created with Pillow `ImageOps.pad`, RGB 448 × 448,
bicubic resampling and RGB background 240. Its SHA-256 is
`969dbe26ad4dd2d94daad99b6b7c172ea7adf9f56305ee416d39794b9fbd299a`.
Both paths receive this same saved fixture; the validation does not conflate
image-resize differences with model errors.

HF's packed image is `[1, 3, 14, 14336]`, with target grid `[32, 32]`. Unpacking
it recovers the same raster patch elements as the CHW ncnn input. The default
HF processor's floating-point normalization differs by at most 5.91390e-8;
trained patch + position embeddings differ by at most 7.15256e-7, with cosine
0.9999999999999983. The complete packing proof is recorded in the JSON.

## Reproduce

Use a complete local checkpoint and the saved fixture. The validation never
downloads models and forces offline HF operation:

```sh
python tests/check_minicpm_v47_vision_parity.py \
  MODEL_DIR refract-448.png RESULTS_DIR CONVERTED_DIR \
  --prefix test_ --threads 2 --atol 1e-3 --rtol 1e-4 --min-cosine 0.999999
```

The reference constructs the full model on the meta device, materializes only
`vision_tower` and `merger`, and verifies that language parameters remain meta.
It releases the reference before streamed export and loads one ncnn graph at
a time. Avoid overlapping this test with a full language-model load on
memory-constrained machines.

Run the synthetic and parser regression suite:

```sh
python -m unittest discover -s tests -p test_minicpm_v47_vision_export.py -v
```

Actual ncnn Python numeric tests cover all six graph components and a dedicated
head-dimension-72 regression. Dedicated pinned native runs also passed all six
components, head dimension 72, and a complete tiny PNG-to-features runtime
regression exercising owned window inputs. A separate test verifies BF16
safetensors-to-FP32 conversion. These checks were run in separate lightweight
batches to keep peak memory bounded.

For the native component probe, build `minicpm_component_probe` linked to the
pinned ncnn library and rerun with:

```sh
NCNN_COMPONENT_PROBE=/path/to/minicpm_component_probe \
NCNN_VISION_PROBE=/path/to/minicpm_trained_vision_probe \
  python -m unittest discover -s tests -p test_minicpm_v47_vision_export.py -v
```

To exercise the actual native runtime without loading language weights:

```sh
minicpm_trained_vision_probe MODEL_DIR refract-448.png native_features.f32 2
```

The output is a row-major FP32 matrix; compare its 64 × 1024 values with
`hf_features.npy` saved by the reference driver using the thresholds above.
The probe prints shape, patch grid, thread count, and confirmation that it
never loads the language model. It is available as a CMake target.

## Correctness constraints discovered during validation

1. **ncnn text-float precision:** ncnn's text parameter parser accumulates the
   fractional numerator and denominator in unsigned 32-bit integers and scans
   at most 15 characters. A long decimal such as `0.117851130198` silently
   becomes approximately `0.52893259`. Export scale and epsilon using `.8e`,
   which preserves FP32 precision without overflow or truncation. The test
   suite reproduces the parser's behavior and checks the head-dimension-72
   attention scale. This is an exporter-format constraint, not an attention
   architecture difference.
2. **Python input ownership:** ncnn's Python `Mat` borrows NumPy storage.
   Retain a named contiguous NumPy owner until `Mat.clone()` has completed.
   Passing an ephemeral conversion directly into `Mat(...).clone()` corrupted
   the first patch in the initial harness. The corrected harness retains the
   owner; the C++ component probe owns its `Mat` input directly.
3. **C++ window ownership:** an external `Mat` row view has a null reference
   count even while its backing window buffer is alive. The pinned ncnn
   light-mode in-place dispatcher dereferenced that count during LayerNorm.
   The runtime now clones each four-token window into an owned input before
   extraction. A tiny full-runtime regression and the real native feature
   comparison both pass with this correction.

The final measurements above come from the corrected path. Failed diagnostic
arrays are not used as reference outputs.
