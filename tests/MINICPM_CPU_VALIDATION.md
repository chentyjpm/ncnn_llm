# Experimental MiniCPM-V 4.7 CPU validation

This port targets original `openbmb/MiniCPM-V-4.7-1B` weights, not GGUF. Its scope is one RGB image, 16x downsampling, CPU FP32 and thumbnail processing. Video, multiple images, detail slices, GPU and reduced-precision accuracy are not established. Component tests and compilation do not establish full image-question inference.

The original model is Apache-2.0: <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B>, tested revision `ef3ab3bf0120a60e1a2d2b52d984a79d8cb7c630`. The 3,109,054,480-byte safetensors file has SHA256 `8c4921ac83182c85190ad10c3aeb118d51e385d2d98ec13b88b514bbf6cac002`. Obtain weights separately. The reference environment uses Python 3.12, CPU PyTorch 2.14.1, Transformers 5.19.0, Pillow 12.3.0, safetensors 0.8.0 and pnnx 20260526.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
python export/minicpm_v47_export.py -m "$HF_MODEL" -o "$NCNN_MODEL"
```

Embedding/output tables stream in small row chunks. Only the decoder is loaded for tracing and is released before pnnx runs; vision conversion streams individual tensors. Allow roughly 6.2 GB for final FP32 weights plus checkpoint/temporary files. Avoid concurrent large reference/conversion jobs.

The following commands compile this repository's original LM head. Set the path variables to existing checkouts and build directories. Use matching ncnn sources, generated headers and library; the tested revision is `b0f97f43f266617ff5e5096d440dbe3f9a4c35b8`. No download is implicit. Do not reconfigure a cache while another build uses it.

```sh
CMAKE=cmake
SRC="$PWD"
# Set NCNN_SRC, NCNN_BUILD, JSON_SRC, SDK and BUILD before running.

"$CMAKE" -S "$NCNN_SRC" -B "$NCNN_BUILD" -DNCNN_VULKAN=OFF -DNCNN_INSTALL_SDK=ON
"$CMAKE" --build "$NCNN_BUILD" -j2
"$CMAKE" --install "$NCNN_BUILD" --prefix "$SDK"

NLOHMANN_JSON_DIR="$JSON_SRC" \
"$CMAKE" -S "$SRC" -B "$BUILD" \
  -DUSE_SYSTEM_NCNN=ON -DNCNN_LLM_ENABLE_VULKAN=OFF \
  -Dncnn_DIR="$SDK/lib/cmake/ncnn" \
  -DCMAKE_CXX_FLAGS="-I$NCNN_SRC/src -I$NCNN_SRC/src/layer -I$NCNN_SRC/src/layer/x86 -I$NCNN_BUILD/src"
"$CMAKE" --build "$BUILD" --target test_minicpm_runtime -j2
"$BUILD/test_minicpm_runtime" --check-limits
"$BUILD/test_minicpm_runtime" --tokenize "$NCNN_MODEL" "$REFERENCE_INPUT_JSON"
```

For the real checkpoint, invoke the same built executable with the exact PNG and question used in the saved reference:

```sh
"$BUILD/test_minicpm_runtime" "$NCNN_MODEL" "$REFERENCE_PNG" \
  'Describe this image in one short sentence.' 32 "$VISION_F32_DUMP"
```

Record actual command outputs, token IDs, projected-vision differences, prefill token, decoded answer and relevant reference settings. These are instructions, not a claim that the commands have already passed. The PIL reference must explicitly use the official `MiniCPMV4_6ImageProcessorPil` backend; default AutoImageProcessor/Torchvision resizing has different semantics.

Run `python tests/run_minicpm_language_smoke.py` and `python tests/test_minicpm_v47_vision_export.py` for synthetic contracts. `tests/check_minicpm_v47_vision_parity.py` and `tests/minicpm_v47_text_reference.py` provide real-checkpoint reference drivers; consult `--help`. The streamed text reference uses official forwarding/caches and the public `mm_encoder_outputs` interface, rather than another inference engine.

Start with an exactly 448x448 PNG. The default converted configuration rejects images requiring detail slices. Explicitly setting `setting.vision.max_slice_nums` to `1` opts into thumbnail-only processing of larger images, omitting detail crops. Linux checks do not establish Windows/macOS/GPU support.

ncnn's scalar parameter parser uses uint32 decimal accumulators and a 15-character input limit. Emitted floats therefore use `.8e`, covered by a real 72-dimensional attention regression. Python `ncnn.Mat` may borrow NumPy storage; test inputs must retain an owning array until copying completes.

## Checkpoint prompt template (no weights)

Build `minicpm_prompt_probe` or compile its two source files directly, then run:

```sh
python tests/test_minicpm_prompt.py "$HF_MODEL" "$BUILD/minicpm_prompt_probe"
```

This compares C++ rendered strings to the checkpoint's actual Jinja template,
then checks their IDs with the checkpoint tokenizer. It does not establish
parity of the separate native C++ BPE implementation. Cases cover single and
multiple turns, past/current reasoning, thinking on/off, generation on/off,
image markers, repeated whitespace, Chinese and digits. The experimental
MiniCPM template rejects tool messages/calls rather than approximate them.

MiniCPM model packages must contain `tokenizer.pretokenizer` with type
`minicpm_v47` and the relative tokenizer artifact filename. Missing metadata
fails closed before weight loading. The tokenizer artifact is bounded to 1 MiB
and must be a regular file within the package without symlink components.
`test_minicpm_runtime --tokenize` uses this same opt-in tokenizer pipeline.

## Exact native tokenizer validation

The converted package carries an opt-in Unicode/pretokenizer artifact generated
from the checkpoint's actual tokenizers backend. It validates the complete
supported NFC/Split/ByteLevel/BPE pipeline before export. MiniCPM runtime rejects
missing or unsupported metadata; legacy model tokenizers keep their prior path.
The artifact's source digest records provenance, not runtime cryptographic
integrity verification.

The actual checkpoint test passed 6,549 exact native/HF token-ID and decode cases,
including Chinese, whitespace, combining accents, Hangul, punctuation, code and
special tokens. Five malformed UTF-8 inputs, 11 malformed artifacts and four
unsupported pipeline configurations reject. See
`minicpm_v47_tokenizer_validation.json`. These tokenizer results do not establish
language-generation accuracy.

```sh
python tests/test_minicpm_tokenizer_reference.py --help
```

The converter requires a new output directory. All components and required
metadata are finalized in a sibling staging directory; only a completed package
is renamed to the requested destination. Failed staging is retained for diagnosis.
No existing graph is reused solely because its filename or size matches.

## Synthetic cached decoder validation

The tiny 64-wide four-layer fixture must be converted with the current exporter:
full-head rotary embeddings must omit the mathematically empty tail crop/concat.
Run `prepare_minicpm_decoder_probe.py FIXTURE_DIR CASE_DIR PROBE` to compare
prefill eight tokens, decode one token, then decode three tokens to independent
HF full-prefix references. The probe checks hidden/ShortConv/GDR values and KV
lengths. Optimized x86 KV caches are opaque panel-packed storage; only their
logical dimensions are reported, never a misleading raw tensor dump.

`MINICPM_PROBE_BLOBS=blob1,blob2` enables optional intermediate shape diagnostics.
A case with an empty `initial_states` object also tests runtime-style omitted
initial cache inputs rather than explicitly supplied zero/empty states.

## Real image-question result

The converted trained model completed native CPU image-question generation for
the public pencil-in-water fixture. Its answer exactly matched the independent
official Transformers reference: “A red pencil with a yellow tip rests in a glass
of water, partially submerged.” The first token was 32 in both implementations;
the native next canvas position was 40, matching 98 prompt tokens plus the
reference rope delta of -58. The test used greedy decoding, repetition penalty 1,
FP32 and one 448×448 PNG. It completed with exit 0 in 141.15 seconds including loading,
with peak RSS 6,441,280 KiB. See `minicpm_v47_vqa_validation.json` for the precise
scope and build provenance; full native generated-token IDs were not recorded.

The native visual features separately match official HF with maximum absolute
error 6.94e-5 and cosine 0.999999999906. The trained decoder's eight-token hidden
states match within the predeclared 5e-4 absolute/relative tolerance. Synthetic
cached prefill/decode lengths 8/9/12 match within 4.77e-7. A full-width-RoPE exporter
edge case was fixed by omitting the mathematically empty tail, without changing
the trained MiniCPM partial-RoPE path.

## Shared vision integration regression

MiniCPM image prefill now goes through the existing
`ncnn_llm_gpt::get_visiual_features` entry point. Model loading shares `load_net`
with the existing text/vision graphs, and both image prompt formats share
`inject_image_embeds`. The original five-argument injection interface remains;
the new overload accepts a pre-expanded visual-token span. Integer placeholder
IDs use `std::fill_n`, correcting the previous byte-wise `memset` behavior.

The model-specific window-attention/projector stages are functions in
`ncnn_llm_gpt.cpp`; their immutable, factory-created state is not an independent
runtime. The manifest, converted graph files, CPU FP32 requirements and
single-thumbnail restrictions are unchanged. No model reconversion is required.

Preprocessing helpers now live in `utils/image_utils.{h,cpp}`, alongside the
existing image decoder and resize functions. MiniCPM needs a separate function
there for antialiased Pillow fixed-point bicubic resizing: neither the existing
bilinear resize nor ncnn's float bicubic has the same rounding/filter semantics.
The original resize functions and the existing ViT/Qwen preprocessing remain
unchanged. Likewise, MiniCPM canvas positions and interleaved MRoPE live in
`utils/rope_embed.{h,cpp}`; substituting the old Qwen image-position formula would
change the canvas markers and cached decode offset. Existing Qwen/ViT RoPE
functions retain their previous formulas.

The independent `minicpm_vision_runtime`, `minicpm_vision`, and `minicpm_resize`
source/header pairs are removed. Reference probes now link the same utility
implementation built by CMake, rather than compiling the removed standalone
sources. No model tensors are needed for the following checks:

```sh
"$CMAKE" --build "$BUILD" --target test_vision_utils test_minicpm_vision \
  minicpm_resize_probe minicpm_preprocess_probe minicpm_vision_reference_probe -j2
ctest --test-dir "$BUILD" --output-on-failure
export NCNN_LLM_BUILD_DIR="$BUILD"
python tests/test_minicpm_resize_reference.py
python tests/test_minicpm_preprocess_reference.py --model-dir "$HF_MODEL"
python tests/test_minicpm_vision_reference.py --model-dir "$HF_MODEL"
```

`test_vision_utils` covers both legacy single-placeholder expansion and the
already-expanded span (including at the end of the prompt), prefix/suffix
preservation, invalid-span rejection, legacy resize behavior and legacy ViT
2-D RoPE ordering. MiniCPM helper checks remain enabled in Release builds.
Existing legacy model integration cases can skip when their optional model
packages are absent; passing CTest is not a claim of full Qwen/ViT checkpoint
inference or Windows/macOS validation.

### Refactor acceptance (2026-10-14)

`minicpm_v47_refactor_validation.json` records the tested source fingerprints and
results using the existing converted checkpoint, with no reconversion:

- All 7 CTest targets and all 14 vision-export tests passed (native component and
  full-vision probes enabled; no exporter-test skips).
- 1,290 Pillow resize comparisons are byte-exact; all 20 complete official PIL
  preprocessing cases have zero pixel error. All 252 official HF canvas,
  nearest-position and rotary-cache cases pass (maximum rotary error 1.573e-5).
- The trained 64x1024 projected image features are bit-for-bit equal to the
  pre-refactor native output. Maximum absolute error against official HF remains
  6.938e-5; cosine similarity is 0.999999999906.
- Trained decoder maximum absolute error against the saved independent HF
  eight-token target is 4.215e-4 (within the original 5e-4 tolerance).
- Pencil VQA exactly matches official HF and the pre-refactor answer:
  “A red pencil with a yellow tip rests in a glass of water, partially submerged.”
- Bee VQA exactly matches the pre-refactor answer:
  “A bumblebee rests on a pink cosmos flower, surrounded by wilting blooms and
  red flowers in a garden.” This second case has no independent HF reference.
- Both image-question runs completed successfully; peak child RSS was
  6,442,196 KiB. Optional legacy model integration tests skipped three missing
  model cases; the new shared-utility legacy-path regressions did run.

Enable the native exporter regressions with the same build:

```sh
NCNN_COMPONENT_PROBE="$BUILD/minicpm_component_probe" \
NCNN_VISION_PROBE="$BUILD/minicpm_trained_vision_probe" \
python -m unittest discover -s tests -p test_minicpm_v47_vision_export.py -v
```
