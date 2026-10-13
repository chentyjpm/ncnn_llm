// Copyright (c) 2026 ncnn_llm authors. Apache-2.0.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ncnn_llm {
namespace minicpm {
struct Thumbnail {
    int height, width, patch_rows, patch_cols, token_rows, token_cols;
    int token_count() const { return token_rows * token_cols; }
};
// Matches HF MiniCPMV4_6ImageProcessor (also used for 4.7). Resize pixels
// separately with the checkpoint's bicubic backend; this function plans dimensions.
// Rejects cases where enabled slicing would produce detail crops.
Thumbnail plan_thumbnail(int height, int width, int patch_size, int scale_resolution,
                         const std::string& mode, bool slice_mode = true, int max_slices = 9);
int merge_factor(const std::string& mode);
// Input: already resized, interleaved RGB uint8. Output: contiguous [3,p,H*W/p]
// NaViT layout with rescale=1/255, mean=std=0.5. No implicit resizing.
std::vector<float> pack_rgb(const std::vector<uint8_t>& rgb, int height, int width, int patch_size);
std::vector<int> nearest_position_ids(int patch_rows, int patch_cols, int learned_side);
struct MarkerIds {
    int image, image_start, image_end, slice_start, slice_end, newline;
};
struct CanvasPositions {
    // One {T,H,W} tuple per sequence token; not the separate causal text channel.
    std::vector<std::array<int64_t, 3> > positions;
    int64_t rope_delta;
    int64_t next_position;
};
// Exactly one image, explicit start/end markers, no padding or video/slices.
// base_position is a scalar offset for an already-computed chat prefix.
CanvasPositions canvas_positions(const std::vector<int>& ids, const MarkerIds& markers,
                                  int patch_rows, int patch_cols, const std::string& mode,
                                  int64_t base_position = 0);
// Row-major [sequence,rotary_dim/2] cache for ncnn RotaryEmbed. Sections are
// checkpoint mrope_section values; do not hardcode Qwen defaults in the caller.
void interleaved_rope(const CanvasPositions& positions, int rotary_dim, double theta,
                      const std::array<int,3>& sections,
                      std::vector<float>& cos_cache, std::vector<float>& sin_cache);
} // namespace minicpm
} // namespace ncnn_llm
