// Copyright (c) 2026 ncnn_llm authors. Apache-2.0.
#pragma once
#include <cstdint>
#include <vector>
namespace ncnn_llm { namespace minicpm {
// Full-image RGB uint8 resize matching Pillow 12.3.0 BICUBIC, no reducing_gap.
// Antialiased on downsampling; tightly packed three-channel input/output.
// input_bgr converts BGR input to RGB output. No crop boxes, alpha or float input.
std::vector<uint8_t> resize_bicubic_rgb(const std::vector<uint8_t>& pixels,
    int height, int width, int target_height, int target_width, bool input_bgr=false);
} }
