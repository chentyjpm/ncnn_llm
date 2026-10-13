#pragma once
#include <map>
#include <memory>
#include <net.h>
#include <nlohmann/json.hpp>
#include "utils/minicpm_vision.h"

namespace ncnn_llm {
// Experimental single-thumbnail MiniCPM-V-4.7 pipeline. CPU FP32 deliberately
// avoids silently mixing precision/layouts across the independently run graphs.
class MiniCPMVisionRuntime {
public:
    MiniCPMVisionRuntime(const std::string& directory, const nlohmann::json& config, int threads);
    static void validate_manifest(const nlohmann::json& manifest);
    static void validate_thumbnail(const minicpm::Thumbnail& thumbnail);
    ncnn::Mat encode(const ncnn::Mat& bgr, minicpm::Thumbnail& thumbnail) const;
    minicpm::MarkerIds markers;
    std::string mode;
private:
    ncnn::Mat run(const std::string& name, const ncnn::Mat& input) const;
    std::map<std::string,std::shared_ptr<ncnn::Net>> graphs;
    std::vector<float> position;
    int hidden, output_hidden, patch, position_side, scale, max_slices;
    bool slice_mode;
};
}
