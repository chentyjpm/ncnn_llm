#include "utils/image_utils.h"
#include "utils/rope_embed.h"
#include <algorithm>
#include <exception>
extern "C" int minicpm_preprocess_probe(const unsigned char* src,int h,int w,int patch,int scale,float* dst,int capacity,int* shape) {
    try {
        const auto t=ncnn_llm::minicpm::plan_thumbnail(h,w,patch,scale,"16x",true,1);
        shape[0]=t.patch_rows; shape[1]=t.patch_cols;
        auto rgb=ncnn_llm::minicpm::resize_bicubic_rgb(std::vector<uint8_t>(src,src+static_cast<size_t>(h)*w*3),h,w,t.height,t.width);
        auto packed=ncnn_llm::minicpm::pack_rgb(rgb,t.height,t.width,patch);
        if(static_cast<int>(packed.size())>capacity) return -2;
        std::copy(packed.begin(),packed.end(),dst);
        return static_cast<int>(packed.size());
    } catch(const std::exception&) { return -1; }
}
