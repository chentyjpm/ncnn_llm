#include "utils/minicpm_resize.h"
#include <algorithm>
#include <exception>
extern "C" int minicpm_resize_probe(const unsigned char* src,int h,int w,int th,int tw,int bgr,unsigned char* dst) {
    try {
        std::vector<uint8_t> input;
        if(h>0 && w>0) input.assign(src,src+static_cast<size_t>(h)*w*3);
        const auto output=ncnn_llm::minicpm::resize_bicubic_rgb(input,h,w,th,tw,bgr!=0);
        std::copy(output.begin(),output.end(),dst);
        return 0;
    } catch(const std::exception&) { return -1; }
}
