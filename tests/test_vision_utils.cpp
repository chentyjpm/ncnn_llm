// Regression coverage for the shared legacy and pre-expanded image paths.
#include "utils/image_utils.h"
#include "utils/rope_embed.h"
#include "utils/vision_rope.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
static void injection(bool expanded, bool at_end) {
    const int pad = 151655; // Must be an integer fill, not byte-wise memset.
    std::vector<int> ids = expanded ? std::vector<int>{11, pad, pad} : std::vector<int>{11, pad};
    if (!at_end) ids.push_back(22);
    ncnn::Mat text(3, (int)ids.size()), image(3, 2);
    for (int y=0; y<text.h; ++y) for (int x=0; x<3; ++x) text.row(y)[x]=y*10+x;
    for (int y=0; y<2; ++y) for (int x=0; x<3; ++x) image.row(y)[x]=100+y*10+x;
    const int old_last = (text.h-1)*10;
    int offset=-1;
    inject_image_embeds(ids, text, offset, pad, image, expanded);
    check(offset==1 && text.h==(at_end ? 3:4), "injected shape or offset");
    check(ids== (at_end ? std::vector<int>{11,pad,pad} : std::vector<int>{11,pad,pad,22}), "image token ids");
    for (int x=0; x<3; ++x) {
        check(text.row(0)[x]==x, "prefix preserved");
        check(text.row(1)[x]==100+x && text.row(2)[x]==110+x, "image rows copied");
        if (!at_end) check(text.row(3)[x]==old_last+x, "suffix preserved");
    }
}
int main() {
    try {
        for (bool expanded : {false,true}) for (bool at_end : {false,true}) injection(expanded,at_end);
        ncnn::Mat text(2,3), image(2,2); text.fill(7.f); image.fill(42.f);
        std::vector<int> malformed{1,99,2}; int index=12;
        inject_image_embeds(malformed,text,index,99,image,true);
        check(index==-1 && text.row(1)[0]==7.f && malformed.size()==3, "invalid expanded span rejected atomically");
        std::vector<int> absent{1,2,3};
        inject_image_embeds(absent,text,index,99,image);
        check(index==-1 && text.row(1)[0]==7.f, "missing legacy placeholder unchanged");
        ncnn::Mat bgr(4,4,3,(size_t)1u);
        for (int i=0;i<48;++i) static_cast<unsigned char*>(bgr.data)[i]=(i*37)%256;
        auto bilinear=ncnn_mat_resize(bgr,4,4);
        auto bicubic=ncnn_mat_resize_bicubic(bgr,4,4);
        check(!bilinear.empty() && !bicubic.empty(), "legacy resize outputs");
        check(std::memcmp(bgr.data,bilinear.data,48)==0, "legacy bilinear identity");
        check(std::memcmp(bgr.data,bicubic.data,48)==0, "legacy bicubic identity");
        check(ncnn_mat_resize({},4,4).empty() && ncnn_mat_resize_bicubic({},4,4).empty(), "empty legacy resize");
        ncnn::Mat cc, ss;
        generate_vision_rope_cache_2d(4,4,2,10000.f,{2,2},true,cc,ss);
        check(cc.w==8 && cc.h==16 && ss.w==8 && ss.h==16, "legacy vision rope shape");
        int row=0;
        for(int gy=0;gy<2;++gy) for(int gx=0;gx<2;++gx)
            for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx,++row) {
                for(int j=0;j<4;++j) {
                    float angle=(j<2?gy*2+dy:gx*2+dx)*(j%2?0.01f:1.f);
                    check(std::abs(cc.row(row)[j]-std::cos(angle))<1e-6f &&
                          std::abs(ss.row(row)[j]-std::sin(angle))<1e-6f, "legacy rope ordering");
                    check(cc.row(row)[j]==cc.row(row)[j+4] && ss.row(row)[j]==ss.row(row)[j+4], "legacy rope duplication");
                }
            }
        std::cout << "PASS: legacy/expanded injection, legacy image resizing and vision RoPE\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
