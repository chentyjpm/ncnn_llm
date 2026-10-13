// Copyright (c) 2026 ncnn_llm authors. Apache-2.0.
// References: transformers/models/minicpmv4_6/image_processing_minicpmv4_6.py;
// transformers/models/minicpmv4_7/modeling_minicpmv4_7.py (canvas M-RoPE).
#include "minicpm_vision.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ncnn_llm { namespace minicpm {
namespace {
void require(bool ok, const char* reason) { if (!ok) throw std::invalid_argument(reason); }
int divide_even(int n, int divisor) {
    // Python round, independent of C++ floating-point rounding mode.
    int q = n / divisor, r = n % divisor;
    if (2LL*r > divisor || (2LL*r == divisor && q % 2)) ++q;
    require(static_cast<int64_t>(q)*divisor <= std::numeric_limits<int>::max(), "resize dimension overflow");
    return std::max(q * divisor, divisor);
}
}
int merge_factor(const std::string& mode) {
    require(mode == "16x" || mode == "4x", "MiniCPM downsample mode must be 16x or 4x");
    return mode == "16x" ? 4 : 2;
}
Thumbnail plan_thumbnail(int h, int w, int p, int scale, const std::string& mode,
                         bool slice_mode, int max_slices) {
    const int merge = merge_factor(mode);
    require(h > 0 && w > 0 && p > 0 && p <= std::numeric_limits<int>::max()/4 && scale > 0,
            "invalid thumbnail dimensions");
    require(max_slices >= 1, "max_slices must be positive");
    const double ratio = static_cast<double>(h)*w / (static_cast<double>(scale)*scale);
    require(!slice_mode || max_slices == 1 || ratio <= 1.0,
            "MiniCPM detail slices are unsupported; explicitly disable slicing or use max_slices=1");
    // No slices => allow_upscale=True, including small images. HF truncates
    // height before calculating width, then rounds both to patch_size*4.
    const double aspect = static_cast<double>(w)/h;
    const double resized_h = scale/std::sqrt(aspect);
    require(resized_h < std::numeric_limits<int>::max(), "resize height overflow");
    h = static_cast<int>(resized_h);
    require(h*aspect < std::numeric_limits<int>::max(), "resize width overflow");
    w = static_cast<int>(h*aspect);
    h = divide_even(h, p*4); w = divide_even(w, p*4);
    return {h, w, h/p, w/p, h/p/merge, w/p/merge};
}
std::vector<float> pack_rgb(const std::vector<uint8_t>& rgb, int h, int w, int p) {
    require(h>0 && w>0 && p>0 && h%p==0 && w%p==0, "invalid NaViT patch dimensions");
    require(static_cast<uint64_t>(h)*w*3 == rgb.size(), "RGB buffer size mismatch");
    std::vector<float> out(rgb.size());
    size_t dst=0;
    for (int c=0;c<3;++c) for (int py=0;py<p;++py)
        for (int gy=0;gy<h/p;++gy) for (int gx=0;gx<w/p;++gx) for (int px=0;px<p;++px) {
            const size_t src=(static_cast<size_t>(gy*p+py)*w+gx*p+px)*3+c;
            out[dst++]=(static_cast<float>(rgb[src])/255.f-0.5f)/0.5f;
        }
    return out;
}
std::vector<int> nearest_position_ids(int h, int w, int side) {
    require(h>0 && w>0 && side>0 && static_cast<int64_t>(side)*side<=std::numeric_limits<int>::max(),
            "invalid learned position grid");
    // torch's default float32 arange boundaries, bucketize(...,right=True).
    const double step=1.0/side;
    std::vector<float> boundaries;
    for (int i=0;i<side-1;++i) boundaries.push_back(static_cast<float>(step+i*step));
    std::vector<int> result;
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        int by=std::upper_bound(boundaries.begin(),boundaries.end(),static_cast<float>(y)/h)-boundaries.begin();
        int bx=std::upper_bound(boundaries.begin(),boundaries.end(),static_cast<float>(x)/w)-boundaries.begin();
        result.push_back(by*side+bx);
    }
    return result;
}
CanvasPositions canvas_positions(const std::vector<int>& ids, const MarkerIds& m,
                                  int h, int w, const std::string& mode, int64_t base) {
    const int merge=merge_factor(mode);
    require(h>0 && w>0 && h%merge==0 && w%merge==0 && base>=0, "invalid canvas dimensions or offset");
    const int64_t count=static_cast<int64_t>(h/merge)*(w/merge);
    size_t start=ids.size(); int starts=0, ends=0;
    for(size_t i=0;i<ids.size();++i) {
        require(ids[i]!=m.slice_start && ids[i]!=m.slice_end, "MiniCPM detail slices are unsupported");
        if(ids[i]==m.image_start) { start=i; ++starts; }
        if(ids[i]==m.image_end) ++ends;
    }
    require(starts==1 && ends==1, "exactly one explicitly marked MiniCPM image is required");
    require(count<=static_cast<int64_t>(ids.size()) && start+1+count<ids.size(), "visual token count mismatch");
    const size_t end=start+1+static_cast<size_t>(count);
    require(ids[end]==m.image_end, "missing image end marker or visual token count mismatch");
    for(size_t i=0;i<ids.size();++i)
        require((ids[i]==m.image)==(i>start && i<end), "visual placeholder span mismatch");
    CanvasPositions out;
    out.positions.resize(ids.size());
    for(size_t i=0;i<start;++i) out.positions[i]={{base+static_cast<int64_t>(i),base+static_cast<int64_t>(i),base+static_cast<int64_t>(i)}};
    const int64_t origin=base+start;
    out.positions[start]={{origin,std::max<int64_t>(origin-1,0),std::max<int64_t>(origin-1,0)}};
    for(int y=0;y<h/merge;++y) for(int x=0;x<w/merge;++x)
        out.positions[start+1+static_cast<size_t>(y)*(w/merge)+x]={{origin,origin+y,origin+x}};
    out.positions[end]={{origin,origin+h/merge,origin+w/merge}};
    // HF groups all trailing marker/newline tokens into the frame; newlines
    // immediately after </image> inherit the canvas origin.
    size_t trailing=end+1;
    while(trailing<ids.size() && ids[trailing]==m.newline) out.positions[trailing++]={{origin,origin,origin}};
    int64_t next=origin+std::max(h/merge,w/merge)+1;
    for(size_t i=trailing;i<ids.size();++i,++next) out.positions[i]={{next,next,next}};
    out.next_position=next;
    out.rope_delta=next-static_cast<int64_t>(ids.size());
    return out;
}
void interleaved_rope(const CanvasPositions& p, int dim, double theta,
                      const std::array<int,3>& sections, std::vector<float>& cc, std::vector<float>& ss) {
    require(dim>0 && dim%2==0 && std::isfinite(theta) && theta>0, "invalid rotary dimensions/theta");
    require(sections[0]>=0 && sections[1]>=0 && sections[2]>=0 &&
            static_cast<int64_t>(sections[0])+sections[1]+sections[2]==dim/2,
            "mrope sections must sum to rotary_dim/2");
    cc.resize(p.positions.size()*(dim/2)); ss.resize(cc.size());
    for(size_t i=0;i<p.positions.size();++i) for(int j=0;j<dim/2;++j) {
        int axis=(j%3==1 && j<3*sections[1])?1:((j%3==2 && j<3*sections[2])?2:0);
        const double angle=p.positions[i][axis]/std::pow(theta,static_cast<double>(2*j)/dim);
        cc[i*(dim/2)+j]=static_cast<float>(std::cos(angle));
        ss[i*(dim/2)+j]=static_cast<float>(std::sin(angle));
    }
}
} } // namespace ncnn_llm::minicpm
