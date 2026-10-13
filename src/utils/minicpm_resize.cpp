// Copyright (c) 2026 ncnn_llm authors. Apache-2.0.
// Fixed-point conventions adapted from Pillow 12.3.0 Resample.c:
// https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c
/*
The Python Imaging Library (PIL) is
    Copyright (c) 1997-2011 by Secret Labs AB
    Copyright (c) 1995-2011 by Fredrik Lundh and contributors
Pillow is the friendly PIL fork. It is
    Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors
Like PIL, Pillow is licensed under the open source MIT-CMU License:
By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:
Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.
SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
*/
#include "minicpm_resize.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ncnn_llm { namespace minicpm { namespace {
constexpr int kPrecision=22;
struct Taps { int first; std::vector<int32_t> weight; };
size_t buffer_size(int h,int w) {
    if(h<=0 || w<=0 || static_cast<uint64_t>(h)*w>std::numeric_limits<size_t>::max()/3)
        throw std::invalid_argument("invalid bicubic image dimensions");
    return static_cast<size_t>(h)*w*3;
}
double cubic(double x) {
    x=std::abs(x);
    if(x<1.0) return (1.5*x-2.5)*x*x+1.0;
    if(x<2.0) return (((x-5.0)*x+8.0)*x-4.0)*-0.5;
    return 0.0;
}
std::vector<Taps> coefficients(int source,int target) {
    const double scale=static_cast<double>(source)/target;
    const double stretch=std::max(scale,1.0), support=2.0*stretch;
    std::vector<Taps> result(target);
    for(int dst=0;dst<target;++dst) {
        const double center=(dst+0.5)*scale;
        const int begin=static_cast<int>(std::max(0.0,center-support+0.5));
        const int end=static_cast<int>(std::min(static_cast<double>(source),center+support+0.5));
        auto& taps=result[dst]; taps.first=begin;
        std::vector<double> weight(end-begin);
        double sum=0;
        for(int s=begin;s<end;++s) {
            const double v=cubic((s-center+0.5)*(1.0/stretch));
            weight[s-begin]=v; sum+=v;
        }
        taps.weight.reserve(weight.size());
        for(double value:weight) {
            if(sum!=0.0) value/=sum;
            const double fixed=value*(1<<kPrecision);
            taps.weight.push_back(static_cast<int32_t>(fixed+(value<0?-0.5:0.5)));
        }
    }
    return result;
}
uint8_t finish(int64_t value) {
    // Rounding bias already accumulated. Avoid signed negative right shift.
    if(value<=0) return 0;
    return static_cast<uint8_t>(std::min<int64_t>(255,value/(1<<kPrecision)));
}
} // namespace
std::vector<uint8_t> resize_bicubic_rgb(const std::vector<uint8_t>& pixels,
    int h,int w,int th,int tw,bool bgr) {
    if(pixels.size()!=buffer_size(h,w)) throw std::invalid_argument("bicubic input size mismatch");
    buffer_size(th,tw);
    std::vector<uint8_t> current=pixels;
    if(bgr) for(size_t i=0;i<current.size();i+=3) std::swap(current[i],current[i+2]);
    // Pillow 12.3.0 quantizes/clips after each pass, horizontal first.
    if(w!=tw) {
        const auto taps=coefficients(w,tw);
        std::vector<uint8_t> next(buffer_size(h,tw));
        for(int y=0;y<h;++y) for(int x=0;x<tw;++x) for(int c=0;c<3;++c) {
            int64_t value=1<<(kPrecision-1);
            for(size_t k=0;k<taps[x].weight.size();++k)
                value+=static_cast<int64_t>(current[(static_cast<size_t>(y)*w+taps[x].first+k)*3+c])*taps[x].weight[k];
            next[(static_cast<size_t>(y)*tw+x)*3+c]=finish(value);
        }
        current.swap(next);
    }
    if(h!=th) {
        const auto taps=coefficients(h,th);
        std::vector<uint8_t> next(buffer_size(th,tw));
        for(int y=0;y<th;++y) for(int x=0;x<tw;++x) for(int c=0;c<3;++c) {
            int64_t value=1<<(kPrecision-1);
            for(size_t k=0;k<taps[y].weight.size();++k)
                value+=static_cast<int64_t>(current[((taps[y].first+k)*tw+x)*3+c])*taps[y].weight[k];
            next[(static_cast<size_t>(y)*tw+x)*3+c]=finish(value);
        }
        current.swap(next);
    }
    return current;
}
} }
