#include "minicpm_vision_runtime.h"
#include "utils/image_utils.h"
#include "utils/minicpm_resize.h"
#include <cstring>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <set>
#include <filesystem>

namespace ncnn_llm {
namespace {
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error("MiniCPM vision: " + why); }
std::string model_file(const std::string& root, const std::string& name, uintmax_t limit) {
    const std::filesystem::path relative(name);
    require(!name.empty() && name.find('\\')==std::string::npos && name.find(':')==std::string::npos &&
            !relative.is_absolute() && !relative.has_root_path(), "manifest resource must be a relative model file");
    auto path=std::filesystem::canonical(root);
    for(const auto& part:relative) {
        require(part!=".." && part!=".", "manifest resource traversal is unsupported");
        path/=part;
        require(!std::filesystem::is_symlink(path), "manifest resource symlinks are unsupported");
    }
    require(std::filesystem::is_regular_file(path), "missing model resource: "+name);
    require(std::filesystem::file_size(path)<=limit, "model resource exceeds size limit: "+name);
    return path.string();
}
void matrix(const ncnn::Mat& m, int rows, int cols, const std::string& name) {
    require(m.dims == 2 && m.w == cols && m.h == rows && m.elempack == 1 && m.elemsize == 4u, name + " shape/precision mismatch");
}
// Row-major 2x2 windows, each flattened in [dy,dx,channel] order.
ncnn::Mat pack_windows(const ncnn::Mat& in, int h, int w) {
    require(h%2==0 && w%2==0, "odd merge grid");
    ncnn::Mat out(in.w*4,h*w/4);
    require(!out.empty(), "window packing allocation failed");
    for(int y=0;y<h/2;++y) for(int x=0;x<w/2;++x) {
        float* dst=out.row(y*(w/2)+x);
        for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx)
            std::memcpy(dst+(dy*2+dx)*in.w,in.row((y*2+dy)*w+x*2+dx),in.w*sizeof(float));
    }
    return out;
}
}
MiniCPMVisionRuntime::MiniCPMVisionRuntime(const std::string& dir, const nlohmann::json& cfg, int threads) {
    std::ifstream file(model_file(dir,cfg.at("manifest").get<std::string>(),1024*1024));
    require(file.good(), "missing manifest");
    nlohmann::json manifest; file>>manifest;
    validate_manifest(manifest);
    mode=manifest.at("downsample_mode").get<std::string>();
    require(mode=="16x", "only 16x component graphs are supported");
    hidden=manifest.at("hidden_size"); output_hidden=manifest.at("output_hidden_size");
    patch=manifest.at("patch_size"); position_side=manifest.at("position_grid_size");
    std::fprintf(stderr, "[MiniCPM vision] Preprocessing: official MiniCPMV4_6ImageProcessorPil-compatible, Pillow 12.3 bicubic; default Torchvision resize is a different backend.\n");
    scale=cfg.value("scale_resolution",448); slice_mode=cfg.value("slice_mode",true); max_slices=cfg.value("max_slice_nums",9);
    require(scale>0 && scale<=896 && max_slices>=1 && max_slices<=9, "unsupported thumbnail scale or slice limit");
    auto ids=cfg.at("marker_ids");
    markers={ids.at("image"),ids.at("image_start"),ids.at("image_end"),ids.at("slice_start"),ids.at("slice_end"),ids.at("newline")};
    std::set<int> unique_ids;
    for(int id:{markers.image,markers.image_start,markers.image_end,markers.slice_start,markers.slice_end,markers.newline}) {
        require(id>=0, "negative marker token id"); unique_ids.insert(id);
    }
    require(unique_ids.size()==6, "marker token ids must be distinct");
    for(const char* name:{"patch_embed","encoder_pre","window_attn","window_mlp","encoder_post","projector"}) {
        auto net=std::make_shared<ncnn::Net>();
        net->opt.use_vulkan_compute=false; net->opt.use_packing_layout=false;
        net->opt.use_fp16_packed=false; net->opt.use_fp16_storage=false; net->opt.use_fp16_arithmetic=false; net->opt.use_bf16_storage=false;
        if(threads>0) net->opt.num_threads=threads;
        const auto& component=manifest.at("components").at(name);
        require(net->load_param(model_file(dir,component.at("param").get<std::string>(),16*1024*1024).c_str())==0,"cannot load graph "+std::string(name));
        require(net->load_model(model_file(dir,component.at("bin").get<std::string>(),uintmax_t(8)*1024*1024*1024).c_str())==0,"cannot load weights "+std::string(name));
        graphs[name]=net;
    }
    auto pos=manifest.at("position_embeddings");
    require(pos.at("rows").get<int>()==position_side*position_side && pos.at("cols").get<int>()==hidden,"position table shape mismatch");
    position.resize(static_cast<size_t>(position_side)*position_side*hidden);
    std::ifstream weights(model_file(dir,pos.at("file").get<std::string>(),position.size()*sizeof(float)),std::ios::binary);
    require(weights.good() && bool(weights.read(reinterpret_cast<char*>(position.data()),position.size()*sizeof(float))),"truncated position table");
    require(weights.peek()==std::char_traits<char>::eof(),"oversized position table");
}
ncnn::Mat MiniCPMVisionRuntime::run(const std::string& name,const ncnn::Mat& input) const {
    auto ex=graphs.at(name)->create_extractor(); ncnn::Mat out;
    require(ex.input("in0",input)==0,"input failed: "+name);
    require(ex.extract("out0",out)==0 && !out.empty(),"extraction failed: "+name);
    return out;
}
void MiniCPMVisionRuntime::validate_manifest(const nlohmann::json& m) {
    require(m.at("type")=="minicpm_v47" && m.at("format_version")==1 && m.at("weight_dtype")=="float32" &&
            m.at("scope")=="single_image_thumbnail" && m.at("downsample_mode")=="16x", "unsupported manifest format/precision/scope");
    const int h=m.at("hidden_size"), out=m.at("output_hidden_size"), side=m.at("position_grid_size");
    require(h>0 && h<=2048 && out>0 && out<=8192 && side>0 && side<=128 && m.at("patch_size")==14,
            "unsupported or excessive architecture dimensions");
    require(m.at("window_kernel_size")==nlohmann::json::array({2,2}) &&
            m.at("merge_kernel_size")==nlohmann::json::array({2,2}), "only 2x2 window and projector merges are supported");
    const int layers=m.at("num_hidden_layers"), insertion=m.at("insert_layer_id");
    require(layers>1 && layers<=64 && insertion>=0 && insertion<layers-1, "invalid encoder partition");
    require(m.at("position_embeddings").at("rows").get<int>()==side*side &&
            m.at("position_embeddings").at("cols").get<int>()==h, "position table shape mismatch");
}
void MiniCPMVisionRuntime::validate_thumbnail(const minicpm::Thumbnail& t) {
    require(t.patch_rows>0 && t.patch_cols>0 && static_cast<int64_t>(t.patch_rows)*t.patch_cols <= 4096,
            "experimental thumbnail exceeds 4096 ViT patches or has invalid dimensions; extreme aspect ratios are unsupported");
}
ncnn::Mat MiniCPMVisionRuntime::encode(const ncnn::Mat& bgr,minicpm::Thumbnail& t) const {
    require(!bgr.empty() && bgr.dims==3 && bgr.c==3 && bgr.elemsize==1u && bgr.elempack==1,"expected interleaved BGR uint8 image");
    require(bgr.w>0 && bgr.h>0 && bgr.w<=16384 && bgr.h<=16384 && static_cast<int64_t>(bgr.w)*bgr.h<=16000000, "input image exceeds the experimental dimension limit");
    t=minicpm::plan_thumbnail(bgr.h,bgr.w,patch,scale,mode,slice_mode,max_slices);
    validate_thumbnail(t);
    const unsigned char* raw=static_cast<const unsigned char*>(bgr.data);
    std::vector<uint8_t> source(raw,raw+static_cast<size_t>(bgr.w)*bgr.h*3);
    const auto rgb=minicpm::resize_bicubic_rgb(source,bgr.h,bgr.w,t.height,t.width,true);
    ncnn::Mat pixels(t.width,t.height,3);
    require(!pixels.empty(), "pixel allocation failed");
    for(int c=0;c<3;++c) {
        float* values=pixels.channel(c);
        for(size_t i=0;i<static_cast<size_t>(t.width)*t.height;++i)
            values[i]=(static_cast<float>(rgb[i*3+c])/255.f-0.5f)/0.5f;
    }
    ncnn::Mat x=run("patch_embed",pixels);
    matrix(x,t.patch_rows*t.patch_cols,hidden,"patch_embed");
    auto positions=minicpm::nearest_position_ids(t.patch_rows,t.patch_cols,position_side);
    for(int i=0;i<x.h;++i) { float* row=x.row(i); const float* p=position.data()+positions[i]*hidden; for(int c=0;c<hidden;++c) row[c]+=p[c]; }
    x=run("encoder_pre",x); matrix(x,t.patch_rows*t.patch_cols,hidden,"encoder_pre");
    ncnn::Mat windows=pack_windows(x,t.patch_rows,t.patch_cols);
    ncnn::Mat attended(hidden*4,windows.h);
    require(!attended.empty(), "window attention allocation failed");
    for(int i=0;i<windows.h;++i) {
        // A row view has no refcount. ncnn's light-mode in-place dispatcher
        // needs an owned input before the attention branch's LayerNorm.
        ncnn::Mat one=ncnn::Mat(hidden,4,static_cast<void*>(windows.row(i))).clone();
        require(!one.empty(), "window input allocation failed");
        ncnn::Mat result=run("window_attn",one); matrix(result,4,hidden,"window_attn");
        std::memcpy(attended.row(i),result.data,4*hidden*sizeof(float));
    }
    x=run("window_mlp",attended); matrix(x,windows.h,hidden,"window_mlp");
    for(int i=0;i<x.h;++i) { float* row=x.row(i); const float* a=attended.row(i); for(int c=0;c<hidden;++c) row[c]+=(a[c]+a[hidden+c]+a[2*hidden+c]+a[3*hidden+c])*0.25f; }
    x=run("encoder_post",x); matrix(x,windows.h,hidden,"encoder_post");
    x=run("projector",pack_windows(x,t.patch_rows/2,t.patch_cols/2));
    matrix(x,t.token_count(),output_hidden,"projector");
    return x;
}
}
