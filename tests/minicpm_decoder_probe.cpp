// Synthetic decoder parity probe; never represents trained model acceptance.
#include "kernel/gdr.h"
#include <net.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <cstdlib>
#include <stdexcept>
#include <sstream>
using J=nlohmann::json;
static ncnn::Mat read_tensor(const std::filesystem::path& root,const J& spec) {
    auto shape=spec.at("shape").get<std::vector<int>>(); ncnn::Mat m;
    for(int n:shape) if(n<1) throw std::runtime_error("invalid tensor dimension");
    if(shape.size()==1) m.create(shape[0]);
    else if(shape.size()==2) m.create(shape[0],shape[1]);
    else if(shape.size()==3) m.create(shape[0],shape[1],shape[2]);
    else throw std::runtime_error("probe supports dimensions1..3 only");
    if(spec.value("zero",false)) {m.fill(0.f); return m;}
    std::ifstream f(root/spec.at("file").get<std::string>(),std::ios::binary);
    if(m.dims<3) f.read(reinterpret_cast<char*>(m.data),m.w*m.h*sizeof(float));
    else for(int q=0;q<m.c;++q) f.read(reinterpret_cast<char*>(m.channel(q).data),m.w*m.h*sizeof(float));
    if(!f || f.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("tensor file length mismatch");
    return m;
}
static J write_tensor(const std::filesystem::path& path,const ncnn::Mat& m) {
    if(m.empty() || m.elempack!=1 || m.elemsize!=4u) throw std::runtime_error("unexpected output precision/packing");
    std::ofstream f(path,std::ios::binary);
    if(m.dims<3) f.write(reinterpret_cast<const char*>(m.data),m.w*m.h*sizeof(float));
    else for(int q=0;q<m.c;++q) f.write(reinterpret_cast<const char*>(m.channel(q).data),m.w*m.h*sizeof(float));
    if(!f) throw std::runtime_error("output write failed");
    J shape=J::array({m.w}); if(m.dims>=2) shape.push_back(m.h); if(m.dims>=3) shape.push_back(m.c);
    return {{"file",path.filename().string()},{"shape",shape}};
}
int main(int argc,char** argv) {
    try {
        if(argc!=5) throw std::runtime_error("Usage: minicpm_decoder_probe PARAM BIN CASE_JSON OUTPUT_DIR");
        std::ifstream input(argv[3]); J spec; input>>spec;
        const auto root=std::filesystem::path(argv[3]).parent_path(), outdir=std::filesystem::path(argv[4]);
        std::filesystem::create_directories(outdir);
        ncnn::Net net; net.opt.num_threads=2; net.opt.use_packing_layout=false;
        net.opt.use_vulkan_compute=false; net.opt.use_bf16_storage=false;
        net.opt.use_fp16_storage=false; net.opt.use_fp16_arithmetic=false; net.opt.use_fp16_packed=false;
        register_gdr_layers(net,true);
        if(net.load_param(argv[1]) || net.load_model(argv[2])) throw std::runtime_error("decoder load failed");
        std::map<std::string,ncnn::Mat> states;
        for(auto it=spec.at("initial_states").begin();it!=spec.at("initial_states").end();++it)
            states[it.key()]=it.value().is_null()?ncnn::Mat():read_tensor(root,it.value());
        J report=J::array(); int index=0;
        for(const auto& step:spec.at("steps")) {
            auto ex=net.create_extractor();
            for(const auto& [name,value]:states) if(ex.input(name.c_str(),value)) throw std::runtime_error("state input failed: "+name);
            for(auto it=step.at("inputs").begin();it!=step.at("inputs").end();++it)
                if(ex.input(it.key().c_str(),read_tensor(root,it.value()))) throw std::runtime_error("input failed: "+it.key());
            if (const char* debug=std::getenv("MINICPM_PROBE_BLOBS")) {
                std::istringstream names(debug); std::string name;
                while(std::getline(names,name,',')) {
                    ncnn::Mat value; const int rc=ex.extract(name.c_str(),value,1);
                    std::cerr<<"blob="<<name<<" rc="<<rc<<" dims="<<value.dims
                             <<" w="<<value.w<<" h="<<value.h<<" d="<<value.d<<" c="<<value.c
#if NCNN_BATCH
                             <<" n="<<value.n
#endif
                             <<" pack="<<value.elempack<<"\n";
                }
            }
            J outputs;
            for(auto it=spec.at("state_outputs").begin();it!=spec.at("state_outputs").end();++it) {
                const auto name=it.value().get<std::string>(); ncnn::Mat result;
                const bool kv=it.key().rfind("cache_k",0)==0 || it.key().rfind("cache_v",0)==0;
                const int rc=ex.extract(name.c_str(),result,kv?1:0);
                if(rc) throw std::runtime_error("state extraction failed: "+name+" rc="+std::to_string(rc));
                // Optimized SDPA KV caches have opaque panel-packed storage;
                // logical w*h bytes include padding and are not tensor data.
                if(kv) outputs[name]={{"shape",J::array({result.w,result.h,result.c})},{"opaque_kvcache",true}};
                else outputs[name]=write_tensor(outdir/(std::to_string(index)+"_"+name+".f32"),result);
                states[it.key()]=result;
            }
            ncnn::Mat result; if(ex.extract("out0",result)) throw std::runtime_error("hidden extraction failed");
            outputs["out0"]=write_tensor(outdir/(std::to_string(index)+"_out0.f32"),result);
            report.push_back(outputs); ++index;
        }
        std::ofstream file(outdir/"outputs.json"); file<<report.dump(2)<<"\n";
        std::cout<<report.dump()<<"\n"; return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n"; return 1;}
}
