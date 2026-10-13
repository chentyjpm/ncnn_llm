#include "utils/minicpm_tokenizer_config.h"
#include <fstream>
#include <iostream>
#include <chrono>
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    namespace fs=std::filesystem;
    fs::create_directories(argv[1]);
    const fs::path root=fs::path(argv[1])/("case-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if(!fs::create_directory(root)) throw std::runtime_error("cannot create fresh fixture directory");
    struct Cleanup {fs::path path; ~Cleanup(){std::error_code ec;fs::remove_all(path,ec);}} cleanup{root};
    std::ofstream(root/"artifact.txt")<<"test";
    auto valid=nlohmann::json{{"pretokenizer",{{"type","minicpm_v47"},{"file","artifact.txt"}}}};
    int checked=0;
    auto reject=[&](const nlohmann::json& config,bool mini) {
        try { ncnn_llm::minicpm_tokenizer_artifact(root.string(),config,mini); }
        catch(const std::exception&) { ++checked; return; }
        throw std::runtime_error("invalid tokenizer configuration accepted");
    };
    if(ncnn_llm::minicpm_tokenizer_artifact(root.string(),valid,true)!=(fs::canonical(root)/"artifact.txt").string()) return 1;
    if(!ncnn_llm::minicpm_tokenizer_artifact(root.string(),nlohmann::json::object(),false).empty()) return 1;
    reject(nlohmann::json::object(),true); reject(valid,false);
    auto bad=valid; bad["pretokenizer"]["type"]="legacy"; reject(bad,true);
    for(const auto& name:{"../artifact.txt","/tmp/escape","a\\b","a:b","missing","./artifact.txt",""}) {
        bad=valid; bad["pretokenizer"]["file"]=name; reject(bad,true);
    }
    std::error_code symlink_error;
    fs::create_symlink("artifact.txt",root/"link",symlink_error);
    if(!symlink_error) {
        bad=valid; bad["pretokenizer"]["file"]="link"; reject(bad,true);
    } else {
#ifdef _WIN32
        std::cout<<"Symlink case skipped: OS did not allow fixture creation\n";
#else
        throw fs::filesystem_error("cannot create symlink fixture",root/"link",symlink_error);
#endif
    }
    fs::resize_file(root/"artifact.txt",1024*1024+1); reject(valid,true);
    std::cout<<"Tokenizer configuration checks passed: 2 accepted, "<<checked<<" rejected\n";
}
