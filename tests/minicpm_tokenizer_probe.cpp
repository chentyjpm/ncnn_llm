// Standalone actual C++ tokenizer probe; no ncnn/Torch dependencies.
#include "utils/tokenizer/bpe_tokenizer.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
static std::string unhex(const std::string& value) {
    std::string out;
    for(size_t i=0;i<value.size();i+=2) out.push_back(static_cast<char>(std::stoul(value.substr(i,2),nullptr,16)));
    return out;
}
static std::string hex(const std::string& value) {
    const char* digits="0123456789abcdef";std::string out;
    for(unsigned char c:value) {out+=digits[c>>4];out+=digits[c&15];}return out;
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("usage: probe vocab merges artifact");
        SpecialTokensConfig spec;
        auto tokenizer=BpeTokenizer::LoadFromFiles(argv[1],argv[2],spec,false,false,true);
        tokenizer.LoadMiniCPMTokenizer(argv[3]);
        // Exercise move construction and assignment after enabling the artifact.
        auto moved=std::move(tokenizer);
        tokenizer=std::move(moved);
        std::string line;
        while(std::getline(std::cin,line)) {
            try {
                auto ids=tokenizer.encode(unhex(line),false,false);
                for(size_t i=0;i<ids.size();++i) {if(i)std::cout<<',';std::cout<<ids[i];}
                std::cout<<'|'<<hex(tokenizer.decode(ids,false))<<'|'<<hex(tokenizer.decode(ids,true))<<'\n';
            } catch(const std::exception& error) {std::cout<<"ERROR:"<<error.what()<<'\n';}
        }
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
