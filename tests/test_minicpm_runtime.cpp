// Real-checkpoint smoke/parity runner. Not registered with CTest: weights are optional.
#include "ncnn_llm_gpt.h"
#include "minicpm_vision_runtime.h"
#include "utils/prompt.h"
#include "utils/minicpm_tokenizer_config.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--check-limits") {
            using ncnn_llm::MiniCPMVisionRuntime;
            MiniCPMVisionRuntime::validate_thumbnail({896,896,64,64,16,16});
            for(auto shape : std::vector<std::pair<int,int>>{{1,4097},{0,64},{-1,64},{2147483647,2147483647}}) {
                bool rejected=false;
                try { MiniCPMVisionRuntime::validate_thumbnail({0,0,shape.first,shape.second,0,0}); }
                catch(const std::runtime_error&) { rejected=true; }
                if(!rejected) throw std::runtime_error("thumbnail limit boundary was accepted");
            }
            const auto extreme=ncnn_llm::minicpm::plan_thumbnail(1,16384,14,448,"16x",false,1);
            bool rejected=false;
            try { MiniCPMVisionRuntime::validate_thumbnail(extreme); }
            catch(const std::runtime_error&) { rejected=true; }
            if(!rejected) throw std::runtime_error("extreme-aspect thumbnail was accepted");
            nlohmann::json manifest={{"type","minicpm_v47"},{"format_version",1},{"weight_dtype","float32"},
                {"scope","single_image_thumbnail"},{"downsample_mode","16x"},{"hidden_size",1152},{"output_hidden_size",1024},
                {"position_grid_size",70},{"patch_size",14},{"window_kernel_size",{2,2}},{"merge_kernel_size",{2,2}},
                {"num_hidden_layers",27},{"insert_layer_id",6},{"position_embeddings",{{"rows",4900},{"cols",1152}}}};
            MiniCPMVisionRuntime::validate_manifest(manifest);
            for(auto bad : std::vector<nlohmann::json>{{{"format_version",2}},{{"weight_dtype","bfloat16"}},
                    {{"hidden_size",2147483647}},{{"position_grid_size",2147483647}},{{"window_kernel_size",{2,3}}},
                    {{"insert_layer_id",27}},{{"num_hidden_layers",0}},{{"position_embeddings",{{"rows",4899},{"cols",1152}}}}}) {
                auto changed=manifest; changed.update(bad); bool rejected=false;
                try { MiniCPMVisionRuntime::validate_manifest(changed); }
                catch(const std::exception&) { rejected=true; }
                if(!rejected) throw std::runtime_error("invalid manifest boundary was accepted");
            }
            ncnn_llm_gpt_base_ctx base; base.image_count=1;
            qwen3_5_ctx hybrid; hybrid.image_count=1;
            if(base.clone()->image_count!=1 || hybrid.clone()->image_count!=1)
                throw std::runtime_error("context clone dropped image count");
            std::cout<<"Thumbnail, manifest and context-clone limits passed\n"; return 0;
        }
        if(argc==4 && std::string(argv[1])=="--tokenize") {
            const std::string dir=argv[2];
            std::ifstream config_file(dir+"/model.json"), reference_file(argv[3]);
            nlohmann::json config, reference; config_file>>config; reference_file>>reference;
            const auto& tc=config.at("tokenizer");
            auto tokenizer=BpeTokenizer::LoadFromFiles(dir+"/"+tc.at("vocab_file").get<std::string>(),
                dir+"/"+tc.at("merges_file").get<std::string>(),SpecialTokensConfig{},false,true,tc.value("type",std::string("bpe"))=="bbpe");
            for(const auto& token:tc.at("additional_special_tokens")) tokenizer.AddAdditionalSpecialToken(token.get<std::string>());
            const bool is_minicpm=config.at("setting").at("vision").value("type",std::string())=="minicpm_v4_7";
            const auto artifact=ncnn_llm::minicpm_tokenizer_artifact(dir,tc,is_minicpm);
            if(is_minicpm) tokenizer.LoadMiniCPMTokenizer(artifact);
            auto ids=tokenizer.encode(reference.at("text").get<std::string>(),false,false);
            const auto bos=tc.at("bos").get<std::string>();
            if(!bos.empty() && config.at("setting").at("vision").value("type",std::string())!="minicpm_v4_7") ids.insert(ids.begin(),tokenizer.token_to_id().at(bos));
            const auto expected=reference.at("ids").get<std::vector<int>>();
            std::cout<<nlohmann::json({{"ids",ids},{"matched",ids==expected},{"count",ids.size()}}).dump()<<"\n";
            return ids==expected?0:1;
        }
        if(argc<4) { std::cerr<<"Usage: test_minicpm_runtime MODEL_DIR IMAGE PROMPT [MAX_TOKENS=32] [VISION_F32_DUMP]\n       test_minicpm_runtime --tokenize MODEL_DIR HF_REFERENCE_JSON\n"; return 2; }
        const std::string dir=argv[1]; auto image=load_image_to_ncnn_mat(argv[2]);
        if(ncnn_mat_empty(image)) throw std::runtime_error("cannot load image");
        std::ifstream file(dir+"/model.json"); nlohmann::json config; file>>config;
        if(argc>5) {
            ncnn_llm::MiniCPMVisionRuntime vision(dir,config.at("setting").at("vision"),4);
            ncnn_llm::minicpm::Thumbnail thumb{}; auto features=vision.encode(image,thumb);
            std::ofstream dump(argv[5],std::ios::binary);
            for(int r=0;r<features.h;++r) dump.write(reinterpret_cast<const char*>(features.row(r)),features.w*sizeof(float));
            if(!dump) throw std::runtime_error("cannot write vision dump");
            std::cerr<<"vision_shape="<<features.h<<","<<features.w<<"\n";
        }
        ncnn_llm_gpt model(dir,false,4,0,false,true);
        const std::vector<Message> messages={{"system","You are a helpful assistant."},
            {"user","<image>\n"+std::string(argv[3])}};
        const auto full=apply_chat_template(TemplateType::MINICPM_V47,messages,{},true,false);
        const auto suffix=apply_chat_template(TemplateType::MINICPM_V47,{messages.back()},{},true,false);
        auto ctx=model.prefill(full.substr(0,full.size()-suffix.size()));
        ctx=model.prefill(suffix,image,ctx);
        std::cerr<<"prefill_token="<<ctx->cur_token<<" next_position="<<ctx->position_id<<"\n";
        GenerateConfig cfg; cfg.max_new_tokens=argc>4?std::stoi(argv[4]):32; cfg.do_sample=0;
        cfg.temperature=0; cfg.repetition_penalty=1; cfg.top_k=0; cfg.top_p=1;
        model.generate(ctx,cfg,[](const std::string& text){ std::cout<<text<<std::flush; });
        std::cout<<"\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
