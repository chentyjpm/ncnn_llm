#include "utils/prompt.h"
#include <iostream>
int main() {
    std::string line;
    while(std::getline(std::cin,line)) {
        try {
            auto j=json::parse(line); std::vector<Message> messages;
            for(const auto& m:j.at("messages")) messages.emplace_back(m.at("role"),m.at("content"),m.value("reasoning_content",std::string()));
            std::cout<<json({{"text",apply_chat_template(TemplateType::MINICPM_V47,messages,{},j.value("add_generation_prompt",true),j.value("enable_thinking",false))}}).dump()<<"\n";
        } catch(const std::exception& e) { std::cout<<json({{"error",e.what()}}).dump()<<"\n"; }
    }
}
