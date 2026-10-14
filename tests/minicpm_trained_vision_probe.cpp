// Real-checkpoint vision-only parity probe; never allocates the language model.
#include "ncnn_llm_gpt.h"
#include "utils/image_utils.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <csignal>
#include <cstdlib>
#if defined(__GLIBC__) || defined(__APPLE__)
#include <execinfo.h>
#include <unistd.h>
#endif

int main(int argc, char** argv) {
#if defined(__GLIBC__) || defined(__APPLE__)
    // Diagnostic executable only: preserve a stack when a native kernel fails.
    std::signal(SIGSEGV, [](int signal) {
        const char text[] = "Native vision SIGSEGV stack:\n";
        write(STDERR_FILENO, text, sizeof(text) - 1);
        void* stack[64];
        const int count = backtrace(stack, 64);
        backtrace_symbols_fd(stack, count, STDERR_FILENO);
        std::_Exit(128 + signal);
    });
#endif
    try {
        if (argc < 4 || argc > 5) {
            std::cerr << "Usage: minicpm_trained_vision_probe MODEL_DIR IMAGE OUTPUT_F32 [THREADS=2]\n";
            return 2;
        }
        const std::string directory = argv[1];
        const int threads = argc == 5 ? std::stoi(argv[4]) : 2;
        if (threads < 1 || threads > 64) throw std::runtime_error("Invalid thread count");
        std::ifstream config_file(directory + "/model.json");
        if (!config_file) throw std::runtime_error("Cannot read model.json");
        nlohmann::json config;
        config_file >> config;
        ncnn::Mat image = load_image_to_ncnn_mat(argv[2]);
        if (image.empty()) throw std::runtime_error("Cannot decode image");
        std::cerr << "vision_load_start\n";
        auto vision = ncnn_llm_gpt::load_minicpm_vision(directory, config.at("setting").at("vision"), threads);
        std::cerr << "vision_load_complete\n";
        ncnn_llm::minicpm::Thumbnail thumbnail{};
        std::cerr << "vision_encode_start\n";
        const ncnn::Mat output = ncnn_llm_gpt::encode_minicpm_vision(*vision, image, thumbnail);
        std::cerr << "vision_encode_complete\n";
        std::ofstream file(argv[3], std::ios::binary);
        for (int row = 0; row < output.h; ++row)
            file.write(reinterpret_cast<const char*>(output.row(row)), output.w * sizeof(float));
        if (!file) throw std::runtime_error("Cannot write feature output");
        std::cout << nlohmann::json({{"rows", output.h}, {"cols", output.w},
            {"patch_rows", thumbnail.patch_rows}, {"patch_cols", thumbnail.patch_cols},
            {"threads", threads}, {"language_model_loaded", false}}).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
