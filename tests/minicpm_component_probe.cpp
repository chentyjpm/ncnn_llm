// Synthetic component parity probe. Input/output floats are planar, unpacked FP32.
#include <net.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void read_exact(std::ifstream& f, float* p, size_t n) {
    if (!f.read(reinterpret_cast<char*>(p), n * sizeof(float)))
        throw std::runtime_error("Truncated input tensor");
}

int main(int argc, char** argv) {
    try {
        if (argc != 9) {
            std::cerr << "Usage: probe param bin input output dims width height channels\n";
            return 2;
        }
        int dims = std::stoi(argv[5]), w = std::stoi(argv[6]);
        int h = std::stoi(argv[7]), c = std::stoi(argv[8]);
        if ((dims != 2 && dims != 3) || w <= 0 || h <= 0 || c <= 0)
            throw std::runtime_error("Invalid dimensions");
        ncnn::Mat input;
        if (dims == 2) input.create(w, h);
        else input.create(w, h, c);
        std::ifstream f(argv[3], std::ios::binary);
        if (dims == 2) read_exact(f, input, size_t(w) * h);
        else for (int q = 0; q < c; ++q) read_exact(f, input.channel(q), size_t(w) * h);
        if (f.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Excess input data");
        ncnn::Net net;
        net.opt.use_vulkan_compute = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage = false;
        net.opt.use_packing_layout = false;
        net.opt.num_threads = 1;
        if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0)
            throw std::runtime_error("Failed to load component");
        auto extractor = net.create_extractor();
        ncnn::Mat output;
        if (extractor.input("in0", input) != 0 || extractor.extract("out0", output) != 0)
            throw std::runtime_error("Component inference failed");
        if (output.dims != 2 || output.elempack != 1 || output.elemsize != sizeof(float))
            throw std::runtime_error("Expected unpacked FP32 output matrix");
        std::ofstream result(argv[4], std::ios::binary);
        result.write(reinterpret_cast<const char*>(static_cast<const float*>(output)),
                     size_t(output.w) * output.h * sizeof(float));
        if (!result) throw std::runtime_error("Failed to write output");
        std::cout << output.h << " " << output.w << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
