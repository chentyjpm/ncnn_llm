// A standard-C++ bridge for test_minicpm_vision_reference.py. No model weights.
#include "utils/image_utils.h"
#include "utils/rope_embed.h"
#include <iomanip>
#include <iostream>
using namespace ncnn_llm::minicpm;
int main() {
    std::cout << std::setprecision(9);
    std::string mode;
    int h,w,side,dim,n;
    int64_t base;
    double theta;
    MarkerIds m;
    std::array<int,3> sections;
    while(std::cin >> mode >> h >> w >> side >> base >> dim >> theta
          >> m.image >> m.image_start >> m.image_end >> m.slice_start >> m.slice_end >> m.newline
          >> sections[0] >> sections[1] >> sections[2] >> n) {
        std::vector<int> ids(n); for(auto& id:ids) std::cin >> id;
        auto p=canvas_positions(ids,m,h,w,mode,base);
        std::vector<float> cc,ss; interleaved_rope(p,dim,theta,sections,cc,ss);
        std::cout << p.rope_delta << ' ' << p.next_position << '\n';
        for(auto xyz:p.positions) { for(auto x:xyz) std::cout << x << ' '; }
        std::cout << '\n';
        for(auto x:nearest_position_ids(h,w,side)) std::cout << x << ' ';
        std::cout << '\n';
        for(auto x:cc) std::cout << x << ' ';
        std::cout << '\n';
        for(auto x:ss) std::cout << x << ' ';
        std::cout << '\n';
    }
}
