// Build and run with CMake/CTest; exercises the shared image and RoPE utilities.
#include "utils/image_utils.h"
#include "utils/rope_embed.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace ncnn_llm::minicpm;
// Keep numerical and rejection checks active in Release builds.
#define CHECK(...) do { if (!(__VA_ARGS__)) throw std::runtime_error("check failed: " #__VA_ARGS__); } while (false)
template<class F> void rejects(F f) { bool caught=false; try {f();} catch(const std::invalid_argument&) {caught=true;} CHECK(caught); }
int main() {
    auto square=plan_thumbnail(100,100,14,448,"16x");
    CHECK(square.height==448 && square.width==448 && square.token_count()==64);
    auto wide=plan_thumbnail(300,400,14,448,"4x");
    CHECK(wide.height==392 && wide.width==504 && wide.token_count()==252);
    auto portrait=plan_thumbnail(400,300,14,448,"16x");
    CHECK(portrait.height==504 && portrait.width==392 && portrait.token_count()==63);
    // Python round tie-to-even: 140 / 56 = 2.5 => 112.
    auto tie=plan_thumbnail(140,140,14,140,"16x"); CHECK(tie.height==112);
    rejects([] {plan_thumbnail(1000,1000,14,448,"16x");});
    CHECK(plan_thumbnail(1000,1000,14,448,"16x",false).token_count()==64);
    CHECK(plan_thumbnail(1000,1000,14,448,"16x",true,1).token_count()==64);
    rejects([] {plan_thumbnail(0,1,14,448,"16x");});
    rejects([] {plan_thumbnail(1,1,14,448,"8x");});
    std::vector<uint8_t> rgb(4*4*3);
    for(size_t i=0;i<rgb.size();++i) rgb[i]=static_cast<uint8_t>(i);
    auto packed=pack_rgb(rgb,4,4,2);
    // c=0, local patch row=0: global pixels 0,1,2,3,8,9,10,11.
    CHECK(std::abs(packed[4]-((24.f/255.f-.5f)/.5f))<1e-7);
    CHECK(std::abs(packed[8]-((12.f/255.f-.5f)/.5f))<1e-7);
    CHECK(std::abs(packed[16]-((1.f/255.f-.5f)/.5f))<1e-7);
    rejects([&] {pack_rgb(rgb,3,4,2);});
    CHECK((nearest_position_ids(2,3,4)==std::vector<int>{0,1,2,8,9,10}));
    const MarkerIds m={100,101,102,103,104,105};
    std::vector<int> ids={7,8,101,100,100,100,100,100,100,102,9};
    auto pos=canvas_positions(ids,m,8,12,"16x");
    CHECK((pos.positions[2]==std::array<int64_t,3>{{2,1,1}}));
    CHECK((pos.positions[3]==std::array<int64_t,3>{{2,2,2}}));
    CHECK((pos.positions[8]==std::array<int64_t,3>{{2,3,4}}));
    CHECK((pos.positions[9]==std::array<int64_t,3>{{2,4,5}}));
    CHECK((pos.positions[10]==std::array<int64_t,3>{{6,6,6}}));
    CHECK(pos.next_position==7 && pos.rope_delta==-4);
    CHECK(canvas_positions(ids,m,4,6,"4x").positions==pos.positions);
    auto offset=canvas_positions(ids,m,8,12,"16x",20);
    CHECK(offset.next_position==27 && offset.positions[2][1]==21);
    std::vector<int> no_prefix={101,100,102,105,105,9};
    auto zero=canvas_positions(no_prefix,m,4,4,"16x");
    CHECK((zero.positions[0]==std::array<int64_t,3>{{0,0,0}}));
    CHECK(zero.positions[2][1]==1 && zero.positions[3][1]==0 && zero.positions[5][1]==2);
    CHECK(zero.next_position==3 && zero.rope_delta==-3);
    rejects([&] {canvas_positions(ids,m,8,12,"4x");});
    auto bad=ids; bad.push_back(103); rejects([&] {canvas_positions(bad,m,8,12,"16x");});
    bad=ids; bad[0]=100; rejects([&] {canvas_positions(bad,m,8,12,"16x");});
    bad=ids; bad.push_back(101); rejects([&] {canvas_positions(bad,m,8,12,"16x");});
    std::vector<float> cc,ss;
    interleaved_rope(pos,12,10000,{{2,2,2}},cc,ss);
    CHECK(cc.size()==ids.size()*6 && ss.size()==cc.size());
    CHECK(std::abs(cc[2*6+1]-std::cos(1.0/std::pow(10000.,2./12)))<1e-6);
    rejects([&] {interleaved_rope(pos,12,10000,{{1,1,1}},cc,ss);});
    std::cout << "MiniCPM thumbnail, packing, canvas and RoPE tests passed\n";
}
