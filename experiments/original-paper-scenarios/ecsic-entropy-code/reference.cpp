// Independent fixture wrapper around Fabian Giesen's unmodified public-domain rANS.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
#include "rans_byte.h"

static std::vector<uint8_t> read(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("missing fixture");
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
static uint32_t u32(const std::vector<uint8_t>& b, size_t i) {
    return uint32_t(b.at(i)) | uint32_t(b.at(i+1)) << 8 |
           uint32_t(b.at(i+2)) << 16 | uint32_t(b.at(i+3)) << 24;
}
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    auto tb=read(argv[1]), input=read(argv[2]);
    if (tb.size()!=256*513*4 || input.empty() || input.size()%8) return 3;
    std::vector<uint32_t> cdf;
    for (size_t i=0; i<tb.size(); i+=4) cdf.push_back(u32(tb,i));
    size_t n=input.size()/8;
    std::vector<uint8_t> buffer(4*n+64);
    uint8_t* end=buffer.data()+buffer.size(); uint8_t* ptr=end;
    RansState state; RansEncInit(&state);
    for (size_t j=n; j>0; --j) {
        auto symbol=u32(input,(j-1)*8), table=u32(input,(j-1)*8+4);
        if (symbol>=512 || table>=256) return 4;
        size_t k=table*513+symbol;
        RansEncPut(&state,&ptr,cdf[k],cdf[k+1]-cdf[k],16);
    }
    RansEncFlush(&state,&ptr);
    uint8_t* start=ptr;
    std::ofstream output(argv[3],std::ios::binary);
    output.write(reinterpret_cast<char*>(start),end-start); output.close();
    RansDecInit(&state,&ptr);
    for (size_t j=0; j<n; ++j) {
        auto symbol=u32(input,j*8), table=u32(input,j*8+4);
        auto begin=cdf.begin()+table*513;
        auto decoded=std::upper_bound(begin,begin+513,RansDecGet(&state,16))-begin-1;
        if (uint32_t(decoded)!=symbol) return 5;
        size_t k=table*513+symbol;
        RansDecAdvance(&state,&ptr,cdf[k],cdf[k+1]-cdf[k],16);
    }
    if (ptr!=end || state!=RANS_BYTE_L) return 6;
    return 0;
}
