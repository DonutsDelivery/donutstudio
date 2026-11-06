#pragma once
// Smooth caustic reflection history: one bit per glass interface, set when the
// branch reflected there. The CPU trace, the OpenGL shader and the Metal shader
// all carry 128 bits as four uint32 words, word i holding interfaces 32i..32i+31.
// OpenGL keeps words 0-1 in Branch.history and words 2-3 in a parallel
// stackHigh entry; Metal keeps words 0-1 in CBranch.history and words 2-3 in
// CBranch.historyWide. The mirror functions below copy each shader's bit-set
// and packing code so a CPU test can prove both backends write the same words.
// Past 128 interfaces both kernels keep the remaining words in a tail after the
// path receipts, sized from the policy's interface count (see the tail mirrors).
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace arbitgpu::glasscaustics {

// Words the GPU kernels carry inline per branch and deposit. The CPU history
// grows past them: bit i always lives in word i/32, and missing words read as
// zero, so a history never truncates and equal bit sets compare equal whatever
// their word count.
inline constexpr std::size_t kHistoryWords = 4;
inline constexpr std::size_t kHistoryBits = 32*kHistoryWords;

// Words needed for a history of the given interface count.
constexpr std::size_t historyWordsFor(std::size_t interfaces) {return std::max(kHistoryWords,(interfaces+31)/32);}

struct CausticHistory
{
    std::vector<std::uint32_t> words=std::vector<std::uint32_t>(kHistoryWords,0u);
    CausticHistory() = default;
    // Paths of 64 or fewer interfaces (ancestry, grating, oracle) widen in place.
    CausticHistory(std::uint64_t low) : words {std::uint32_t(low),std::uint32_t(low>>32),0,0} {}
    std::uint32_t word(std::size_t w) const {return w<words.size()?words[w]:0u;}
    bool test(std::size_t i) const {return ((word(i/32)>>(i%32))&1u)!=0;}
    CausticHistory with(std::size_t i,bool reflected=true) const
    {
        auto r=*this;
        if(reflected)
        {
            if(r.words.size()<=i/32)r.words.resize(i/32+1,0u);
            r.words[i/32]|=std::uint32_t(1)<<(i%32);
        }
        return r;
    }
    // True when no bit at or above interface n is set.
    bool clearFrom(std::size_t n) const
    {
        for(std::size_t w=0;w<words.size();++w)
        {
            const auto first=32*w;
            if(n<=first) {if(words[w]){return false;}continue;}
            if(n<first+32 && (words[w]>>(n-first))!=0)return false;
        }
        return true;
    }
    std::uint64_t low64() const {return std::uint64_t(word(0))|(std::uint64_t(word(1))<<32);}
    std::uint64_t high64() const {return std::uint64_t(word(2))|(std::uint64_t(word(3))<<32);}
    static CausticHistory fromWords(std::uint32_t a,std::uint32_t b,std::uint32_t c,std::uint32_t d)
    {
        CausticHistory r;r.words={a,b,c,d};return r;
    }
    // Appends words 4 and up, as a kernel's extra history storage returns them.
    CausticHistory withExtraWords(const std::uint32_t* extra,std::size_t count) const
    {
        auto r=*this;r.words.resize(std::max(r.words.size(),kHistoryWords+count),0u);
        for(std::size_t k=0;k<count;++k)r.words[kHistoryWords+k]=extra[k];
        return r;
    }
    friend bool operator==(const CausticHistory& a,const CausticHistory& b)
    {
        const auto n=std::max(a.words.size(),b.words.size());
        for(std::size_t w=0;w<n;++w)if(a.word(w)!=b.word(w))return false;
        return true;
    }
    friend bool operator!=(const CausticHistory& a,const CausticHistory& b) {return !(a==b);}
    // Orders by the highest differing word, so trailing zero words never matter.
    friend bool operator<(const CausticHistory& a,const CausticHistory& b)
    {
        const auto n=std::max(a.words.size(),b.words.size());
        for(std::size_t w=n;w-->0;)if(a.word(w)!=b.word(w))return a.word(w)<b.word(w);
        return false;
    }
};

// OpenGL mirror: Branch.history is uvec2 (x,y), stackHigh is uvec2 (x,y).
struct OpenGlHistoryState {std::uint32_t x=0,y=0,highX=0,highY=0;};
constexpr OpenGlHistoryState openGlHistoryReflect(OpenGlHistoryState s,std::uint32_t interfaces)
{
    if(interfaces<32)s.x|=1u<<interfaces;else if(interfaces<64)s.y|=1u<<(interfaces-32);
    else if(interfaces<96)s.highX|=1u<<(interfaces-64);else if(interfaces<128)s.highY|=1u<<(interfaces-96);
    return s;
}
// Deposit texels 1 and 30 store each word as two exact 16-bit floats, low half first.
constexpr std::array<std::array<float,4>,2> openGlHistoryTexels(const OpenGlHistoryState& s)
{
    const auto halves=[](std::uint32_t a,std::uint32_t b) {
        return std::array<float,4>{float(a&65535u),float(a>>16),float(b&65535u),float(b>>16)};
    };
    return {halves(s.x,s.y),halves(s.highX,s.highY)};
}
inline CausticHistory openGlHistoryDecode(const std::array<std::array<float,4>,2>& texels)
{
    const auto word=[](float low,float high) {return std::uint32_t(low)|(std::uint32_t(high)<<16);};
    return CausticHistory::fromWords(word(texels[0][0],texels[0][1]),word(texels[0][2],texels[0][3]),
        word(texels[1][0],texels[1][1]),word(texels[1][2],texels[1][3]));
}

// Metal mirror: CBranch.history and CBranch.historyWide are ulong.
struct MetalHistoryState {std::uint64_t history=0,historyWide=0;};
constexpr MetalHistoryState metalHistoryReflect(MetalHistoryState s,std::uint32_t interfaces)
{
    if(interfaces<64)s.history|=std::uint64_t(1)<<interfaces;
    else if(interfaces<128)s.historyWide|=std::uint64_t(1)<<(interfaces-64);
    return s;
}
// CDeposit stores historyLow, historyHigh, historyWideLow and historyWideHigh.
constexpr std::array<std::uint32_t,4> metalHistoryDepositWords(const MetalHistoryState& s)
{
    return {std::uint32_t(s.history),std::uint32_t(s.history>>32),std::uint32_t(s.historyWide),std::uint32_t(s.historyWide>>32)};
}
inline CausticHistory metalHistoryDecode(const std::array<std::uint32_t,4>& w)
{
    return CausticHistory::fromWords(w[0],w[1],w[2],w[3]);
}

// History tail past 128 interfaces: extra = historyWordsFor(policy)-4 words per
// block. OpenGL stores word k as 16-bit halves in texel k/2, components
// (k%2)*2 and (k%2)*2+1, like deposit texel 1; Metal stores plain uint words.
// The reflect mirrors copy each kernel's historySet code.
inline void openGlHistoryTailReflect(std::vector<std::array<float,4>>& block,std::size_t extra,std::uint32_t interfaces)
{
    if(interfaces<128||interfaces-128>=32*extra)return;
    const auto bit=interfaces-128,word=bit/32,component=(word%2)*2+(bit%32)/16;
    auto& half=block[word/2][component];
    half=float(std::uint32_t(half)|(1u<<(bit%16)));
}
inline void metalHistoryTailReflect(std::vector<std::uint32_t>& block,std::size_t extra,std::uint32_t interfaces)
{
    if(interfaces>=128&&interfaces-128<32*extra)block[(interfaces-128)/32]|=1u<<((interfaces-128)%32);
}
// Host decode of one OpenGL tail block of (extra+1)/2 texels. Refuses halves
// that are not exact 16-bit integers and a nonzero pad half past the last word.
inline bool openGlHistoryTailDecode(const std::array<float,4>* block,std::size_t extra,std::uint32_t* words)
{
    for(std::size_t k=0;k<(extra+1)/2*2;++k)
    {
        const float low=block[k/2][(k%2)*2],high=block[k/2][(k%2)*2+1];
        const auto exact=[](float v){return v>=0&&v<=65535&&float(std::uint32_t(v))==v;};
        if(!exact(low)||!exact(high)||(k>=extra&&(low!=0||high!=0)))return false;
        if(k<extra)words[k]=std::uint32_t(low)|(std::uint32_t(high)<<16);
    }
    return true;
}

// Grating incidents keep history words 0 and 1 in texel 1 and words 2 and up
// in a per-incident tail of 16-bit halves, laid out like the history tail.
// The mirrors copy each kernel's tail writer: OpenGL takes stackHigh and then
// the walked branch block, Metal takes branchWide and then walked.
inline std::vector<std::array<float,4>> openGlGratingIncidentTail(const OpenGlHistoryState& s,
    const std::vector<std::array<float,4>>& walked,std::size_t tailWords)
{
    std::vector<std::array<float,4>> out((tailWords+1)/2,std::array<float,4>{});
    for(std::size_t j=0;j<tailWords;++j)
    {
        float low=0,high=0;
        if(j<2) {const auto w=j==0?s.highX:s.highY;low=float(w&65535u);high=float(w>>16);}
        else if((j-2)/2<walked.size()) {const auto k=j-2;low=walked[k/2][(k%2)*2];high=walked[k/2][(k%2)*2+1];}
        out[j/2][(j%2)*2]=low;out[j/2][(j%2)*2+1]=high;
    }
    return out;
}
inline std::vector<std::array<float,4>> metalGratingIncidentTail(const MetalHistoryState& s,
    const std::vector<std::uint32_t>& walked,std::size_t tailWords)
{
    std::vector<std::array<float,4>> out((tailWords+1)/2,std::array<float,4>{});
    for(std::size_t j=0;j<tailWords;++j)
    {
        const std::uint32_t w=j==0?std::uint32_t(s.historyWide):j==1?std::uint32_t(s.historyWide>>32)
            :(j-2<walked.size()?walked[j-2]:0u);
        out[j/2][(j%2)*2]=float(w&65535u);out[j/2][(j%2)*2+1]=float(w>>16);
    }
    return out;
}
// Host decode of one incident: texel 1's 64 bits plus its tail words.
inline bool gratingIncidentHistoryDecode(std::uint64_t low,const std::array<float,4>* tail,std::size_t tailWords,CausticHistory& out)
{
    if(!tailWords) {out=CausticHistory(low);return true;}
    if(tailWords<2)return false;
    std::vector<std::uint32_t> words(tailWords);
    if(!openGlHistoryTailDecode(tail,tailWords,words.data()))return false;
    out=CausticHistory::fromWords(std::uint32_t(low),std::uint32_t(low>>32),words[0],words[1]).withExtraWords(words.data()+2,tailWords-2);
    return true;
}
// Camera trace history words 2 and up for one visit, mirroring the visit
// writers on both kernels: the parent's extra lanes, plus this branch's bit
// when it reflected at interface interfaces-1 (64 and up). A root visit has
// no parent and zero lanes. Lane k/4 component k%4 holds word 2+k.
inline std::vector<std::array<std::uint32_t,4>> cameraTraceHistoryLanesMirror(
    const std::vector<std::array<std::uint32_t,4>>* parent,std::uint32_t interfaces,std::uint32_t token,std::size_t extraLanes)
{
    std::vector<std::array<std::uint32_t,4>> out(extraLanes,std::array<std::uint32_t,4>{});
    for(std::size_t lane=0;lane<extraLanes;++lane)
    {
        auto words=parent?(*parent)[lane]:std::array<std::uint32_t,4>{};
        const std::uint32_t reflected=interfaces-1u,extra=reflected/32u-2u;
        if(parent&&token==1u&&reflected>=64u&&extra/4u==lane)words[extra%4u]|=1u<<(reflected%32u);
        out[lane]=words;
    }
    return out;
}

// Cell transport token at one interface, mirroring cellToken (OpenGL) and
// gratingCellToken (Metal): the 32 inline bytes in texels 2..9, then the
// history bit from texel 1 (words 0-1) or the incident's tail (words 2 and up).
// 1 reflects, 65 transmits, 0 is the endpoint past the last interface.
inline std::uint32_t gratingCellTokenMirror(const std::array<float,4>* incident,const std::array<float,4>* tail,
    std::size_t depth,std::size_t interfaces)
{
    if(depth>=interfaces)return 0;
    if(depth<32)return std::uint32_t(incident[2+depth/4][depth%4]);
    const auto w=depth/32;const std::array<float,4>* t=nullptr;bool low=false;
    if(w<2) {t=&incident[1];low=w==0;}
    else {const auto j=w-2;t=&tail[j/2];low=(j%2)==0;}
    const auto word=low?std::uint32_t((*t)[0])|(std::uint32_t((*t)[1])<<16):std::uint32_t((*t)[2])|(std::uint32_t((*t)[3])<<16);
    return ((word>>(depth%32))&1u)!=0u?1u:65u;
}

// Multiple-scatter ancestry tokens for interfaces 32 and up follow the history
// words in the same tail block: token b (interface 32+b) is byte b%4 of word
// historyExtra+b/4. The reflect mirrors copy each kernel's ancestrySet code.
inline void openGlAncestryTailReflect(std::vector<std::array<float,4>>& block,std::size_t historyExtra,
    std::size_t ancestryExtra,std::uint32_t interfaces,std::uint32_t token)
{
    if(interfaces<32||interfaces-32>=4*ancestryExtra)return;
    const std::size_t b=interfaces-32,word=historyExtra+b/4,component=(word%2)*2+(b%4)/2;
    auto& half=block[word/2][component];
    half=float(std::uint32_t(half)|(token<<(8*(b%2))));
}
inline void metalAncestryTailReflect(std::vector<std::uint32_t>& block,std::size_t historyExtra,
    std::size_t ancestryExtra,std::uint32_t interfaces,std::uint32_t token)
{
    if(interfaces>=32&&interfaces-32<4*ancestryExtra)
        block[historyExtra+(interfaces-32)/4]|=token<<(8*((interfaces-32)%4));
}
// Splits a decoded tail block into history words and ancestry tokens.
inline std::vector<std::uint8_t> ancestryTailTokens(const std::uint32_t* words,std::size_t historyExtra,std::size_t ancestryExtra)
{
    std::vector<std::uint8_t> tokens(4*ancestryExtra);
    for(std::size_t b=0;b<tokens.size();++b)tokens[b]=std::uint8_t(words[historyExtra+b/4]>>(8*(b%4)));
    return tokens;
}

} // namespace arbitgpu::glasscaustics
