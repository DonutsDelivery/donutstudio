// Selected rough camera ownership: raster and the camera ray at a silhouette edge.
//
// A 1080p V2 frame refused because the camera ray hit the glass sphere's triangle 121
// at a pixel where raster drew no fragment. The first case is that tile's map. These
// run the production reconcile rule on the CPU and need no GPU.

#include "../src/gpu_backend/fixture_rough_glass_tiles.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
namespace rt = arbitgpu::roughglasstiles;
using arbitgpu::NativeRawPassPixels;
using arbitgpu::NativeTexturePixelFormat;
constexpr std::uint32_t kGlass=1070017,kGlassMaterial=6,kOpaque=500,kOpaqueMaterial=7;
int failures=0;
void check(bool condition,const char* label)
{
    std::printf("%s: %s\n",condition?"ok":"FAILED",label);
    if(!condition)++failures;
}
struct Tile {
    std::uint32_t width=0,height=0;
    std::vector<std::array<std::uint32_t,4>> selected;
    NativeRawPassPixels objects,materials;
    std::vector<std::size_t> promoted;
    std::string error;
    rt::SelectedOwnerFrame frame;
    // This tile alone is the frame.
    bool run() {
        return frame.prepare(width,height,error) && frame.reconcile(0,0,width,height,kGlass,selected,objects,materials,promoted,error)
            && frame.finish(kGlass,error);
    }
    std::uint32_t object(std::size_t pixel) const {std::uint32_t v=0;std::memcpy(&v,objects.bytes.data()+pixel*4,4);return v;}
    std::uint32_t material(std::size_t pixel) const {std::uint32_t v=0;std::memcpy(&v,materials.bytes.data()+pixel*4,4);return v;}
};
std::array<std::uint32_t,2> owner(char c)
{
    if(c=='G')return {kGlass,kGlassMaterial};
    if(c=='O')return {kOpaque,kOpaqueMaterial};
    return {0,0};
}
// Each string is a row. G is glass, O an opaque object, . nothing.
Tile tile(const std::vector<std::string>& selection,const std::vector<std::string>& raster)
{
    Tile out;out.width=std::uint32_t(selection[0].size());out.height=std::uint32_t(selection.size());
    const auto pixels=std::size_t(out.width)*out.height;
    out.selected.resize(pixels);
    for(auto* image:{&out.objects,&out.materials})
    {*image={NativeTexturePixelFormat::R32Uint,out.width,out.height,std::vector<std::uint8_t>(pixels*4)};}
    for(std::size_t y=0;y<out.height;++y)for(std::size_t x=0;x<out.width;++x) {
        const auto pixel=y*out.width+x;
        const auto s=owner(selection[y][x]),r=owner(raster[y][x]);
        out.selected[pixel]={s[0],s[1],0,121};
        std::memcpy(out.objects.bytes.data()+pixel*4,&r[0],4);
        std::memcpy(out.materials.bytes.data()+pixel*4,&r[1],4);
    }
    return out;
}
}

int main()
{
    {
        // Tile 1088,232 of the refused frame. Pixel 88 is row 5, column 3.
        auto t=tile({".................",".................",".................",".................",
                     "GG...............","GGGG.............","GGGGG............","GGGGGGG.........."},
                    {".................",".................",".................",".................",
                     "GG...............","GGG..............","GGGGG............","GGGGGGG.........."});
        check(t.run(),"the refused edge pixel passes");
        check(t.promoted==std::vector<std::size_t>{88},"only that pixel is promoted");
        check(t.object(88)==kGlass && t.material(88)==kGlassMaterial,"it takes the walk's object and material");
        check(t.object(89)==0,"its neighbor stays a miss");
    }
    {
        auto t=tile({"...","GGO","..."},{"...","GOO","..."});
        check(t.run(),"glass selected over an opaque edge passes");
        check(t.object(4)==kGlass,"the walk owns that pixel");
    }
    {
        auto t=tile({"OOO","OO.","..."},{"OOO","O..","..."});
        check(t.run() && t.promoted.empty(),"an opaque silhouette disagreement passes without a promotion");
        check(t.object(4)==0,"raster keeps a pixel with no walk");
    }
    {
        auto t=tile({".....",".....","..G..",".....","....."},{".....",".....",".....",".....","....."});
        check(t.run(),"a needle only the camera ray hit passes");
        check(t.promoted==std::vector<std::size_t>{12},"and the walk owns it");
    }
    {
        // The same frame split into tiles, in both orders, reaches the same answer.
        for(const bool leftFirst:{true,false}) {
            auto left=tile({"..","..","..",".."},{"..","..","..",".."});
            auto right=tile({"..","G.","GG","GG"},{"..","..","GG","GG"});
            rt::SelectedOwnerFrame frame;std::string error;
            check(frame.prepare(4,4,error),"a split frame prepares");
            const auto runLeft=[&]{return frame.reconcile(0,0,2,4,kGlass,left.selected,left.objects,left.materials,left.promoted,error);};
            const auto runRight=[&]{return frame.reconcile(2,0,2,4,kGlass,right.selected,right.objects,right.materials,right.promoted,error);};
            check(leftFirst?(runLeft()&&runRight()):(runRight()&&runLeft()),"its tiles pass");
            check(frame.finish(kGlass,error),"and so does the frame");
            check(frame.promotedPixels()==1,"one pixel is promoted");
            check(frame.silhouettePixels()==4,"of the glass silhouette's 4");
        }
    }
    {
        // The old exemption: a tile border pixel used to pass because its neighbor wasn't read.
        // Selected glass with raster empty in a block wider than one pixel, cut by the border.
        auto left=tile({"GG","GG","GG","GG","GG"},{"..","..","..","..",".."});
        auto right=tile({"GG","GG","GG","GG","GG"},{"..","..","..","..",".."});
        rt::SelectedOwnerFrame frame;std::string error;
        check(frame.prepare(4,5,error),"a border frame prepares");
        const bool tiles=frame.reconcile(0,0,2,5,kGlass,left.selected,left.objects,left.materials,left.promoted,error)
            && frame.reconcile(2,0,2,5,kGlass,right.selected,right.objects,right.materials,right.promoted,error)
            && frame.finish(kGlass,error);
        check(!tiles,"a wide mismatch cut by tile borders refuses");
        check(error.find("primary raster ownership")!=std::string::npos,"and names the mismatch");
    }
    {
        // Pixel on a tile's border whose only explanation sits in the later tile.
        auto left=tile({"G"},{"."});
        auto right=tile({"."},{"G"});
        rt::SelectedOwnerFrame frame;std::string error;
        check(frame.prepare(2,1,error),"a deferred frame prepares");
        check(frame.reconcile(0,0,1,1,kGlass,left.selected,left.objects,left.materials,left.promoted,error),"its border pixel waits for its neighbor");
        check(frame.deferredPixels()==1,"and is counted as deferred");
        check(!frame.reconcile(1,0,1,1,kGlass,right.selected,right.objects,right.materials,right.promoted,error),
              "raster glass with no walk still refuses in the later tile");
    }
    {
        // A deferred pixel with nothing to explain it fails at the frame's end, not at its tile.
        rt::SelectedOwnerFrame frame;std::string error;
        auto left=tile({"G"},{"O"});
        auto right=tile({"."},{"."});
        check(frame.prepare(2,1,error),"a late refusal prepares");
        check(frame.reconcile(0,0,1,1,kGlass,left.selected,left.objects,left.materials,left.promoted,error),"its first tile defers");
        check(frame.reconcile(1,0,1,1,kGlass,right.selected,right.objects,right.materials,right.promoted,error),"its second tile passes");
        check(!frame.finish(kGlass,error),"and the frame refuses");
    }
    {
        // Every silhouette pixel promoted is the most a frame can report, and it still passes.
        auto t=tile({"G"},{"."});
        std::string error;
        check(t.frame.prepare(1,1,error) && t.frame.reconcile(0,0,1,1,kGlass,t.selected,t.objects,t.materials,t.promoted,error),"an all-promoted frame reconciles");
        check(t.frame.finish(kGlass,error),"and the counts don't refuse it");
        check(t.frame.promotedPixels()==1 && t.frame.silhouettePixels()==1,"they report 1 of 1");
    }
    {
        // Linear depth against the raster vertex's own arithmetic, under a rotation.
        const std::array<float,4> q{0.1830127f,0.1830127f,0.6830127f,0.6830127f};
        const auto rotate=[](std::array<float,4> r,std::array<float,3> v) {
            const auto cross=[](std::array<float,3> a,std::array<float,3> b)->std::array<float,3> {
                return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};};
            const std::array<float,3> u{r[0],r[1],r[2]};
            auto c=cross(u,v);for(int i=0;i<3;++i)c[i]+=r[3]*v[i];
            const auto d=cross(u,c);
            return std::array<float,3>{v[0]+2*d[0],v[1]+2*d[1],v[2]+2*d[2]};
        };
        const std::array<float,3> translation{1,2,3},position{4,-1,-5};
        const auto camera=rotate({-q[0],-q[1],-q[2],q[3]},{position[0]-translation[0],position[1]-translation[1],position[2]-translation[2]});
        const auto expected=std::clamp((-camera[2]-.5f)/(100.f-.5f),0.f,1.f);
        check(std::abs(rt::promotedLinearDepth(position,translation,q,.5f,100.f)-expected)<1e-5f,"promoted depth matches the vertex shader's");
    }
    {
        auto t=tile({"GGGGG","GGGGG","GGGGG","GGGGG","GGGGG"},{"GGGGG","GGGGG","GG.GG","GGGGG","GGGGG"});
        check(t.run(),"a one-pixel raster hole inside glass passes");
    }
    {
        // A selection shifted past one pixel disagrees where no neighbor explains it.
        auto t=tile({"GGGGGGG","GGGGGGG","GGGGGGG","GGGGGGG","GGGGGGG","GGGGGGG","GGGGGGG"},
                    {".......",".......",".......",".......",".......",".......","......."});
        check(!t.run(),"selected glass over wide raster emptiness refuses");
        check(t.error.find("primary raster ownership")!=std::string::npos,"it names the mismatch");
    }
    {
        auto t=tile({"...","...","..."},{"...",".G.","..."});
        check(!t.run(),"raster glass with no selected glass refuses, since no walk ran there");
    }
    {
        auto t=tile({"GG"},{"..",});
        t.selected.pop_back();
        check(!t.run(),"a short selection refuses");
    }
    return failures==0?0:1;
}
