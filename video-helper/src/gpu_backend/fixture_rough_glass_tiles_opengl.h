#pragma once

#include "fixture_rough_glass_tiles.h"
#include "../gl_loader.h"
#include <new>

namespace arbitgpu
{
std::string openGlSelectedRoughCameraShaderForFixture(bool transport);
bool readOpenGlRoughGlassTileSummaryForFixture(const NativeFixtureSceneFrame&,
    std::shared_ptr<const roughglasstiles::Schedule>&,roughglasstiles::Aggregate&,std::string&);

namespace roughglasstiles::opengl
{
inline bool completeSubmission(const arbitgl::GlFuncs& gl,std::string& error)
{
    const auto fence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
    if(!fence) {error="OpenGL rough tile completion fence allocation failed";return false;}
    const auto status=gl.ClientWaitSync(fence,GL_SYNC_FLUSH_COMMANDS_BIT,30000000000ull);
    gl.DeleteSync(fence);
    if((status!=GL_ALREADY_SIGNALED&&status!=GL_CONDITION_SATISFIED)||glGetError()!=GL_NO_ERROR)
    {error="OpenGL rough tile submission did not complete successfully";return false;}
    return true;
}
// Native readback of one completed attachment region. Its pixel coordinates
// are bottom-first GL coordinates; the returned tight bytes are top-first.
inline bool readRegion(const arbitgl::GlFuncs& gl,unsigned framebuffer,unsigned attachment,
    std::uint32_t x,std::uint32_t y,std::uint32_t width,std::uint32_t height,
    NativeTexturePixelFormat format,NativeRawPassPixels& output,std::string& error)
{
    output={};
    if(!width || !height || std::uint64_t(width)*height>kMaximumTilePixels
        || std::uint64_t(x)+width>kMaximumNativeFixtureExtent
        || std::uint64_t(y)+height>kMaximumNativeFixtureExtent)
    {error="OpenGL rough tile readback extent exceeds its bounded native region";return false;}
    unsigned layout=GL_RGBA,type=GL_HALF_FLOAT;
    if(format==NativeTexturePixelFormat::Rgba32Float)type=GL_FLOAT;
    else if(format==NativeTexturePixelFormat::Rgba8Unorm)type=GL_UNSIGNED_BYTE;
    else if(format==NativeTexturePixelFormat::R32Uint) {layout=GL_RED_INTEGER;type=GL_UNSIGNED_INT;}
    else if(format!=NativeTexturePixelFormat::Rgba16Float)
    {error="OpenGL rough tile readback format is unsupported";return false;}
    int previousFramebuffer=0,previousReadBuffer=0,previousPackBuffer=0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING,&previousFramebuffer);glGetIntegerv(GL_READ_BUFFER,&previousReadBuffer);
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING,&previousPackBuffer);
    constexpr std::array<unsigned,6> names{GL_PACK_ALIGNMENT,GL_PACK_ROW_LENGTH,GL_PACK_SKIP_PIXELS,
        GL_PACK_SKIP_ROWS,GL_PACK_SWAP_BYTES,GL_PACK_LSB_FIRST};
    std::array<int,6> values{};
    for(std::size_t i=0;i<names.size();++i) {glGetIntegerv(names[i],&values[i]);glPixelStorei(names[i],i==0?1:0);}
    gl.BindBuffer(GL_PIXEL_PACK_BUFFER,0);gl.BindFramebuffer(GL_READ_FRAMEBUFFER,framebuffer);glReadBuffer(attachment);
    const auto rowBytes=std::size_t(width)*rawPassChannels(format)*rawPassScalarBytes(format);
    NativeRawPassPixels candidate{format,width,height,{}};
    bool allocated=true;
    try {candidate.bytes.resize(rowBytes*height);} catch(const std::bad_alloc&) {allocated=false;}
    if(allocated)glReadPixels(int(x),int(y),int(width),int(height),layout,type,candidate.bytes.data());
    const auto gpuError=glGetError();
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER,unsigned(previousFramebuffer));glReadBuffer(unsigned(previousReadBuffer));
    gl.BindBuffer(GL_PIXEL_PACK_BUFFER,unsigned(previousPackBuffer));
    for(std::size_t i=0;i<names.size();++i)glPixelStorei(names[i],values[i]);
    if(!allocated) {error="OpenGL rough tile native readback allocation failed";return false;}
    if(gpuError!=GL_NO_ERROR) {error="OpenGL rough tile completed attachment readback failed";return false;}
    for(std::uint32_t row=0;row<height/2;++row)
        std::swap_ranges(candidate.bytes.data()+row*rowBytes,candidate.bytes.data()+(row+1)*rowBytes,
            candidate.bytes.data()+(height-1-row)*rowBytes);
    output=std::move(candidate);return true;
}
} // namespace roughglasstiles::opengl
} // namespace arbitgpu
