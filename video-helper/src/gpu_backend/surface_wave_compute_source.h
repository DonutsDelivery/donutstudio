#pragma once

// The surface wave compute shader. Only a string, so the GPU-vs-replay test
// (video-helper/tests/surface_wave_gpu_test.cpp) compiles the exact source the
// OpenGL backend runs.

namespace arbitgpu
{

// The GPU frame of a NativeSurfaceWaveFrame, in place in the fixture vertex buffer.
// Mode 0 writes each grid vertex's height from its own rest X and Y, which no pass
// changes. Mode 1, after a barrier, writes its normal: Shade Smooth's angle-weighted
// sum over the up to six grid triangles at the vertex (the grid splits each cell
// a-b-c, b-d-c with a at (x, y), b at (x+1, y), c at (x, y+1)), normalized, with a
// corner under the same sine floor adding nothing and +Z for an empty sum. The grid's
// rest Z is zero, so the height is the whole Z. uBase offsets each dispatch, so a
// grid wider than one dispatch's group limit runs in several.
constexpr const char* kSurfaceWaveComputeSource =
    "#version 430\n"
    "layout(local_size_x=64) in;\n"
    "layout(std430,binding=0) buffer SurfaceVertices{float vertexData[];};\n"
    "layout(std430,binding=1) readonly buffer SurfaceWaves{vec4 waves[];};\n"
    "uniform uint uMode;\n"
    "uniform uint uFirst;\n"
    "uniform uint uColumns;\n"
    "uniform uint uRows;\n"
    "uniform uint uWaves;\n"
    "uniform uint uBase;\n"
    "uint slot(uint x,uint y){return (uFirst+y*uColumns+x)*18u;}\n"
    "vec3 at(uint x,uint y){uint v=slot(x,y);return vec3(vertexData[v],vertexData[v+1u],vertexData[v+2u]);}\n"
    "vec3 corner(vec3 p,vec3 q,vec3 r){\n"
    "    vec3 e1=q-p,e2=r-p,n=cross(e1,e2);float s=length(n);\n"
    "    if(!(s>1.0e-6*sqrt(dot(e1,e1)*dot(e2,e2))))return vec3(0.0);\n"
    "    return n*(atan(s,dot(e1,e2))/s);\n"
    "}\n"
    "void main(){\n"
    "    uint i=uBase+gl_GlobalInvocationID.x;\n"
    "    if(i>=uColumns*uRows)return;\n"
    "    uint x=i%uColumns,y=i/uColumns,v=slot(x,y);\n"
    "    if(uMode==0u){\n"
    "        float px=vertexData[v],py=vertexData[v+1u],h=0.0;\n"
    "        for(uint w=0u;w<uWaves;++w){vec4 k=waves[w];h+=k.z*sin(k.x*px+k.y*py+k.w);}\n"
    "        vertexData[v+2u]=h;\n"
    "        return;\n"
    "    }\n"
    "    vec3 p=at(x,y),sum=vec3(0.0);\n"
    "    bool left=x>0u,right=x+1u<uColumns,down=y>0u,up=y+1u<uRows;\n"
    "    if(right&&up)sum+=corner(p,at(x+1u,y),at(x,y+1u));\n"
    "    if(left&&up){sum+=corner(p,at(x-1u,y+1u),at(x-1u,y));sum+=corner(p,at(x,y+1u),at(x-1u,y+1u));}\n"
    "    if(right&&down){sum+=corner(p,at(x,y-1u),at(x+1u,y-1u));sum+=corner(p,at(x+1u,y-1u),at(x+1u,y));}\n"
    "    if(left&&down)sum+=corner(p,at(x-1u,y),at(x,y-1u));\n"
    "    float l=length(sum);vec3 n=l>0.0?sum/l:vec3(0.0,0.0,1.0);\n"
    "    vertexData[v+3u]=n.x;vertexData[v+4u]=n.y;vertexData[v+5u]=n.z;\n"
    "}\n";

} // namespace arbitgpu
