#include "gpu_backend/metal_shader_source.h"

#include <fstream>
#include <iostream>
#include <iterator>

namespace
{
int failures = 0;

void check (bool condition, const char* message)
{
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool contains (const std::string& source, const std::string& text)
{
    return source.find (text) != std::string::npos;
}

std::string uniformBlock (const std::string& source)
{
    const auto begin = source.find ("uniform ArbitUniformBlock {");
    const auto end = source.find ("} arbitUniforms;", begin);
    if (begin == std::string::npos || end == std::string::npos) return {};
    return source.substr (begin, end - begin);
}

std::string sampler (int binding, const std::string& name)
{
    return "layout(set=0,binding=" + std::to_string (binding)
         + ") uniform sampler2D " + name + ";";
}

std::string translate (const std::string& raw)
{
    const auto wrap = arbitshader::wrapToContract (raw);
    check (! wrap.hasError(), "fixture passes the existing dialect wrapper");
    return arbitshader::metalGlsl (wrap);
}

void bareOperations()
{
    const std::string main = "void mainImage(out vec4 c, in vec2 p) {"
        " vec2 uv=p/uResolution; c=mix(texture(startImage,uv),texture(endImage,uv),progress); }";
    const std::string source = "uniform sampler2D startImage; uniform sampler2D endImage;"
        "uniform float progress;" + main;
    const auto wrap = arbitshader::wrapToContract (source);
    const auto originalGlsl = wrap.glsl;
    const auto metal = arbitshader::metalGlsl (wrap);
    check (wrap.params.empty() && wrap.glsl == originalGlsl,
           "bare operation inputs do not become admitted generator parameters or mutate the wrapper");
    check (metal.rfind ("#version 450\n", 0) == 0,
           "Metal source selects Vulkan GLSL");
    check (contains (metal, "layout(location=0) out vec4 fragColor;"),
           "Metal source retains the fragment output location");
    check (contains (metal, sampler (5, "startImage"))
           && contains (metal, sampler (6, "endImage")),
           "bare transition images get distinct bindings after the contract samplers");
    check (contains (uniformBlock (metal), "float progress;")
           && contains (metal, "#define progress arbitUniforms.progress\n")
           && ! contains (metal, "uniform float progress;"),
           "bare progress resides in the reflected value block");
    check (contains (metal, main)
           && contains (metal, "mainImage(c, gl_FragCoord.xy); fragColor = arbitGuard(c);"),
           "same-line declarations preserve the entire authored function and entry adapter");

    const std::string filterMain = "void main(){fragColor=texture(inputImage,vec2(0.5));}";
    const auto filter = translate ("uniform sampler2D inputImage; " + filterMain);
    check (contains (filter, sampler (5, "inputImage")) && contains (filter, filterMain),
           "bare filter image is bound without dropping its same-line entry point");
    check (contains (uniformBlock (filter), "float uTime;")
           && contains (uniformBlock (filter), "vec3 uCamPos;")
           && contains (filter, "layout(set=0,binding=1) uniform sampler1D uAudioBands;"),
           "clock, camera and audio contract adaptation is retained");
}

void declarationBoundaries()
{
    const auto metal = translate (
        "// uniform sampler2D inputImage;\n"
        "/* uniform float progress; */\n"
        "#define UNUSED uniform sampler2D inputImage;\\\n uniform float progress;\n"
        "#if 0\nuniform sampler2D endImage;\n#endif\n"
        "#/**/if/**/0\n{ uniform float progress;\n#endif\n"
        "uniform /* image inputs */ sampler2D\n startImage, endImage;"
        "uniform\tfloat /* amount */ progress;\n"
        "void main(){fragColor=mix(texture(startImage,vec2(0.5)),"
        "texture(endImage,vec2(0.5)),progress);}");
    check (contains (metal, sampler (5, "startImage"))
           && contains (metal, sampler (6, "endImage")),
           "comments, newlines, tabs and comma-separated image declarations are adapted");
    check (! contains (metal, sampler (5, "inputImage"))
           && ! contains (metal, sampler (7, "inputImage"))
           && contains (metal, "#define UNUSED uniform sampler2D inputImage;\\\n uniform float progress;\n")
           && contains (metal, "#if 0\nuniform sampler2D endImage;\n#endif"),
           "comments, continued directives and conditional declarations are not hoisted");
    const auto block = uniformBlock (metal);
    const auto progress = block.find ("float progress;");
    check (progress != std::string::npos
           && block.find ("float progress;", progress + 1) == std::string::npos,
           "commented and macro progress declarations do not create duplicate block members");

    for (const std::string declaration : {
            "uniform float unrelated;", "uniform vec2 progress;",
            "uniform sampler2D inputImage[2];", "uniform sampler2D startImage,;",
            "layout(binding=9) uniform sampler2D inputImage;" })
        check (contains (translate (declaration + "void main(){fragColor=vec4(1.0);}"), declaration),
               "unsupported or malformed declarations remain available for compiler diagnostics");
}

void generatedParameters()
{
    const auto wrap = arbitshader::wrapToContract (
        "/*{\"ISFVSN\":\"2\",\"INPUTS\":["
        "{\"NAME\":\"inputImage\",\"TYPE\":\"image\"},"
        "{\"NAME\":\"progress\",\"TYPE\":\"float\",\"DEFAULT\":0.25},"
        "{\"NAME\":\"tint\",\"TYPE\":\"color\",\"DEFAULT\":[1,0,0,1]}],"
        "\"PASSES\":[{\"TARGET\":\"trail\",\"PERSISTENT\":true},{}]}*/\n"
        "void main(){gl_FragColor=mix(IMG_THIS_PIXEL(inputImage),"
        "IMG_THIS_PIXEL(trail)*tint,progress)+vec4(float(PASSINDEX)*0.1);}");
    check (! wrap.hasError(), "ISF multipass fixture is accepted by the dialect wrapper");
    const auto metal = arbitshader::metalGlsl (wrap);
    check (contains (metal, sampler (5, "inputImage"))
           && contains (metal, sampler (6, "trail")),
           "ISF inputs and persistent pass targets retain separate bindings");
    check (contains (uniformBlock (metal), "float progress;")
           && contains (uniformBlock (metal), "vec4 tint;")
           && contains (uniformBlock (metal), "int PASSINDEX;"),
           "ISF scalar, vector and pass values remain reflected block members");
    check (wrap.params.size() == 3 && wrap.params[1].defaultScalar == 0.25
           && wrap.passes.size() == 2 && wrap.passes[0].persistent,
           "Metal adaptation retains ISF defaults and pass metadata");

    std::ifstream file (std::string (SHADER_CATALOG_SOURCE_ROOT)
                       + "/arbit-3d-raymarch/shaders/fractal_bass_temple.fs", std::ios::binary);
    const std::string source { std::istreambuf_iterator<char> (file), std::istreambuf_iterator<char>() };
    check (! source.empty(), "exact bundled Temple source is available");
    const auto temple = arbitshader::wrapToContract (source);
    check (! temple.hasError() && temple.params.size() == 1
           && temple.params[0].name == "sceneScale" && temple.params[0].defaultScalar == 1.0,
           "Temple exposes its existing catalog parameter through the wrapper");
    const auto declaration = temple.glsl.find ("uniform float sceneScale;");
    check (declaration != std::string::npos
           && declaration > temple.glsl.find ("// --- user shader ---"),
           "Temple reproduces the generated parameter after the user marker");
    const auto translated = arbitshader::metalGlsl (temple);
    check (contains (uniformBlock (translated), "float sceneScale;")
           && contains (translated, "#define sceneScale arbitUniforms.sceneScale\n")
           && ! contains (translated, "uniform float sceneScale;")
           && contains (translated, source),
           "Temple's generated uniform moves into the block while its exact authored source is preserved");
}

} // namespace

int main()
{
    bareOperations();
    declarationBoundaries();
    generatedParameters();
    if (failures != 0) return 1;
    std::cout << "Metal shader source translation passed; native compilation and pixels require macOS\n";
    return 0;
}
