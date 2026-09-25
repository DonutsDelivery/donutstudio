#pragma once

#include "../shader_dialect.h"

#include <sstream>
#include <string_view>

namespace arbitshader
{
namespace detail
{

struct MetalSourceToken
{
    std::size_t begin = 0, end = 0;
};

// Keep offsets into the original text so adapting a declaration never consumes
// a neighbouring declaration or function on the same line.
inline std::vector<MetalSourceToken> metalSourceTokens (const std::string& source)
{
    std::vector<MetalSourceToken> tokens;
    for (std::size_t i = 0; i < source.size();)
    {
        if (std::isspace (static_cast<unsigned char> (source[i]))) { ++i; continue; }
        if (source.compare (i, 2, "//") == 0)
        { while (i < source.size() && source[i] != '\n') ++i; continue; }
        if (source.compare (i, 2, "/*") == 0)
        {
            const auto end = source.find ("*/", i + 2);
            i = end == std::string::npos ? source.size() : end + 2;
            continue;
        }
        const auto begin = i++;
        if (source[begin] == '#')
        {
            // A directive, including its continued lines, is one opaque token.
            while (i < source.size())
            {
                if (source[i++] != '\n') continue;
                auto last = i - 1;
                if (last > begin && source[last - 1] == '\r') --last;
                if (last == begin || source[last - 1] != '\\') break;
            }
        }
        else if (isIdentChar (source[begin]))
            while (i < source.size() && isIdentChar (source[i])) ++i;
        tokens.push_back ({ begin, i });
    }
    return tokens;
}

inline bool metalOperationUniform (std::string_view type, std::string_view name,
                                   const WrapResult& wrap)
{
    if (type == "sampler2D"
        && (name == "inputImage" || name == "startImage" || name == "endImage"))
        return true;
    if (type == "float" && name == "progress") return true;
    for (const auto& param : wrap.params)
        if (name == param.name && type == inputTypeToGlsl (param.type)) return true;
    return false;
}

} // namespace detail

// Vulkan GLSL needs a block for value uniforms and bindings for samplers. Adapt
// the generated contract plus the operation's plain global uniforms. Source
// admission and parameter discovery continue to use the unmodified WrapResult.
inline std::string metalGlsl (const WrapResult& wrap)
{
    std::string source = wrap.glsl;
    if (source.rfind ("#version ", 0) != 0) return source;
    source.replace (0, source.find ('\n'), "#version 450");
    const auto userStart = source.find ("// --- user shader ---\n");
    const std::string contractMarker = "// --- Arbit uniform contract (auto-prepended) ---\n";
    if (userStart == std::string::npos || source.find (contractMarker) == std::string::npos)
        return source;

    struct Uniform { std::string type, name; };
    std::vector<Uniform> values;
    const auto tokens = detail::metalSourceTokens (source);
    const auto text = [&] (std::size_t index) -> std::string_view
    {
        if (index >= tokens.size()) return {};
        const auto& token = tokens[index];
        return std::string_view (source).substr (token.begin, token.end - token.begin);
    };
    std::string body;
    std::size_t copied = 0;
    const auto replace = [&] (std::size_t begin, std::size_t end, const std::string& replacement)
    {
        body.append (source, copied, begin - copied);
        body += replacement;
        copied = end;
    };
    int samplerBinding = 1, braces = 0, parentheses = 0, conditionalDepth = 0;
    bool declarationStart = true;
    for (std::size_t i = 0; i < tokens.size(); ++i)
    {
        const auto token = text (i);
        if (token.front() == '#')
        {
            const std::string directive (token.substr (1));
            const auto words = detail::metalSourceTokens (directive);
            const auto name = words.empty() ? std::string_view()
                : std::string_view (directive).substr (words[0].begin, words[0].end - words[0].begin);
            if (name == "if" || name == "ifdef" || name == "ifndef") ++conditionalDepth;
            else if (name == "endif" && conditionalDepth > 0) --conditionalDepth;
            continue;
        }
        // Conditional source belongs to the GLSL preprocessor. In particular,
        // disabled branches may contain unmatched braces or unused uniforms.
        if (conditionalDepth > 0) continue;
        const bool prelude = tokens[i].begin < userStart;
        if (declarationStart && braces == 0 && parentheses == 0)
        {
            if (prelude && token == "out" && text (i + 1) == "vec4"
                && text (i + 2) == "fragColor" && text (i + 3) == ";")
            {
                replace (tokens[i].begin, tokens[i + 3].end,
                         "layout(location=0) out vec4 fragColor;");
                i += 3;
                continue;
            }
            if (token == "uniform")
            {
                const auto type = text (i + 1);
                std::vector<std::string> names;
                auto end = i + 2;
                bool needsName = true;
                while (end < tokens.size()
                       && detail::isValidGlslIdentifier (std::string (text (end))))
                {
                    names.emplace_back (text (end++));
                    needsName = false;
                    if (text (end) != ",") break;
                    ++end;
                    needsName = true;
                }
                bool supported = ! needsName && text (end) == ";";
                for (const auto& name : names)
                    supported &= prelude || detail::metalOperationUniform (type, name, wrap);
                if (supported)
                {
                    std::string replacement;
                    for (const auto& name : names)
                    {
                        if (type.substr (0, 7) == "sampler")
                            replacement += "layout(set=0,binding=" + std::to_string (samplerBinding++)
                                         + ") uniform " + std::string (type) + " " + name + ";";
                        else
                            values.push_back ({ std::string (type), name });
                    }
                    // Retain newlines for diagnostics when value declarations
                    // move into the block. All other source spans stay intact.
                    for (auto offset = tokens[i].begin; offset < tokens[end].end; ++offset)
                        if (source[offset] == '\n') replacement += '\n';
                    replace (tokens[i].begin, tokens[end].end, replacement);
                    i = end;
                    continue;
                }
            }
        }
        if (token == "{") ++braces;
        else if (token == "}") --braces;
        else if (token == "(") ++parentheses;
        else if (token == ")") --parentheses;
        declarationStart = braces == 0 && parentheses == 0 && (token == ";" || token == "}");
    }
    body.append (source, copied, std::string::npos);

    std::ostringstream block;
    block << "layout(std140,set=0,binding=0) uniform ArbitUniformBlock {\n";
    for (const auto& uniform : values)
        block << "    " << uniform.type << ' ' << uniform.name << ";\n";
    block << "} arbitUniforms;\n";
    for (const auto& uniform : values)
        block << "#define " << uniform.name << " arbitUniforms." << uniform.name << '\n';
    body.insert (body.find (contractMarker) + contractMarker.size(), block.str());
    return body;
}

} // namespace arbitshader
