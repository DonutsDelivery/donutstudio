#pragma once

#include "gl_loader.h"
#include "../../shared/ColorTransformContract.h"

#include <string>

namespace videorender
{
class ColorTransformGl final
{
public:
    bool initialize(arbitgl::GlFuncs* gl, std::string& error);
    bool render(unsigned sourceTexture, unsigned targetTexture,
                int width, int height,
                const colortransform::AdmittedTransform& transform,
                std::string& error);
    void shutdown() noexcept;
    bool ready() const noexcept { return gl_ != nullptr && program_ != 0 && vao_ != 0; }

private:
    arbitgl::GlFuncs* gl_ = nullptr;
    unsigned program_ = 0;
    unsigned vao_ = 0;
    unsigned framebuffer_ = 0;
};
} // namespace videorender
