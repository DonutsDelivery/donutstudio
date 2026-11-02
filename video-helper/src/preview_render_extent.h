#pragma once

// Size at which the live viewport renders an imported 3D scene. The docked panel
// shows the canvas at a few hundred pixels, so rendering the scene at a 4K canvas
// spends most of its time on pixels nobody sees. Export and the composite probe
// never use this: they keep the full canvas.
//
// Plain integers only, so the rule can be tested without a GL context.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace videohelper::previewextent
{
struct Extent
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    bool valid() const { return width != 0 && height != 0; }
    bool operator== (const Extent& other) const { return width == other.width && height == other.height; }
    bool operator!= (const Extent& other) const { return !(*this == other); }
};

// Shortest side the preview scene is rendered at, unless the canvas is smaller.
inline constexpr std::uint32_t kFloorShortSide = 64;

// The zoom the extent follows, as a whole power of two from 1 to 32. Zooming in
// past the fitted view needs more scene pixels; zooming out never asks for fewer.
// Steps keep a pinch or wheel zoom from re-rendering the scene on every tick.
inline std::uint32_t zoomBucket (double zoom)
{
    if (!std::isfinite (zoom) || zoom <= 1.0) return 1;
    std::uint32_t bucket = 1;
    while (bucket < 32 && static_cast<double> (bucket) < zoom) bucket *= 2;
    return bucket;
}

// The canvas aspect fitted inside the presentation framebuffer, scaled by the zoom
// bucket, rounded to whole pixels. Never larger than the canvas, never below
// kFloorShortSide on the short side (a canvas already smaller than that stays
// as it is). A canvas of zero, or an unknown framebuffer, gives the canvas back.
inline Extent fitToPresentation (int canvasWidth, int canvasHeight,
                                 int presentWidth, int presentHeight,
                                 std::uint32_t zoom = 1)
{
    if (canvasWidth <= 0 || canvasHeight <= 0) return {};
    const auto cw = static_cast<std::uint64_t> (canvasWidth);
    const auto ch = static_cast<std::uint64_t> (canvasHeight);
    const Extent canvas { static_cast<std::uint32_t> (cw), static_cast<std::uint32_t> (ch) };
    if (presentWidth <= 0 || presentHeight <= 0) return canvas;

    const std::uint64_t z = zoom == 0 ? 1 : zoom;
    const std::uint64_t pw = static_cast<std::uint64_t> (presentWidth) * z;
    const std::uint64_t ph = static_cast<std::uint64_t> (presentHeight) * z;

    // The framebuffer side that runs out first limits the fit.
    std::uint64_t w = 0, h = 0;
    if (pw * ch <= ph * cw) { w = pw; h = (ch * pw + cw / 2) / cw; }
    else                    { h = ph; w = (cw * ph + ch / 2) / ch; }

    const std::uint64_t floorSide = kFloorShortSide;
    if (std::min (cw, ch) <= floorSide) return canvas;
    if (std::min (w, h) < floorSide)
    {
        if (cw <= ch) { w = floorSide; h = (ch * floorSide + cw / 2) / cw; }
        else          { h = floorSide; w = (cw * floorSide + ch / 2) / ch; }
    }
    if (w >= cw || h >= ch) return canvas;
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    return { static_cast<std::uint32_t> (w), static_cast<std::uint32_t> (h) };
}

// The extent the viewport renders the scene at. A caller that asked for the full
// canvas (viewport_set_full_canvas_scene) gets it at any panel size or zoom. Otherwise
// the scene follows the panel.
inline Extent sceneExtent (int canvasWidth, int canvasHeight,
                           int presentWidth, int presentHeight,
                           std::uint32_t zoom, bool fullCanvas)
{
    if (fullCanvas && canvasWidth > 0 && canvasHeight > 0)
        return { static_cast<std::uint32_t> (canvasWidth), static_cast<std::uint32_t> (canvasHeight) };
    return fitToPresentation (canvasWidth, canvasHeight, presentWidth, presentHeight, zoom);
}
} // namespace videohelper::previewextent
