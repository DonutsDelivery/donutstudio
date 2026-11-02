#pragma once

#include <string>

namespace arbitgpu::smoothglass
{
// What a V1 walk reports when its frame can't finish. The receipts are the first place
// this is known: it depends on what each ray hits, so nothing decides it before tracing.
inline constexpr const char* kRoughUnresolvedError = "Rough glass requires microfacet multiple scattering beyond its unresolved-power bound";
// Decides one frame's second try. A frame V1 finishes never gets here, so it stays V1. A
// renderer whose second try hits the same wall has no V2 to offer, so it stops paying for
// the try and the frame reports what it always did.
class RoughV1Fallback final
{
public:
    bool shouldTry(const std::string& error) const noexcept {return !unavailable_ && error.find(kRoughUnresolvedError)!=std::string::npos;}
    void secondTryFailed(const std::string& error) noexcept {if(error.find(kRoughUnresolvedError)!=std::string::npos)unavailable_=true;}
private:
    bool unavailable_=false;
};
}
