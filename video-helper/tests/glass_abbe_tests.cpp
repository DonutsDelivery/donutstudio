#include "../../shared/SurfaceMaterialIR.h"

#include <cmath>
#include <cstdio>
#include <initializer_list>

// Abbe number glass: the Cauchy B that an IOR and Abbe number set, and the
// per-wavelength IOR spread it gives at the Fraunhofer F and C lines.
namespace {
using namespace surfacematerial;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

bool near(double a, double b, double tolerance) { return std::abs(a - b) <= tolerance; }

void abbe() {
  // BK7 crown: n_d 1.5168, V 64.17.
  const auto crown = cauchyBFromAbbe(1.5168, 64.17);
  check(crown > 0.0035 && crown < 0.0045, "crown glass B is near 0.004 square micrometres");
  check(near(abbeFromCauchyB(1.5168, crown), 64.17, 1e-9), "Abbe number round-trips through B");

  // A lower Abbe number disperses more for the same IOR.
  check(cauchyBFromAbbe(1.62, 36.0) > cauchyBFromAbbe(1.62, 60.0), "flint disperses more than crown");
  check(cauchyBFromAbbe(1.0, 40.0) == 0.0, "an IOR of 1 has no dispersion");
  check(near(cauchyBFromAbbe(1.5, 50.0), 0.5 * cauchyBPerIorExcess(50.0), 1e-12),
        "B scales with the IOR excess");

  // wavelengthIor at the F and C lines reproduces the Abbe number.
  for (const double v : {kMinimumAbbeNumber, 30.0, 64.17, kMaximumAbbeNumber}) {
    const float n = 1.5168f;
    const auto b = static_cast<float>(cauchyBFromAbbe(n, v));
    const double f = wavelengthIor(n, b, static_cast<float>(kAbbeFLine));
    const double c = wavelengthIor(n, b, static_cast<float>(kAbbeCLine));
    const double d = wavelengthIor(n, b, kGlassReferenceWavelength);
    check(near(d, n, 1e-6), "the d line keeps the authored IOR");
    check(f > d && d > c, "blue bends more than red");
    check(near((d - 1.0) / (f - c), v, v * 1e-3), "n_F - n_C gives back the Abbe number");
  }
}
} // namespace

int main() {
  abbe();
  if (failures != 0) return 1;
  std::puts("glass Abbe tests passed");
  return 0;
}
