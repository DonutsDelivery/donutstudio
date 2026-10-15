#include "../src/export_rgba_trace.h"

#include <cstdio>

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
}

int main()
{
    using namespace videohelper::exporttrace;
    check(validate(-1, {}, 24.0, 2.0, 1).empty(), "disabled trace is valid");
    check(validate(15, "/owned/export.mp4.frame15.rgba", 30.0, 0.0, 30).empty(), "frame 15 trace is valid");
    check(!validate(14, "/owned/export.mp4.frame15.rgba", 30.0, 0.0, 30).empty(), "wrong frame is rejected");
    check(!validate(15, {}, 30.0, 0.0, 30).empty(), "missing path is rejected");
    check(!validate(15, "/owned/export.mp4.frame15.rgba", 29.97, 0.0, 30).empty(), "wrong fps is rejected");
    check(!validate(15, "/owned/export.mp4.frame15.rgba", 30.0, 0.1, 30).empty(), "wrong start is rejected");
    check(!validate(15, "/owned/export.mp4.frame15.rgba", 30.0, 0.0, 15).empty(), "short range is rejected");
    check(!selected(15, 14, true), "previous frame is not selected");
    check(selected(15, 15, true), "owned frame 15 is selected");
    check(!selected(15, 15, false), "unowned frame is not selected");
    check(!selected(15, 16, true), "next frame is not selected");
    check(!selected(-1, 15, true), "disabled trace selects no frame");
    return failures == 0 ? 0 : 1;
}
