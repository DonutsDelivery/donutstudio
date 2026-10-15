// LD_PRELOAD shim for ARBIT_TEST_GPU=1 in tools/private-test-session.sh. It
// moves every GLFW test onto GLFW's null platform with an EGL context, so the
// GL driver renders surfacelessly on the GPU and no window reaches Xvfb, X11 :0
// or wayland-0. Tests keep their own glfwInit and glfwCreateWindow calls; the
// shim adds the two hints those calls need and logs GL_RENDERER once. Tests that
// read the default framebuffer instead of an FBO have no surface to read here
// and fail rather than pass.
#define _GNU_SOURCE
#include <GLFW/glfw3.h>
#include <dlfcn.h>
#include <stdio.h>

int glfwInit (void)
{
    static int (*real) (void);
    if (real == NULL)
        real = (int (*) (void)) dlsym (RTLD_NEXT, "glfwInit");
    if (real == NULL)
    {
        fprintf (stderr, "glfw-headless-egl: glfwInit not found\n");
        return GLFW_FALSE;
    }
#if defined(GLFW_PLATFORM) && defined(GLFW_PLATFORM_NULL)
    glfwInitHint (GLFW_PLATFORM, GLFW_PLATFORM_NULL);
#endif
    return real();
}

GLFWwindow* glfwCreateWindow (int width, int height, const char* title,
                              GLFWmonitor* monitor, GLFWwindow* share)
{
    static GLFWwindow* (*real) (int, int, const char*, GLFWmonitor*, GLFWwindow*);
    if (real == NULL)
        real = (GLFWwindow* (*) (int, int, const char*, GLFWmonitor*, GLFWwindow*))
            dlsym (RTLD_NEXT, "glfwCreateWindow");
    if (real == NULL)
    {
        fprintf (stderr, "glfw-headless-egl: glfwCreateWindow not found\n");
        return NULL;
    }
    // Tests may reset hints with glfwDefaultWindowHints, so set this per window.
    glfwWindowHint (GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    return real (width, height, title, monitor, share);
}

// Logs the renderer once per process, so each test log shows which driver ran it.
void glfwMakeContextCurrent (GLFWwindow* window)
{
    static void (*real) (GLFWwindow*);
    static int reported;
    if (real == NULL)
        real = (void (*) (GLFWwindow*)) dlsym (RTLD_NEXT, "glfwMakeContextCurrent");
    if (real == NULL)
        return;
    real (window);
    if (window == NULL || reported)
        return;
    reported = 1;
    typedef const unsigned char* (*GetString) (unsigned int);
    GetString getString = (GetString) glfwGetProcAddress ("glGetString");
    const unsigned char* renderer = getString != NULL ? getString (0x1F01 /* GL_RENDERER */) : NULL;
    fprintf (stderr, "glfw-headless-egl: GL_RENDERER=%s\n",
             renderer != NULL ? (const char*) renderer : "(unknown)");
}
