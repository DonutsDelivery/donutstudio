#include "gpu_export_vk.h"

#if ARBIT_HAVE_VKFD

#include "SharedGpuSurfaceProtocol.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <vector>

// ---- GL types/enums/entry points for GL_EXT_memory_object(_fd) ----
typedef void (*PFNGLCREATEMEMORYOBJECTSEXT) (int, unsigned*);
typedef void (*PFNGLDELETEMEMORYOBJECTSEXT) (int, const unsigned*);
typedef void (*PFNGLMEMORYOBJECTPARAMETERIVEXT) (unsigned, unsigned, const int*);
typedef void (*PFNGLIMPORTMEMORYFDEXT) (unsigned, uint64_t, unsigned, int);
typedef void (*PFNGLTEXSTORAGEMEM2DEXT) (unsigned, int, unsigned, int, int, unsigned, uint64_t);
typedef void (*PFNGLBLITFRAMEBUFFER) (int, int, int, int, int, int, int, int, unsigned, unsigned);

#ifndef GL_HANDLE_TYPE_OPAQUE_FD_EXT
#define GL_HANDLE_TYPE_OPAQUE_FD_EXT 0x9586
#endif
#ifndef GL_DEDICATED_MEMORY_OBJECT_EXT
#define GL_DEDICATED_MEMORY_OBJECT_EXT 0x9581
#endif
#ifndef GL_TEXTURE_TILING_EXT
#define GL_TEXTURE_TILING_EXT 0x9580
#endif
#ifndef GL_OPTIMAL_TILING_EXT
#define GL_OPTIMAL_TILING_EXT 0x9584
#endif

namespace gpuexp
{

namespace
{
VkInstance        asInst (void* p)  { return (VkInstance) p; }
VkPhysicalDevice  asPhys (void* p)  { return (VkPhysicalDevice) p; }
VkDevice          asDev  (void* p)  { return (VkDevice) p; }
} // namespace

bool VkFdExporter::initialize (const arbitgl::GlFuncs* gl, std::string& errorOut)
{
    gl_ = gl;
    ready_ = false;

    // --- GL_EXT_memory_object_fd entry points (core GL, GLX or EGL) ---
    glCreateMemoryObjects_     = (void*) glfwGetProcAddress ("glCreateMemoryObjectsEXT");
    glDeleteMemoryObjects_     = (void*) glfwGetProcAddress ("glDeleteMemoryObjectsEXT");
    glMemoryObjectParameteriv_ = (void*) glfwGetProcAddress ("glMemoryObjectParameterivEXT");
    glImportMemoryFd_          = (void*) glfwGetProcAddress ("glImportMemoryFdEXT");
    glTexStorageMem2D_         = (void*) glfwGetProcAddress ("glTexStorageMem2DEXT");
    blitFramebuffer_           = (void*) glfwGetProcAddress ("glBlitFramebuffer");
    if (glCreateMemoryObjects_ == nullptr || glImportMemoryFd_ == nullptr
        || glTexStorageMem2D_ == nullptr || glMemoryObjectParameteriv_ == nullptr
        || glDeleteMemoryObjects_ == nullptr)
    {
        errorOut = "GL lacks GL_EXT_memory_object_fd entry points";
        return false;
    }
    if (blitFramebuffer_ == nullptr)
    {
        errorOut = "glBlitFramebuffer unavailable";
        return false;
    }

    // --- minimal Vulkan instance + device on the discrete GPU ---
    VkApplicationInfo ai { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &ai;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance (&ici, nullptr, &inst) != VK_SUCCESS)
    {
        errorOut = "vkCreateInstance failed (no Vulkan loader/driver)";
        return false;
    }
    vkInstance_ = inst;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices (inst, &n, nullptr);
    if (n == 0) { errorOut = "no Vulkan physical devices"; shutdown(); return false; }
    std::vector<VkPhysicalDevice> devs (n);
    vkEnumeratePhysicalDevices (inst, &n, devs.data());
    VkPhysicalDevice phys = devs[0];
    for (auto d : devs)
    {
        VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties (d, &pr);
        if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { phys = d; break; }
    }
    vkPhysical_ = phys;

    // Device UUID (multi-GPU demotion guard on the consumer).
    {
        VkPhysicalDeviceIDProperties idp { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
        VkPhysicalDeviceProperties2 p2 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        p2.pNext = &idp;
        vkGetPhysicalDeviceProperties2 (phys, &p2);
        char hex[VK_UUID_SIZE * 2 + 1];
        for (int i = 0; i < (int) VK_UUID_SIZE; ++i)
            std::snprintf (hex + i * 2, 3, "%02x", idp.deviceUUID[i]);
        deviceUuid_ = hex;
    }

    float pri = 1.0f;
    VkDeviceQueueCreateInfo q { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    q.queueFamilyIndex = 0; q.queueCount = 1; q.pQueuePriorities = &pri;
    const char* exts[] = { "VK_KHR_external_memory_fd" };
    VkDeviceCreateInfo dci { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &q;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = exts;
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice (phys, &dci, nullptr, &dev) != VK_SUCCESS)
    {
        errorOut = "vkCreateDevice failed (VK_KHR_external_memory_fd unsupported?)";
        shutdown();
        return false;
    }
    vkDevice_ = dev;

    vkGetMemoryFd_ = (void*) vkGetDeviceProcAddr (dev, "vkGetMemoryFdKHR");
    if (vkGetMemoryFd_ == nullptr)
    {
        errorOut = "vkGetMemoryFdKHR unavailable";
        shutdown();
        return false;
    }

    if (readFbo_ == 0)
        gl_->GenFramebuffers (1, &readFbo_);

    ready_ = true;
    return true;
}

bool VkFdExporter::allocate (ExportedBuffer& b, int width, int height, std::string& errorOut)
{
    if (! ready_) { errorOut = "exporter not initialized"; return false; }

    b = ExportedBuffer {};
    b.width = width;
    b.height = height;
    b.fourcc = gpusurf::kHandleVkFd;
    b.planeCount = 1;

    VkDevice dev = asDev (vkDevice_);

    // --- VkImage (RGBA8 OPTIMAL, external OPAQUE_FD) ---
    VkExternalMemoryImageCreateInfo ext { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkImageCreateInfo ic { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ic.pNext = &ext;
    ic.imageType = VK_IMAGE_TYPE_2D;
    ic.format = VK_FORMAT_R8G8B8A8_UNORM;
    ic.extent = { (uint32_t) width, (uint32_t) height, 1 };
    ic.mipLevels = 1; ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
             | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ic.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage img = VK_NULL_HANDLE;
    if (vkCreateImage (dev, &ic, nullptr, &img) != VK_SUCCESS)
    { errorOut = "vkCreateImage failed"; destroy (b); return false; }
    b.vkImage = (uint64_t) img;

    VkMemoryRequirements mr; vkGetImageMemoryRequirements (dev, img, &mr);
    if (memTypeIndex_ == 0xffffffffu)
    {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties (asPhys (vkPhysical_), &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mr.memoryTypeBits & (1u << i))
                && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            { memTypeIndex_ = i; break; }
        if (memTypeIndex_ == 0xffffffffu)
        { errorOut = "no device-local memory type"; destroy (b); return false; }
    }

    VkExportMemoryAllocateInfo emi { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO };
    emi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryDedicatedAllocateInfo ded { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    ded.image = img; ded.pNext = &emi;
    VkMemoryAllocateInfo mai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.pNext = &ded; mai.allocationSize = mr.size; mai.memoryTypeIndex = memTypeIndex_;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory (dev, &mai, nullptr, &mem) != VK_SUCCESS)
    { errorOut = "vkAllocateMemory failed"; destroy (b); return false; }
    b.vkMemory = (uint64_t) mem;
    if (vkBindImageMemory (dev, img, mem, 0) != VK_SUCCESS)
    { errorOut = "vkBindImageMemory failed"; destroy (b); return false; }
    b.modifier = (uint64_t) mr.size;     // allocation size for glImportMemoryFdEXT
    b.offsets[0] = 0;

    // Export two fds for the same memory: one to import here, one for the consumer.
    auto getFd = [&] (int& outFd) -> bool
    {
        VkMemoryGetFdInfoKHR gi { VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
        gi.memory = mem; gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        return ((PFN_vkGetMemoryFdKHR) vkGetMemoryFd_) (dev, &gi, &outFd) == VK_SUCCESS && outFd >= 0;
    };
    int localFd = -1;
    if (! getFd (localFd) || ! getFd (b.fds[0]))
    { errorOut = "vkGetMemoryFdKHR failed"; if (localFd >= 0) ::close (localFd); destroy (b); return false; }

    // --- import the memory into THIS GL context as a renderable texture ---
    const auto glCreateMemoryObjects = (PFNGLCREATEMEMORYOBJECTSEXT) glCreateMemoryObjects_;
    const auto glMemoryObjectParameteriv = (PFNGLMEMORYOBJECTPARAMETERIVEXT) glMemoryObjectParameteriv_;
    const auto glImportMemoryFd = (PFNGLIMPORTMEMORYFDEXT) glImportMemoryFd_;
    const auto glTexStorageMem2D = (PFNGLTEXSTORAGEMEM2DEXT) glTexStorageMem2D_;

    glCreateMemoryObjects (1, &b.glMemObject);
    const int dedicated = 1;
    glMemoryObjectParameteriv (b.glMemObject, GL_DEDICATED_MEMORY_OBJECT_EXT, &dedicated);
    glImportMemoryFd (b.glMemObject, (uint64_t) mr.size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, localFd); // consumes localFd

    glGenTextures (1, &b.tex);
    glBindTexture (GL_TEXTURE_2D, b.tex);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_TILING_EXT, GL_OPTIMAL_TILING_EXT);
    glTexStorageMem2D (GL_TEXTURE_2D, 1, GL_RGBA8, width, height, b.glMemObject, 0);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    gl_->GenFramebuffers (1, &b.fbo);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, b.fbo);
    gl_->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, b.tex, 0);
    if (gl_->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        errorOut = "vkfd export buffer FBO incomplete";
        gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
        destroy (b);
        return false;
    }
    glViewport (0, 0, width, height);
    glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
    glClear (GL_COLOR_BUFFER_BIT);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    glFinish();
    return true;
}

void VkFdExporter::destroy (ExportedBuffer& b)
{
    if (b.fbo != 0 && gl_ != nullptr) gl_->DeleteFramebuffers (1, &b.fbo);
    if (b.tex != 0) glDeleteTextures (1, &b.tex);
    if (b.glMemObject != 0 && glDeleteMemoryObjects_ != nullptr)
        ((PFNGLDELETEMEMORYOBJECTSEXT) glDeleteMemoryObjects_) (1, &b.glMemObject);
    if (vkDevice_ != nullptr)
    {
        VkDevice dev = asDev (vkDevice_);
        if (b.vkImage != 0)  vkDestroyImage (dev, (VkImage) b.vkImage, nullptr);
        if (b.vkMemory != 0) vkFreeMemory (dev, (VkDeviceMemory) b.vkMemory, nullptr);
    }
    for (int& fd : b.fds)
        if (fd >= 0) { ::close (fd); fd = -1; }
    b = ExportedBuffer {};
}

bool VkFdExporter::blit (unsigned srcTexture, const ExportedBuffer& b)
{
    if (! ready_ || b.fbo == 0)
        return false;
    gl_->BindFramebuffer (GL_READ_FRAMEBUFFER, readFbo_);
    gl_->FramebufferTexture2D (GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, srcTexture, 0);
    gl_->BindFramebuffer (GL_DRAW_FRAMEBUFFER, b.fbo);
    ((PFNGLBLITFRAMEBUFFER) blitFramebuffer_) (
        0, 0, b.width, b.height, 0, 0, b.width, b.height,
        GL_COLOR_BUFFER_BIT, GL_NEAREST);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    return true;
}

void VkFdExporter::shutdown()
{
    if (readFbo_ != 0 && gl_ != nullptr)
    {
        gl_->DeleteFramebuffers (1, &readFbo_);
        readFbo_ = 0;
    }
    if (vkDevice_ != nullptr)   { vkDestroyDevice (asDev (vkDevice_), nullptr); vkDevice_ = nullptr; }
    if (vkInstance_ != nullptr) { vkDestroyInstance (asInst (vkInstance_), nullptr); vkInstance_ = nullptr; }
    vkPhysical_ = nullptr;
    vkGetMemoryFd_ = nullptr;
    memTypeIndex_ = 0xffffffffu;
    ready_ = false;
}

} // namespace gpuexp

#endif // ARBIT_HAVE_VKFD
