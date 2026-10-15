// rife implemented with ncnn library

#ifndef RIFE_H
#define RIFE_H

#include <string>

// ncnn
#include "net.h"

class RIFE
{
public:
    RIFE(int gpuid, bool tta_mode = false, bool tta_temporal_mode = false, bool uhd_mode = false, int num_threads = 1, bool rife_v2 = false, bool rife_v4 = false);
    ~RIFE();

#if _WIN32
    int load(const std::wstring& modeldir);
#else
    int load(const std::string& modeldir);
#endif

    int process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

    int process_cpu(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

    int process_v4(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

    int process_v4_cpu(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

#ifdef ARBIT_RIFE_ZEROCOPY
    // Phase-1b zero-copy GPU output. Runs process_v4 but, instead of downloading
    // out_gpu to CPU, copies it (GPU->GPU, on RIFE's own VkDevice/queue) into the
    // caller-owned, OPAQUE_FD-exportable VkBuffer `exportBuf` — no CPU round-trip of
    // the output. With ncnn's int8 storage (forced on), out_gpu is interleaved RGB8,
    // so exportBuf holds tightly-packed RGB8 (w*h*3 bytes). exportCapacity = buffer
    // byte size (>= w*h*3). Returns 0 and sets outW/outH on success.
    // NOTE: timestep 0 and 1 are NOT handled here (process_v4 early-returns the
    // source frame without touching exportBuf); the caller handles those trivially.
    int process_v4_to_buffer(const ncnn::Mat& in0image, const ncnn::Mat& in1image,
                             float timestep, VkBuffer exportBuf, size_t exportCapacity,
                             int& outW, int& outH) const;
    ncnn::VulkanDevice* vulkanDevice() const { return vkdev; }
#endif

private:
    ncnn::VulkanDevice* vkdev;
    ncnn::Net flownet;
    ncnn::Net contextnet;
    ncnn::Net fusionnet;
    ncnn::Pipeline* rife_preproc;
    ncnn::Pipeline* rife_postproc;
    ncnn::Pipeline* rife_flow_tta_avg;
    ncnn::Pipeline* rife_flow_tta_temporal_avg;
    ncnn::Pipeline* rife_out_tta_temporal_avg;
    ncnn::Pipeline* rife_v4_timestep;
    ncnn::Layer* rife_uhd_downscale_image;
    ncnn::Layer* rife_uhd_upscale_flow;
    ncnn::Layer* rife_uhd_double_flow;
    ncnn::Layer* rife_v2_slice_flow;
    bool tta_mode;
    bool tta_temporal_mode;
    bool uhd_mode;
    int num_threads;
    bool rife_v2;
    bool rife_v4;

#ifdef ARBIT_RIFE_ZEROCOPY
    // Phase-1b zero-copy sink, set by process_v4_to_buffer for the duration of one
    // process_v4 call (mutable: process_v4 is const). VK_NULL_HANDLE => normal CPU
    // download path (unchanged).
    mutable VkBuffer m_zcExportBuf = VK_NULL_HANDLE;
    mutable size_t   m_zcExportCap = 0;
    mutable int      m_zcOutW = 0;
    mutable int      m_zcOutH = 0;
#endif
};

#endif // RIFE_H
