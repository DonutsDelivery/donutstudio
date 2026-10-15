// fd_socket.h — helper-owned unix domain socket for the zero-copy GPU
// surface channel (SharedGpuSurfaceProtocol.h). The stdio JSON-RPC channel
// cannot carry file descriptors; on Linux dmabuf fds ride here via
// SCM_RIGHTS. On macOS and Windows the buffer handles are plain uint64
// values inside the message payload (IOSurfaceID / DXGI shared HANDLE), so
// the socket only carries framed bytes.
//
// The HELPER listens (so a helper restart closes the socket and the Arbit
// side sees EOF -> drops all imported buffers); Arbit connects as the single
// client. Transport per platform:
//   Linux   — SOCK_SEQPACKET: one sendMsg() == one datagram == one
//             MsgHeader + payload (+ ancillary fds).
//   macOS   — SOCK_STREAM (Darwin has no unix SOCK_SEQPACKET); messages are
//             framed by MsgHeader.payloadBytes. fds unsupported.
//   Windows — AF_UNIX SOCK_STREAM via Winsock (Win10 1803+), framed like
//             macOS. fds unsupported.
//
// Single-threaded use from the viewport render loop: poll()/recvMsg() are
// non-blocking and called once per tick.
#pragma once

#if ARBIT_HAVE_DMABUF || ARBIT_HAVE_IOSURFACE || ARBIT_HAVE_D3D

#include <cstdint>
#include <string>
#include <vector>

class FdSocketServer
{
public:
    FdSocketServer() = default;
    ~FdSocketServer();
    FdSocketServer (const FdSocketServer&) = delete;
    FdSocketServer& operator= (const FdSocketServer&) = delete;

    // Creates + binds the socket at $XDG_RUNTIME_DIR/arbit-vp-<pid>-<rand>.sock
    // (mode 0600, /tmp fallback; %TEMP% on Windows) and starts listening.
    // Returns false with errorOut set on failure.
    bool listen (std::string& pathOut, std::string& errorOut);

    // Non-blocking accept. Returns true exactly once per NEW client
    // connection (caller then sends HELLO + BUFFERS). Only one client is
    // served at a time; extra connection attempts queue until dropClient().
    bool poll();

    bool hasClient() const { return clientFd_ >= 0; }

    // Sends one framed message (gpusurf::MsgHeader + payload) with optional
    // SCM_RIGHTS fds (Linux only — non-empty fds fail the call elsewhere).
    // Returns false on send failure (client likely gone — caller should
    // treat as disconnect). No SIGPIPE.
    bool sendMsg (uint32_t type, const void* payload, size_t payloadBytes,
                  const int* fds = nullptr, int fdCount = 0);

    enum class RecvResult { NoData, Message, Disconnected };
    struct Received
    {
        uint32_t type = 0;
        std::vector<uint8_t> payload;
        std::vector<int> fds; // ownership transfers to caller (close them); always empty off-Linux
    };
    // Non-blocking receive of one message. Disconnected => client closed;
    // the client fd is dropped internally (poll() may re-accept later).
    RecvResult recvMsg (Received& out);

    void dropClient();
    void close(); // also unlinks the socket path

private:
    // Socket handles as signed ints: POSIX fds directly; Windows SOCKETs
    // cast through intptr_t (kernel handle values fit — documented WinSock
    // behavior) with -1 as the invalid sentinel on every platform.
    intptr_t listenFd_ = -1;
    intptr_t clientFd_ = -1;
    std::string path_;
#if ! defined (__linux__)
    std::vector<uint8_t> rxBuf_; // stream reassembly buffer (mac/win)
#endif
};

#endif // ARBIT_HAVE_DMABUF || ARBIT_HAVE_IOSURFACE || ARBIT_HAVE_D3D
