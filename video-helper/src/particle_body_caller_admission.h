#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <mutex>
#include <thread>
#include <limits>
#include <string>
#include <new>

namespace videorender {

// Logical caller objects and conservative byte-slot copy visits plus source
// comparison element visits only. Geometry
// source/sampler and native resource admission remain separate. STL capacities,
// control blocks, callbacks, compiler, allocator and driver internals are unknown;
// these limits do not bound elapsed time or all process memory.
struct ParticleBodyCallerLimits final
{
    std::size_t maxLogicalBytes=256u*1024u*1024u;
    std::size_t maxWork=1000000000000ULL;
};
struct ParticleBodyCallerUsage final
{
    std::size_t logicalBytes=0, peakLogicalBytes=0, work=0;
};
class ParticleBodyCallerLedger;
class ParticleBodyCallerCeiling;
class ParticleBodyCallerLease final
{
public:
    ParticleBodyCallerLease(const ParticleBodyCallerLease&)=delete;
    ParticleBodyCallerLease& operator=(const ParticleBodyCallerLease&)=delete;
    ~ParticleBodyCallerLease();
    const std::shared_ptr<ParticleBodyCallerLedger> ledger;
    const std::size_t bytes;
private:
    ParticleBodyCallerLease(std::shared_ptr<ParticleBodyCallerLedger> l,std::size_t b)
        : ledger(std::move(l)),bytes(b) {}
    friend class ParticleBodyCallerLedger;
};
class ParticleBodyCallerLedger final : public std::enable_shared_from_this<ParticleBodyCallerLedger>
{
private:
    friend std::shared_ptr<ParticleBodyCallerLedger> makeParticleBodyCallerLedger(ParticleBodyCallerLimits,std::string&);
    explicit ParticleBodyCallerLedger(ParticleBodyCallerLimits limits) : limits_(limits),thread_(std::this_thread::get_id())
    { usage_.logicalBytes=usage_.peakLogicalBytes=sizeof(ParticleBodyCallerLedger); }
public:
    ParticleBodyCallerUsage usage() const
    { std::lock_guard<std::mutex> lock(mutex_); return usage_; }
    bool onCallerThread() const noexcept { return thread_==std::this_thread::get_id(); }
    bool charge(std::size_t work)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return chargeLocked(work);
    }
    std::shared_ptr<const ParticleBodyCallerLease> reserve(std::size_t payloadBytes,std::size_t work)
    {
        std::size_t bytes=0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Even a storage refusal retains the admitted attempted work. A work
            // shortage exhausts the epoch, so retry cannot recover failed work.
            if (!chargeLocked(work) || payloadBytes>std::numeric_limits<std::size_t>::max()-sizeof(ParticleBodyCallerLease)) return {};
            bytes=payloadBytes+sizeof(ParticleBodyCallerLease);
            const auto limit=std::min(limits_.maxLogicalBytes,scopedLogicalLimit_);
            if (usage_.logicalBytes>limit || bytes>limit-usage_.logicalBytes) return {};
            usage_.logicalBytes+=bytes;
            usage_.peakLogicalBytes=std::max(usage_.peakLogicalBytes,usage_.logicalBytes);
        }
        ParticleBodyCallerLease* token=nullptr;
        try { token=new ParticleBodyCallerLease(shared_from_this(),bytes); }
        catch (const std::bad_alloc&) { release(bytes); return {}; }
        // shared_ptr deletes token if its control-block allocation fails. That
        // destructor refunds once; only failed token allocation refunds here.
        try { return std::shared_ptr<const ParticleBodyCallerLease>(token); }
        catch (const std::bad_alloc&) { return {}; }
    }
private:
    bool chargeLocked(std::size_t work)
    {
        if (!onCallerThread()) return false;
        if (work>limits_.maxWork-usage_.work) { usage_.work=limits_.maxWork; return false; }
        usage_.work+=work; return true;
    }
    void release(std::size_t bytes)
    { std::lock_guard<std::mutex> lock(mutex_); usage_.logicalBytes-=bytes; }
    const ParticleBodyCallerLimits limits_;
    const std::thread::id thread_;
    mutable std::mutex mutex_;
    ParticleBodyCallerUsage usage_;
    std::size_t scopedLogicalLimit_=std::numeric_limits<std::size_t>::max();
    friend class ParticleBodyCallerLease;
    friend class ParticleBodyCallerCeiling;
};
// A synchronous native attempt can only lower this caller's existing limit.
// Nested guards restore the previous ceiling. Escaped leases retain their debit;
// leaving the scope neither retires storage nor restores attempted work.
class ParticleBodyCallerCeiling final
{
public:
    ParticleBodyCallerCeiling(const std::shared_ptr<ParticleBodyCallerLedger>& caller,std::size_t limit)
        : caller_(caller)
    {
        if (!caller_ || !caller_->onCallerThread()) return;
        std::lock_guard<std::mutex> lock(caller_->mutex_);
        previous_=caller_->scopedLogicalLimit_;
        caller_->scopedLogicalLimit_=std::min(previous_,limit); active_=true;
    }
    ParticleBodyCallerCeiling(const ParticleBodyCallerCeiling&)=delete;
    ParticleBodyCallerCeiling& operator=(const ParticleBodyCallerCeiling&)=delete;
    ~ParticleBodyCallerCeiling()
    {
        if (active_) {
            std::lock_guard<std::mutex> lock(caller_->mutex_);
            caller_->scopedLogicalLimit_=previous_;
        }
    }
private:
    const std::shared_ptr<ParticleBodyCallerLedger> caller_;
    std::size_t previous_=0;
    bool active_=false;
};
inline ParticleBodyCallerLease::~ParticleBodyCallerLease() { ledger->release(bytes); }

inline std::shared_ptr<ParticleBodyCallerLedger> makeParticleBodyCallerLedger(
    ParticleBodyCallerLimits limits,std::string& error)
{
    if (limits.maxLogicalBytes<sizeof(ParticleBodyCallerLedger) || limits.maxWork==0) {
        error="Body caller ledger fixed metadata admission is short"; return {};
    }
    try { return std::shared_ptr<ParticleBodyCallerLedger>(new ParticleBodyCallerLedger(limits)); }
    catch (const std::bad_alloc&) { error="Body caller ledger allocation failed"; return {}; }
}

// One stable work epoch per actual executing render thread. Source, plan,
// snapshot, seek and request replacements never reset it. Thread retirement
// ends the epoch; escaped payloads keep the ledger and storage leases alive.
// No global or cross-thread work allowance is claimed. Destruction may refund
// storage on another thread, but replay/validation must use the original thread.
inline std::shared_ptr<ParticleBodyCallerLedger> particleBodyRenderThreadCaller(std::string& error)
{
    thread_local const auto ledger=makeParticleBodyCallerLedger({},error);
    return ledger;
}
// Declare the lease before its payload: refund follows payload destruction.
template <typename T> struct ParticleBodyCallerPayload final
{
    const std::shared_ptr<const ParticleBodyCallerLease> lease;
    T value;
    template <typename... Args> ParticleBodyCallerPayload(std::shared_ptr<const ParticleBodyCallerLease> l,Args&&... args)
        : lease(std::move(l)),value(std::forward<Args>(args)...) {}
};
template <typename T,typename... Args>
inline std::shared_ptr<T> retainParticleBodyCallerPayload(
    std::shared_ptr<const ParticleBodyCallerLease> lease,Args&&... args)
{
    auto owner=std::make_shared<ParticleBodyCallerPayload<T>>(std::move(lease),std::forward<Args>(args)...);
    return std::shared_ptr<T>(owner,&owner->value);
}
} // namespace videorender
