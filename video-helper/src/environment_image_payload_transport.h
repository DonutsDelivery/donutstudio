#pragma once

#include "model_payload_transport.h"
#include "../../shared/VisualEnvironmentImageAssetPayload.h"
#include "../../shared/RadianceEnvironmentImageDecoder.h"
#include <tuple>

namespace videohelper::environmentimage
{
struct Resident final
{
    visualenvironmentimagepayload::PayloadPtr payload;
    visualenvironmentimage::DecodedImage image;
    std::array<float,3> maximumRadiance{};
};
using ResidentPtr = std::shared_ptr<const Resident>;

// Helper-owned exact bytes and linear pixels. Native preparations retain a lease
// so resident eviction cannot invalidate a submitted preview or export frame.
class Store final
{
public:
    bool begin(const std::string& transferId, const visualenvironmentimage::Identity& key,
               const visualenvironmentimage::Metadata& metadata, std::uint64_t bytes,
               std::string& error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!visualmodelasset::validStableAssetId(transferId) || pending_.count(transferId)
            || !visualmodelasset::validStableAssetId(key.id) || key.version==0
            || key.version>std::uint64_t(std::numeric_limits<std::int64_t>::max())
            || !visualmodelasset::validSha256(key.contentSha256) || !visualenvironmentimage::valid(metadata)
            || bytes==0 || bytes>visualenvironmentimagepayload::kMaxPayloadBytes
            || pending_.size()>=visualenvironmentimagepayload::kMaxConcurrentTransfers
            || bytes>visualenvironmentimagepayload::kMaxPendingBytes-pendingBytes_)
            return reject(error,"HDR environment transfer identity or budget is invalid");
        try
        {
            Pending pending {key,metadata,std::vector<std::uint8_t>(static_cast<std::size_t>(bytes)),0};
            pending_.emplace(transferId,std::move(pending)); pendingBytes_+=bytes;
        }
        catch (const std::exception&) { return reject(error,"HDR environment transfer allocation failed"); }
        error.clear(); return true;
    }
    bool append(const std::string& transferId, std::uint64_t offset, const std::string& encoded,
                std::uint64_t& received, std::string& error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=pending_.find(transferId);
        if (found==pending_.end()) return reject(error,"HDR environment transfer is unknown");
        auto& pending=found->second;
        std::vector<std::uint8_t> chunk;
        if (offset!=pending.received || !modelpayload::detail::decodeBase64Bounded(encoded,
                visualenvironmentimagepayload::kMaxChunkBytes,chunk)
            || chunk.size()>pending.bytes.size()-pending.received)
            return reject(error,"HDR environment chunk is malformed or out of order");
        std::copy(chunk.begin(),chunk.end(),pending.bytes.begin()+static_cast<std::ptrdiff_t>(offset));
        received=pending.received+=chunk.size(); error.clear(); return true;
    }
    ResidentPtr commit(const std::string& transferId, std::string& error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=pending_.find(transferId);
        if (found==pending_.end()) { reject(error,"HDR environment transfer is unknown"); return {}; }
        if (found->second.received!=found->second.bytes.size())
        { reject(error,"HDR environment transfer is incomplete"); return {}; }
        auto pending=std::move(found->second);
        pendingBytes_-=pending.bytes.size(); pending_.erase(found);
        if (modelpayload::detail::digestHex(pending.bytes)!=pending.key.contentSha256)
        { reject(error,"HDR environment digest differs from exact identity"); return {}; }
        const Key key {pending.key.id,pending.key.version,pending.key.contentSha256};
        const auto prior=resident_.find(key);
        if (prior!=resident_.end())
        {
            if (!sameMetadata(prior->second->image.metadata,pending.metadata)
                || prior->second->payload->bytes().size()!=pending.bytes.size())
            { reject(error,"HDR environment interpretation differs from resident identity"); return {}; }
            error.clear(); return prior->second;
        }
        const auto decoded=visualenvironmentimage::decodedBytes(pending.metadata);
        const auto source=pending.bytes.size();
        while (resident_.size()>=visualenvironmentimagepayload::kMaxResidentPayloads
            || source>visualenvironmentimagepayload::kMaxResidentSourceBytes-sourceBytes_
            || decoded>visualenvironmentimagepayload::kMaxResidentDecodedBytes-decodedBytes_
            || source+decoded>visualenvironmentimagepayload::kMaxResidentCombinedBytes-sourceBytes_-decodedBytes_)
        {
            auto victim=std::find_if(resident_.begin(),resident_.end(),[](const auto& value)
                { return value.second.use_count()==1; });
            if (victim==resident_.end()) { reject(error,"HDR environment resident budget is occupied"); return {}; }
            sourceBytes_-=victim->second->payload->bytes().size();
            decodedBytes_-=visualenvironmentimage::decodedBytes(victim->second->image.metadata);
            resident_.erase(victim);
        }
        try
        {
            auto resident=std::make_shared<Resident>();
            if (!visualenvironmentimage::decodeRadianceEnvironment(pending.bytes.data(),pending.bytes.size(),
                    resident->image,error) || !sameMetadata(resident->image.metadata,pending.metadata))
            { reject(error,"HDR environment source interpretation is invalid"); return {}; }
            for (const auto& pixel:resident->image.pixels)
                for (std::size_t c=0;c<3;++c) resident->maximumRadiance[c]=std::max(resident->maximumRadiance[c],pixel[c]);
            resident->payload=visualenvironmentimagepayload::ImmutablePayload::create(
                std::move(pending.key),pending.metadata,std::move(pending.bytes));
            if (!resident->payload) { reject(error,"HDR environment immutable payload admission failed"); return {}; }
            resident_.emplace(key,resident); sourceBytes_+=source; decodedBytes_+=decoded;
            error.clear(); return resident;
        }
        catch (const std::exception&) { reject(error,"HDR environment resident allocation failed"); return {}; }
    }
    bool abort(const std::string& transferId)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=pending_.find(transferId);
        if (found==pending_.end()) return false;
        pendingBytes_-=found->second.bytes.size(); pending_.erase(found); return true;
    }
    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.clear(); pendingBytes_=0;
        // Submitted native frames may still retain a resident. Keep its budget
        // charged until that lease is released, even across a project reset.
        for (auto found=resident_.begin();found!=resident_.end();)
        {
            if (found->second.use_count()!=1) {++found;continue;}
            sourceBytes_-=found->second->payload->bytes().size();
            decodedBytes_-=visualenvironmentimage::decodedBytes(found->second->image.metadata);
            found=resident_.erase(found);
        }
    }
    ResidentPtr resolve(const visualenvironmentimage::Identity& key) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=resident_.find(Key{key.id,key.version,key.contentSha256});
        return found==resident_.end() ? ResidentPtr{} : found->second;
    }
private:
    static bool reject(std::string& error,const char* text) { error=text; return false; }
    static bool sameMetadata(const visualenvironmentimage::Metadata& a,const visualenvironmentimage::Metadata& b)
    { return a.width==b.width && a.height==b.height && a.sourcePrimaries==b.sourcePrimaries; }
    using Key=std::tuple<std::string,std::uint64_t,std::string>;
    struct Pending
    {
        visualenvironmentimage::Identity key;
        visualenvironmentimage::Metadata metadata;
        std::vector<std::uint8_t> bytes;
        std::uint64_t received=0;
    };
    mutable std::mutex mutex_;
    std::map<std::string,Pending> pending_;
    std::map<Key,ResidentPtr> resident_;
    std::uint64_t pendingBytes_=0,sourceBytes_=0,decodedBytes_=0;
};
inline Store& store() { static Store value; return value; }
} // namespace videohelper::environmentimage
