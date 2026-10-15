#pragma once

#include "fixture_surface_source_admission.h"

namespace arbitgpu::completebase
{
// Source-level application work. Allocator, compiler and driver internals are
// unknown. Each unit remains separate, including unsuccessful attempts/scans.
struct Costs final {
    std::uint64_t setup=0,parser=0,provenance=0,scalar=0,upload=0,draw=0,
        shadow=0,transfer=0,copy=0,compare=0,retain=0,publication=0;
    bool operator==(const Costs& b) const noexcept {
        return setup==b.setup&&parser==b.parser&&provenance==b.provenance&&scalar==b.scalar
            &&upload==b.upload&&draw==b.draw&&shadow==b.shadow&&transfer==b.transfer
            &&copy==b.copy&&compare==b.compare&&retain==b.retain&&publication==b.publication;
    }
};
inline Costs add(const Costs& a,const Costs& b) {
    using surfacesource::add;
    return {add(a.setup,b.setup),add(a.parser,b.parser),add(a.provenance,b.provenance),add(a.scalar,b.scalar),
        add(a.upload,b.upload),add(a.draw,b.draw),add(a.shadow,b.shadow),add(a.transfer,b.transfer),
        add(a.copy,b.copy),add(a.compare,b.compare),add(a.retain,b.retain),add(a.publication,b.publication)};
}
inline Costs multiply(const Costs& a,std::uint64_t n) {
    using surfacesource::multiply;
    return {multiply(a.setup,n),multiply(a.parser,n),multiply(a.provenance,n),multiply(a.scalar,n),
        multiply(a.upload,n),multiply(a.draw,n),multiply(a.shadow,n),multiply(a.transfer,n),
        multiply(a.copy,n),multiply(a.compare,n),multiply(a.retain,n),multiply(a.publication,n)};
}
inline constexpr std::uint64_t scanBytes=renderpassoutput::kMaximumTotalBytes*64;
inline constexpr Costs factoryCost{65536,65536,4096*2048,scanBytes/4*64,0,0,0,0,scanBytes,scanBytes,0,0};
inline constexpr Costs inspectionCost{65536,65536,4096*2048,scanBytes/4*64,0,0,0,0,scanBytes,scanBytes,4096,0};
class Admission final {
public:
    static constexpr std::size_t maximumAttempts=2048;
    // A shared scan ceiling keeps the scalar envelope below2^64 even with
    // every factory and construction charge. Nested parents remain separate.
    static constexpr std::uint64_t maximumInspections=maximumAttempts*maximumAttempts;
    struct Attempt final {
        std::uint64_t serial=0,bytes=0,inspections=0;
        Costs construction{},inspection{};
        bool live=false,published=false;
    };
    static std::uint64_t fixedBytes(std::uint64_t parent,std::uint64_t metadata) {
        return surfacesource::add(parent,surfacesource::add(sizeof(Admission),metadata));
    }
    explicit Admission(std::size_t factories):factories_(factories),work_(multiply(factoryCost,factories)) {}
    Admission(std::size_t factories,const std::uint64_t* actual):Admission(factories) {
        if(!actual||*actual<sizeof(Admission)||*actual>sizeof(Admission)+256)throw std::length_error("Complete base control allocation exceeds fixed metadata");
        baseBytes_=bytes_=*actual;
    }
    void factory() {
        if(factories_==maximumAttempts) throw std::length_error("Complete base factory journal exhausted");
        const auto next=add(work_,factoryCost);++factories_;work_=next;
    }
    std::size_t begin(std::uint64_t serial,std::uint64_t parent,std::uint64_t bytes,
        const Costs& costs,std::uint64_t limit) {
        if(count_==maximumAttempts||!serial||(count_&&serial<=records_[count_-1].serial))
            throw std::length_error("Complete base attempt serial refused");
        const auto next=add(work_,costs);auto& record=records_[count_++];
        record.serial=serial;record.construction=costs;work_=next;
        if(limit>renderpassoutput::kMaximumTotalBytes||surfacesource::add(parent,surfacesource::add(bytes_,bytes))>limit)
            throw std::length_error("Complete base simultaneous ownership exceeds aggregate admission");
        bytes_=surfacesource::add(bytes_,bytes);record.bytes=bytes;record.live=true;return count_-1;
    }
    void inspect(std::size_t slot) {
        if(slot>=count_||records_[slot].inspections==maximumInspections||inspections_+registries_==maximumInspections)
            throw std::length_error("Complete base immutable scan journal exhausted");
        auto& record=records_[slot];
        const auto expected=multiply(inspectionCost,record.inspections);
        if(!(expected==record.inspection)) throw std::length_error("Complete base inspection journal differs");
        const auto next=add(work_,inspectionCost);const auto scan=multiply(inspectionCost,record.inspections+1);
        ++record.inspections;++inspections_;record.inspection=scan;work_=next;
    }
    void publish(std::size_t slot) noexcept {records_[slot].published=true;}
    void retire(std::size_t slot) noexcept {
        auto& record=records_[slot];if(record.live){bytes_-=record.bytes;record.live=false;}
    }
    void registry() {
        if(inspections_+registries_==maximumInspections)throw std::length_error("Complete base registry scan journal exhausted");
        if(!(registryCosts_==multiply(inspectionCost,registries_)))throw std::length_error("Complete base registry work differs");
        const auto next=add(work_,inspectionCost);const auto scan=multiply(inspectionCost,registries_+1);
        ++registries_;registryCosts_=scan;work_=next;
    }
    bool inspectJournal() const {
        Costs expected=add(multiply(factoryCost,factories_),multiply(inspectionCost,registries_));std::uint64_t bytes=baseBytes_,inspections=0;
        for(std::size_t i=0;i<count_;++i){const auto& record=records_[i];
            if(!(record.inspection==multiply(inspectionCost,record.inspections)))return false;
            expected=add(expected,add(record.construction,record.inspection));
            inspections=surfacesource::add(inspections,record.inspections);
            if(record.live)bytes=surfacesource::add(bytes,record.bytes);
        }
        return expected==work_&&bytes==bytes_&&inspections==inspections_&&registryCosts_==multiply(inspectionCost,registries_);
    }
    const Attempt& record(std::size_t slot) const {return records_.at(slot);}
    std::uint64_t bytes() const noexcept {return bytes_;}
    std::size_t attempts() const noexcept {return count_;}
    const Costs& work() const noexcept {return work_;}
private:
    std::array<Attempt,maximumAttempts> records_{};
    std::size_t count_=0,factories_=0;
    std::uint64_t baseBytes_=sizeof(Admission),bytes_=sizeof(Admission),inspections_=0,registries_=0;
    Costs work_{},registryCosts_{};
};
} // namespace arbitgpu::completebase
