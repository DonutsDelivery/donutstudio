#pragma once
#include "fixture_surface_source_admission.h"
#include <array>

namespace arbitgpu::finiteradiance
{
// Application statements only. Allocator, compiler, Foundation, driver internals
// and compiled spills are unknown. Native/host arithmetic and parent inspectors
// retain their own journals; these units describe only the carrier's wrapper.
enum Unit : unsigned { Setup, Parser, Provenance, Scalar, CopiedBytes, ComparedBytes, Retains, Publication, UnitCount };
using Costs=std::array<std::uint64_t,UnitCount>;
inline Costs add(const Costs& a,const Costs& b) {
    Costs result{};for(unsigned k=0;k<UnitCount;++k)result[k]=surfacesource::add(a[k],b[k]);return result;
}
inline Costs multiply(const Costs& a,std::uint64_t count) {
    Costs result{};for(unsigned k=0;k<UnitCount;++k)result[k]=surfacesource::multiply(a[k],count);return result;
}
// A scan walks at most 2048 wrapper records and 8 child associations per record.
// Identity comparisons/copies, registry lookup, unwind, retained-byte scans and
// no-elision Costs temporaries fit 32 whole-frame byte walks and 64 operations per
// word. This excludes the separately charged optical/numerical child inspectors.
constexpr std::uint64_t scanBytes=32*renderpassoutput::kMaximumTotalBytes;
constexpr Costs scanCost{2048ull*2048*16,65536,2048ull*2048*32,scanBytes/4*64,scanBytes,scanBytes,65536,64};
constexpr Costs factoryCost=scanCost;
constexpr Costs inspectionCost=scanCost;
inline bool validJournal(std::uint64_t count,const Costs& costs,const Costs& per) {
    if(count>2048ull*2048)return false;
    return costs==multiply(per,count);
}
inline bool journal(std::uint64_t& count,Costs& costs,const Costs& per) {
    if(count==2048ull*2048||!validJournal(count,costs,per))return false;
    const auto next=multiply(per,count+1);++count;costs=next;return true;
}
class Admission final {
public:
    static constexpr std::size_t maximumAttempts=2048;
    static constexpr std::uint64_t maximumScans=maximumAttempts*maximumAttempts;
    struct Attempt final {
        std::uint64_t serial=0,bytes=0,inspections=0,identityBytes=0,identityBound=0,firstChildSlot=0;
        Costs construction{},inspection{};
        std::array<std::uint32_t,5> support{};
        std::uint32_t cellX=0,cellY=0;
        bool live=false,published=false;
    };
    Admission(std::size_t factories,const std::uint64_t* observed):factories_(factories),work_(multiply(factoryCost,factories)) {
        if(!observed||*observed<sizeof(Admission)||*observed>sizeof(Admission)+256)
            throw std::length_error("Finite radiance fixed control allocation refused");
        baseBytes_=bytes_=*observed;
    }
    void factory() {
        if(factories_==maximumAttempts)throw std::length_error("Finite radiance factory journal exhausted");
        const auto next=add(work_,factoryCost);++factories_;work_=next;
    }
    std::size_t begin(std::uint64_t serial,std::uint64_t identityBytes,std::uint64_t identityBound,std::uint64_t firstChildSlot,
        std::uint64_t parent,std::uint64_t bytes,const Costs& construction,std::uint64_t limit) {
        if(count_==maximumAttempts||!serial||(count_&&serial<=records_[count_-1].serial))
            throw std::length_error("Finite radiance attempt serial refused");
        const auto next=add(work_,construction);auto& record=records_[count_++];
        record.serial=serial;record.identityBytes=identityBytes;record.identityBound=identityBound;record.firstChildSlot=firstChildSlot;
        record.construction=construction;work_=next;
        if(limit>renderpassoutput::kMaximumTotalBytes||surfacesource::add(parent,surfacesource::add(bytes_,bytes))>limit)
            throw std::length_error("Finite radiance whole-frame simultaneous storage refused");
        record.bytes=bytes;record.live=true;bytes_=surfacesource::add(bytes_,bytes);return count_-1;
    }
    void inspect(std::size_t slot) {
        if(slot>=count_)throw std::length_error("Finite radiance inspection slot refused");
        scan(records_[slot].inspections,records_[slot].inspection);
    }
    void publish(std::size_t slot,const std::array<std::uint32_t,5>& support) noexcept {records_[slot].support=support;records_[slot].published=true;}
    void retire(std::size_t slot) noexcept {auto& record=records_[slot];if(record.live){bytes_-=record.bytes;record.live=false;}}
    bool inspectJournal() const {
        Costs expected=multiply(factoryCost,factories_);
        std::uint64_t bytes=baseBytes_,scans=0;
        for(std::size_t i=0;i<count_;++i){const auto& r=records_[i];
            if(!r.serial||(i&&r.serial<=records_[i-1].serial)||r.inspection!=multiply(inspectionCost,r.inspections))return false;
            expected=add(expected,add(r.construction,r.inspection));scans=surfacesource::add(scans,r.inspections);
            if(r.live)bytes=surfacesource::add(bytes,r.bytes);
        }
        return expected==work_&&bytes==bytes_&&scans==scans_;
    }
    const Attempt& record(std::size_t slot) const {return records_.at(slot);}
    const Costs& work() const noexcept {return work_;}
    std::uint64_t bytes() const noexcept {return bytes_;}
    std::size_t attempts() const noexcept {return count_;}
    std::size_t factories() const noexcept {return factories_;}
private:
    void scan(std::uint64_t& count,Costs& costs) {
        if(scans_==maximumScans||costs!=multiply(inspectionCost,count))throw std::length_error("Finite radiance permanent scan journal refused");
        const auto next=add(work_,inspectionCost),nextCosts=multiply(inspectionCost,count+1);
        ++count;++scans_;costs=nextCosts;work_=next;
    }
    std::array<Attempt,maximumAttempts> records_{};
    std::size_t factories_=0,count_=0;
    std::uint64_t baseBytes_=0,bytes_=0,scans_=0;
    Costs work_{};
};
} // namespace arbitgpu::finiteradiance
