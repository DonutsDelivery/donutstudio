#pragma once
#include "fixture_surface_source_admission.h"

namespace arbitgpu::finitergb
{
// Application schedules only. Compiler, Foundation, driver and allocator
// internals and compiled spills remain unknown. Optical parents have their
// own journals and receive no credit from these wrapper costs.
enum Unit : unsigned { Setup, Parser, Provenance, Scalar, CopiedBytes, ComparedBytes, Retains, Publication,
    Dispatches, WorkItems, BitInterpretations, BandLoads, WeightLoads, Multiplications, Additions,
    RangeChecks, Stores, Control, WeightProducts, WeightAdds, WeightDivisions, WeightConversions, TransferredBytes, UnitCount };
using Costs=std::array<std::uint64_t,UnitCount>;
inline Costs add(const Costs& a,const Costs& b) {Costs out{};for(unsigned k=0;k<UnitCount;++k)out[k]=surfacesource::add(a[k],b[k]);return out;}
inline Costs multiply(const Costs& a,std::uint64_t n) {Costs out{};for(unsigned k=0;k<UnitCount;++k)out[k]=surfacesource::multiply(a[k],n);return out;}
// Exact finiteradiance wrapper vector. RGB scans visit at most 2048 records,
// five buffers, 16 KiB source and fixed stamps, within 32 complete 2 GiB
// walks and 64 operations per word. No scan recursively inspects a parent.
constexpr std::uint64_t scanBytes=32*renderpassoutput::kMaximumTotalBytes;
constexpr Costs scanCost{2048ull*2048*16,65536,2048ull*2048*32,scanBytes/4*64,scanBytes,scanBytes,65536,64};
constexpr Costs factoryCost=scanCost;
constexpr Costs inspectionCost=[](){auto out=scanCost;out[WeightProducts]=8;out[WeightAdds]=8;
    out[WeightDivisions]=8;out[WeightConversions]=8;return out;}();
// One source-scheduled work item, not measured hardware instructions. XYZ
// uses radiance*w once and three component products in each of eight bands.
constexpr Costs nativeCost{0,0,0,0,0,0,0,0,1,1,8,8,8,41,30,6,8,128,0,0,0,0,208};
inline bool journal(std::uint64_t& n,Costs& costs,const Costs& per) {
    if(n==2048ull*2048||costs!=multiply(per,n))return false;
    auto next=multiply(per,n+1);++n;costs=next;return true;
}
inline Costs constructionCost(std::uint64_t sourceBytes) {
    auto out=scanCost;out[CopiedBytes]=surfacesource::add(out[CopiedBytes],8*(sourceBytes+1)+4*208);
    out[WeightProducts]=8;out[WeightAdds]=8;out[WeightDivisions]=8;out[WeightConversions]=8;return out;
}
class Admission final {
public:
    static constexpr std::size_t maximumAttempts=2048;
    static constexpr std::uint64_t maximumScans=maximumAttempts*maximumAttempts;
    struct Attempt final {
        std::uint64_t serial=0,bytes=0,inspections=0,dispatches=0;
        Costs construction{},inspection{},native{};
        std::weak_ptr<const void> result;
        void* payload=nullptr;
        bool live=false,published=false;
    };
    Admission(std::uint64_t factories,std::uint64_t publics,std::uint64_t registries,const std::uint64_t* observed)
        :factories_(factories),publics_(publics),registries_(registries) {
        if(!observed||*observed<sizeof(Admission)||*observed>sizeof(Admission)+256)
            throw std::length_error("Finite RGB fixed control allocation refused");
        baseBytes_=bytes_=*observed;publicCosts_=multiply(scanCost,publics);registryCosts_=multiply(scanCost,registries);
        work_=multiply(factoryCost,factories);
    }
    void factory() {if(factories_==maximumAttempts)throw std::length_error("Finite RGB factory journal exhausted");work_=add(work_,factoryCost);++factories_;}
    bool publicInspection() {return journal(publics_,publicCosts_,scanCost);}
    bool registryInspection() {return journal(registries_,registryCosts_,scanCost);}
    std::size_t begin(std::uint64_t serial,std::uint64_t parent,std::uint64_t bytes,const Costs& construction,std::uint64_t limit) {
        if(count_==maximumAttempts||!serial||(count_&&serial<=records_[count_-1].serial))throw std::length_error("Finite RGB serial refused");
        const auto next=add(work_,construction);auto& r=records_[count_++];r.serial=serial;r.construction=construction;work_=next;
        if(limit>renderpassoutput::kMaximumTotalBytes||surfacesource::add(parent,surfacesource::add(bytes_,bytes))>limit)
            throw std::length_error("Finite RGB whole-frame reservation refused");
        r.bytes=bytes;r.live=true;bytes_=surfacesource::add(bytes_,bytes);return count_-1;
    }
    void inspect(std::size_t slot) {
        auto& r=records_.at(slot);if(scans_==maximumScans||!journal(r.inspections,r.inspection,inspectionCost))throw std::length_error("Finite RGB inspection journal exhausted");
        work_=add(work_,inspectionCost);++scans_;
    }
    void dispatch(std::size_t slot) {
        auto& r=records_.at(slot);if(r.dispatches)throw std::length_error("Finite RGB second dispatch refused");
        work_=add(work_,nativeCost);r.native=nativeCost;r.dispatches=1;
    }
    void attach(std::size_t slot,void* payload) noexcept {records_[slot].payload=payload;}
    void stage(std::size_t slot,const std::shared_ptr<const void>& result) noexcept {records_[slot].result=result;}
    void publish(std::size_t slot) noexcept {records_[slot].published=true;}
    void retire(std::size_t slot) noexcept {auto& r=records_[slot];if(r.live){bytes_-=r.bytes;r.live=false;r.payload=nullptr;r.result.reset();}}
    bool inspectJournal() const {
        if(factories_>maximumAttempts||publics_>maximumScans||registries_>maximumScans
            ||publicCosts_!=multiply(scanCost,publics_)||registryCosts_!=multiply(scanCost,registries_))return false;
        auto total=multiply(factoryCost,factories_);std::uint64_t bytes=baseBytes_,scans=0;
        for(std::size_t k=0;k<count_;++k){const auto& r=records_[k];if(!r.serial||(k&&r.serial<=records_[k-1].serial)
                ||r.dispatches>1||r.native!=multiply(nativeCost,r.dispatches)||r.inspection!=multiply(inspectionCost,r.inspections))return false;
            total=add(total,add(r.construction,add(r.inspection,r.native)));scans=surfacesource::add(scans,r.inspections);
            if(r.live)bytes=surfacesource::add(bytes,r.bytes);else if(r.payload||!r.result.expired())return false;}
        return bytes==bytes_&&scans==scans_&&total==work_;
    }
    const Attempt& record(std::size_t slot) const {return records_.at(slot);}
    Costs work() const {return add(work_,add(publicCosts_,registryCosts_));}
    std::uint64_t bytes() const noexcept {return bytes_;}
    std::uint64_t publics() const noexcept {return publics_;}
    std::uint64_t registries() const noexcept {return registries_;}
    std::size_t attempts() const noexcept {return count_;}
    std::size_t factories() const noexcept {return factories_;}
private:
    std::array<Attempt,maximumAttempts> records_{};
    std::size_t count_=0,factories_=0;
    std::uint64_t baseBytes_=0,bytes_=0,publics_=0,registries_=0,scans_=0;
    Costs work_{},publicCosts_{},registryCosts_{};
};
} // namespace arbitgpu::finitergb
