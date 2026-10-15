#pragma once
#include "fixture_native_lambertian_radiance.h"
#include <memory>
#include <string>

namespace arbitgpu
{
class NativeFixtureSceneFrame;
class MetalLambertianArithmeticResult;
namespace lambertian::native
{
// Work units remain separate from geometric queries and deposition.
enum Cost : unsigned {NativeVisits,NativeMoves,NativeScalars,NativeParser,NativeComparisons,
    HostVisits,HostMoves,HostScalars,HostParser,HostComparisons,
    SourceVisits,SourceMoves,SourceScalars,DecimalCalls,WrapperWords,WrapperScalars,
    RegistryVisits,CopiedBytes,ComparedBytes,TransferredBytes,LogicalScratch,CostCount};
using Costs=std::array<std::uint64_t,CostCount>;
struct Storage {std::uint64_t frameBytes=0,attemptBytes=0,nativeBytes=0,attempts=0,publicInspections=0,registryInspections=0,identityBytes=0,arithmeticInspections=0;};
}
// Private fixture hook. The return value describes authenticated native
// completion/publication. The arithmetic entry's bool is read separately.
bool produceMetalLambertianArithmeticForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const lambertian::Request&,const lambertian::Result&,
    std::shared_ptr<const MetalLambertianArithmeticResult>&,std::string&,int stop=-1,
    std::shared_ptr<const MetalLambertianArithmeticResult>* escaped=nullptr);
bool readMetalLambertianArithmeticForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalLambertianArithmeticResult>&,lambertian::Result&,bool&,std::string&);
bool inspectMetalLambertianArithmeticRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalLambertianArithmeticResult>&,std::string&);
bool readMetalLambertianStorageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    lambertian::native::Costs&,lambertian::native::Storage&,std::string&);
bool admitMetalLambertianPreparedForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const lambertian::Ceilings&,std::string&);
} // namespace arbitgpu
