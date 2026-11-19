#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace visualnoteinstancing
{
inline constexpr std::uint32_t kMappingSchemaVersion = 2;
inline constexpr std::string_view kMappingTransportHeader = "visual.note-instances.mapping.v2";
inline constexpr std::uint32_t kDirectAxesMappingSchemaVersion = 3;
inline constexpr std::string_view kDirectAxesMappingTransportHeader = "visual.note-instances.mapping.v3";
inline constexpr std::uint32_t kVelocityScaleMappingSchemaVersion = 4;
inline constexpr std::string_view kVelocityScaleMappingTransportHeader = "visual.note-instances.mapping.v4";
inline constexpr std::string_view kLegacyMappingTransportHeader = "visual.note-instances.mapping.v1";
inline constexpr std::size_t kMaximumTransportBytes = 1024;

enum class MappingAxis : std::uint8_t
{
    Onset = 0, LogFrequency = 1, Velocity = 2, Duration = 3, Track = 4,
    PitchBendSemitones = 5, NoteAgeBeats = 6, RemainingDurationBeats = 7,
    RootFlag = 8, MidiNote = 9, ResolvedFrequencyHz = 10,
    Prime2Exponent = 11, Prime3Exponent = 12, Prime5Exponent = 13,
    Prime7Exponent = 14, Prime11Exponent = 15, Prime13Exponent = 16
};
enum class AppearanceSource : std::uint8_t { Velocity = 0, StableNoteIdentity = 1 };
// Which part of the score window becomes instances. The canonical frame holds
// sounding notes, upcoming notes (negative age) and recently ended notes
// (remaining duration at or below zero).
enum class NoteWindow : std::uint8_t
{
    All = 0, Sounding = 1, Upcoming = 2, RecentlyEnded = 3,
    SoundingAndUpcoming = 4, SoundingAndRecentlyEnded = 5
};
inline constexpr std::int32_t kNoFilterAxis = -1;
inline constexpr float kMaximumWindowBeats = 1000000.0f;
inline constexpr float kMaximumFilterMagnitude = 1000000.0f;
inline constexpr std::string_view kSelectionTrailer = "select";

struct Mapping final
{
    std::uint32_t schemaVersion = kMappingSchemaVersion;
    MappingAxis x = MappingAxis::Onset;
    MappingAxis y = MappingAxis::LogFrequency;
    MappingAxis z = MappingAxis::Track;
    float xScale = 1.0f, yScale = 1.0f, zScale = 1.0f, meshScale = 0.1f;
    bool includeMuted = false;
    AppearanceSource appearanceSource = AppearanceSource::Velocity;
    // Linear RGB tint and added emission at the source's normalized endpoints.
    std::array<float, 4> appearanceLow { 1.0f, 1.0f, 1.0f, 0.0f };
    std::array<float, 4> appearanceHigh { 1.0f, 1.0f, 1.0f, 0.0f };
    float velocityScaleLow = 1.0f, velocityScaleHigh = 1.0f;
    // Note selection. The defaults keep every resident note, so graphs saved
    // before it encode the same bytes. windowBeats 0 keeps the whole score window.
    NoteWindow window = NoteWindow::All;
    float windowBeats = 0.0f;
    std::int32_t filterAxis = kNoFilterAxis;
    float filterMin = 0.0f, filterMax = 0.0f;
};

inline bool defaultSelection(const Mapping& m) noexcept
{
    return m.window == NoteWindow::All && m.windowBeats == 0.0f && m.filterAxis == kNoFilterAxis
        && m.filterMin == 0.0f && m.filterMax == 0.0f;
}
inline bool validSelection(const Mapping& m) noexcept
{
    const bool filterRange = m.filterAxis == kNoFilterAxis
        ? m.filterMin == 0.0f && m.filterMax == 0.0f
        : std::isfinite(m.filterMin) && std::isfinite(m.filterMax) && m.filterMin <= m.filterMax
            && std::abs(m.filterMin) <= kMaximumFilterMagnitude && std::abs(m.filterMax) <= kMaximumFilterMagnitude;
    return static_cast<unsigned>(m.window) <= static_cast<unsigned>(NoteWindow::SoundingAndRecentlyEnded)
        && std::isfinite(m.windowBeats) && m.windowBeats >= 0.0f && m.windowBeats <= kMaximumWindowBeats
        && m.filterAxis >= kNoFilterAxis && m.filterAxis <= static_cast<std::int32_t>(MappingAxis::Prime13Exponent)
        && filterRange;
}
// True when a note row with these values becomes an instance.
inline bool selects(const Mapping& m, float ageBeats, float remainingBeats, float filterValue) noexcept
{
    const bool upcoming = ageBeats < 0.0f;
    const bool ended = !upcoming && remainingBeats <= 0.0f;
    const bool sounding = !upcoming && !ended;
    const bool withinBeats = m.windowBeats == 0.0f
        || (upcoming ? -ageBeats <= m.windowBeats : ended ? -remainingBeats <= m.windowBeats : true);
    bool inWindow = true;
    switch (m.window)
    {
        case NoteWindow::All: inWindow = sounding || withinBeats; break;
        case NoteWindow::Sounding: inWindow = sounding; break;
        case NoteWindow::Upcoming: inWindow = upcoming && withinBeats; break;
        case NoteWindow::RecentlyEnded: inWindow = ended && withinBeats; break;
        case NoteWindow::SoundingAndUpcoming: inWindow = sounding || (upcoming && withinBeats); break;
        case NoteWindow::SoundingAndRecentlyEnded: inWindow = sounding || (ended && withinBeats); break;
    }
    if (!inWindow) return false;
    return m.filterAxis == kNoFilterAxis
        || (std::isfinite(filterValue) && filterValue >= m.filterMin && filterValue <= m.filterMax);
}

inline bool validAxis(MappingAxis axis) noexcept
{
    return static_cast<unsigned>(axis) <= static_cast<unsigned>(MappingAxis::Prime13Exponent);
}
inline bool valid(const Mapping& m) noexcept
{
    const auto validAppearance = [](const auto& value)
    {
        for (std::size_t channel = 0; channel < value.size(); ++channel)
            if (!std::isfinite(value[channel]) || value[channel] < 0.0f
                || (channel != 3 && value[channel] > 1.0f))
                return false;
        return true;
    };
    const Mapping defaults;
    const bool exactVersion = m.schemaVersion == kMappingSchemaVersion
        || m.schemaVersion == kDirectAxesMappingSchemaVersion
        || m.schemaVersion == kVelocityScaleMappingSchemaVersion
        || (m.schemaVersion == 1 && m.appearanceSource == defaults.appearanceSource
            && m.appearanceLow == defaults.appearanceLow && m.appearanceHigh == defaults.appearanceHigh);
    const auto versionedAxis = [&m](MappingAxis axis)
    {
        return validAxis(axis) && (m.schemaVersion == kDirectAxesMappingSchemaVersion
            || m.schemaVersion == kVelocityScaleMappingSchemaVersion
            || static_cast<unsigned>(axis) <= static_cast<unsigned>(MappingAxis::Track));
    };
    const auto validResponse = [&m](float response)
    {
        if (m.schemaVersion != kVelocityScaleMappingSchemaVersion) return response == 1.0f;
        const auto product = static_cast<double>(m.meshScale) * response;
        return std::isfinite(response) && response >= 0.000001f && response <= 10000.0f
            && std::isfinite(product) && product <= 10000.0
            && static_cast<float>(product) > 0.0f;
    };
    return exactVersion && validResponse(m.velocityScaleLow) && validResponse(m.velocityScaleHigh)
        && versionedAxis(m.x) && versionedAxis(m.y) && versionedAxis(m.z)
        && static_cast<unsigned>(m.appearanceSource) <= static_cast<unsigned>(AppearanceSource::StableNoteIdentity)
        && validAppearance(m.appearanceLow) && validAppearance(m.appearanceHigh)
        && validSelection(m) && (m.schemaVersion != 1 || defaultSelection(m))
        && std::isfinite(m.xScale) && std::isfinite(m.yScale) && std::isfinite(m.zScale)
        && std::isfinite(m.meshScale) && std::abs(m.xScale) <= 10000.0f
        && std::abs(m.yScale) <= 10000.0f && std::abs(m.zScale) <= 10000.0f
        && m.meshScale > 0.0f && m.meshScale <= 10000.0f;
}
inline std::uint32_t floatBits(float value) noexcept { std::uint32_t bits=0; std::memcpy(&bits,&value,sizeof(bits)); return bits; }
inline float floatFromBits(std::uint32_t bits) noexcept { float value=0; std::memcpy(&value,&bits,sizeof(value)); return value; }
inline std::string encode(const Mapping& m)
{
    if (!valid(m)) return {};
    std::ostringstream s;
    s << (m.schemaVersion == 1 ? kLegacyMappingTransportHeader
        : m.schemaVersion == kVelocityScaleMappingSchemaVersion ? kVelocityScaleMappingTransportHeader
        : m.schemaVersion == kDirectAxesMappingSchemaVersion ? kDirectAxesMappingTransportHeader
        : kMappingTransportHeader)
      << '\n' << m.schemaVersion << ' ' << static_cast<unsigned>(m.x) << ' '
      << static_cast<unsigned>(m.y) << ' ' << static_cast<unsigned>(m.z) << ' ' << floatBits(m.xScale) << ' '
      << floatBits(m.yScale) << ' ' << floatBits(m.zScale) << ' ' << floatBits(m.meshScale) << ' '
      << (m.includeMuted ? 1 : 0) << '\n';
    if (m.schemaVersion != 1)
    {
        s << static_cast<unsigned>(m.appearanceSource);
        for (float value : m.appearanceLow) s << ' ' << floatBits(value);
        for (float value : m.appearanceHigh) s << ' ' << floatBits(value);
        s << '\n';
    }
    if (m.schemaVersion == kVelocityScaleMappingSchemaVersion)
        s << floatBits(m.velocityScaleLow) << ' ' << floatBits(m.velocityScaleHigh) << '\n';
    if (!defaultSelection(m))
        s << kSelectionTrailer << ' ' << static_cast<unsigned>(m.window) << ' ' << floatBits(m.windowBeats) << ' '
          << (m.filterAxis + 1) << ' ' << floatBits(m.filterMin) << ' ' << floatBits(m.filterMax) << '\n';
    auto result=s.str(); return result.size() <= kMaximumTransportBytes ? result : std::string{};
}
inline bool decode(std::string_view encoded, Mapping& output)
{
    output = {};
    if (encoded.empty() || encoded.size() > kMaximumTransportBytes || encoded.find('\r') != std::string_view::npos) return false;
    std::istringstream stream{std::string(encoded)}; std::string header;
    if (!std::getline(stream,header)
        || (header != kMappingTransportHeader && header != kLegacyMappingTransportHeader
            && header != kDirectAxesMappingTransportHeader
            && header != kVelocityScaleMappingTransportHeader)) return false;
    std::uint64_t schema=0,x=0,y=0,z=0,xs=0,ys=0,zs=0,scale=0; int muted=-1;
    if (!(stream>>schema>>x>>y>>z>>xs>>ys>>zs>>scale>>muted) || schema>0xffffffffull || x>255 || y>255 || z>255
        || xs>0xffffffffull || ys>0xffffffffull || zs>0xffffffffull || scale>0xffffffffull || (muted!=0 && muted!=1)) return false;
    Mapping candidate; candidate.schemaVersion=static_cast<std::uint32_t>(schema); candidate.x=static_cast<MappingAxis>(x);
    candidate.y=static_cast<MappingAxis>(y); candidate.z=static_cast<MappingAxis>(z);
    candidate.xScale=floatFromBits(static_cast<std::uint32_t>(xs)); candidate.yScale=floatFromBits(static_cast<std::uint32_t>(ys));
    candidate.zScale=floatFromBits(static_cast<std::uint32_t>(zs)); candidate.meshScale=floatFromBits(static_cast<std::uint32_t>(scale)); candidate.includeMuted=muted!=0;
    if (header != kLegacyMappingTransportHeader)
    {
        std::uint64_t source = 0;
        const auto expectedSchema = header == kVelocityScaleMappingTransportHeader
            ? kVelocityScaleMappingSchemaVersion : header == kDirectAxesMappingTransportHeader
            ? kDirectAxesMappingSchemaVersion : kMappingSchemaVersion;
        if (schema != expectedSchema || !(stream >> source) || source > 1) return false;
        candidate.appearanceSource = static_cast<AppearanceSource>(source);
        for (auto* endpoint : { &candidate.appearanceLow, &candidate.appearanceHigh })
            for (auto& value : *endpoint)
            {
                std::uint64_t bits = 0;
                if (!(stream >> bits) || bits > 0xffffffffull) return false;
                value = floatFromBits(static_cast<std::uint32_t>(bits));
            }
    }
    else if (schema != 1) return false;
    if (header == kVelocityScaleMappingTransportHeader)
    {
        std::uint64_t low = 0, high = 0;
        if (!(stream >> low >> high) || low > 0xffffffffull || high > 0xffffffffull) return false;
        candidate.velocityScaleLow = floatFromBits(static_cast<std::uint32_t>(low));
        candidate.velocityScaleHigh = floatFromBits(static_cast<std::uint32_t>(high));
    }
    stream>>std::ws;
    if (!stream.eof())
    {
        std::string word; std::uint64_t window=0, beats=0, axis=0, low=0, high=0;
        if (!(stream>>word>>window>>beats>>axis>>low>>high) || word!=kSelectionTrailer || window>255
            || beats>0xffffffffull || axis>255 || low>0xffffffffull || high>0xffffffffull) return false;
        candidate.window=static_cast<NoteWindow>(window);
        candidate.windowBeats=floatFromBits(static_cast<std::uint32_t>(beats));
        candidate.filterAxis=static_cast<std::int32_t>(axis)-1;
        candidate.filterMin=floatFromBits(static_cast<std::uint32_t>(low));
        candidate.filterMax=floatFromBits(static_cast<std::uint32_t>(high));
        stream>>std::ws;
    }
    if (!stream.eof() || !valid(candidate) || encode(candidate)!=encoded) return false;
    output=candidate; return true;
}

inline float appearanceValue(const Mapping& mapping, std::int64_t identity, float velocity) noexcept
{
    if (mapping.appearanceSource == AppearanceSource::Velocity)
        return std::fmax(0.0f, std::fmin(1.0f, velocity));
    // This is a palette coordinate, never a replacement for the canonical ID.
    // Fixed integer arithmetic keeps the same signed ID stable across backends.
    auto bits = static_cast<std::uint64_t>(identity) + 0x9e3779b97f4a7c15ull;
    bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ull;
    bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebull;
    bits ^= bits >> 31;
    return static_cast<float>(bits >> 40) / 16777215.0f;
}
} // namespace visualnoteinstancing
