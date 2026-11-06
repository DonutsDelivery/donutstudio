#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace videorender
{
template<class T>
struct ParticleZeroFill { static T value() noexcept { return T {}; } };
// A dead planar body: off screen, no life, no hue.
struct ParticleDeadBodyFill { static std::array<float,4> value() noexcept { return {-10,-10,0,0}; } };

// A list sized from the scene. It reads like the filled fixed array it
// replaced: an index past the end reads the fill value, a write grows the
// list, and equality pads the shorter list with the fill value. Size it up
// front where the count is known, so the heap holds exactly what is charged.
template<class T, class Fill = ParticleZeroFill<T>>
struct ParticleSizedList
{
    std::vector<T> values;
    ParticleSizedList() = default;
    explicit ParticleSizedList(std::size_t count) : values(count, Fill::value()) {}
    T& operator[](std::size_t index)
    {
        if (index >= values.size()) values.resize(index + 1, Fill::value());
        return values[index];
    }
    const T& operator[](std::size_t index) const
    {
        static const T fill = Fill::value();
        return index < values.size() ? values[index] : fill;
    }
    std::size_t size() const noexcept { return values.size(); }
    std::size_t capacity() const noexcept { return values.capacity(); }
    bool empty() const noexcept { return values.empty(); }
    void clear() noexcept { values.clear(); }
    void resize(std::size_t count) { values.resize(count, Fill::value()); }
    void reserve(std::size_t count) { values.reserve(count); }
    void push_back(const T& value) { values.push_back(value); }
    T* data() noexcept { return values.data(); }
    const T* data() const noexcept { return values.data(); }
    std::uint64_t heapBytes() const noexcept { return std::uint64_t(values.capacity())*sizeof(T); }
    auto begin() noexcept { return values.begin(); }
    auto end() noexcept { return values.end(); }
    auto begin() const noexcept { return values.begin(); }
    auto end() const noexcept { return values.end(); }
    friend bool operator==(const ParticleSizedList& a, const ParticleSizedList& b)
    {
        const auto count = std::max(a.values.size(), b.values.size());
        for (std::size_t index = 0; index < count; ++index)
            if (!(a[index] == b[index])) return false;
        return true;
    }
    friend bool operator!=(const ParticleSizedList& a, const ParticleSizedList& b) { return !(a == b); }
};
}
