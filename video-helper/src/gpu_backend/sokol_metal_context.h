#pragma once

#include <mutex>
#include <string>

namespace arbitgpu::sokolmetal
{
// Shared by strict Metal executors so all sokol_gfx calls use the process-owned
// backend lock and initialization state from backend_sokol.mm.
std::mutex& mutex() noexcept;
bool ensure() noexcept;
void* device() noexcept;
const std::string& error() noexcept;
std::string log();
} // namespace arbitgpu::sokolmetal
