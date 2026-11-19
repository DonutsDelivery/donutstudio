# Video helper notices

DonutStudio 1.0.0 uses the GPL-linked video helper on Linux and macOS. The GPL text is in `COPYING`. The private distribution conditions in `FIRST_PARTY_LICENSE.txt` describe the profile with an audited non-GPL dependency closure. Third-party copyrights and licenses listed below apply to their respective files and libraries.

## Source used by the helper

- QuickJS-ng and xsum are under `third_party/quickjs`. Their license and file headers carry the upstream notices.
- Sokol is under `third_party/sokol`. Its upstream license accompanies the source.
- SPIRV-Cross is under `third_party/spirv_cross`. Its source headers offer Apache-2.0 or MIT. See the underlying files for notices from Khronos and Arm.
- RIFE's vendored ncnn implementation is under `src/rife_net`. Its MIT license is in `LICENSE.nihui`.
- The fetched ncnn source retains its BSD-3-Clause license and vendored glslang notices. The CMake recipe records the archive checksum and the patch applied to ncnn.
- The nlohmann JSON, Vulkan-Headers and other pinned source packages retain their upstream license files. The source locks give the versions used by the packaged helper.

## Linux

The bundled Ubuntu FFmpeg libraries come from source package `ffmpeg` version `7:6.1.1-3ubuntu5`. Their configuration enables GPL code, and the shipped libraries identify their license as GPL version 2 or later. The Ubuntu source lock lists the files, patches and versions for every shipped helper library by ELF build ID. Ubuntu source packages include their distribution build recipes and copyright notices.

## macOS

The arm64 Homebrew FFmpeg libraries come from FFmpeg 9.0.1. The formula enables GPL and version 3 code and reports GPL-3.0-or-later. The recorded Homebrew formulas and the source lock give the upstream sources, patches and reference revisions for the bundled dylibs. Packaging rewrites dylib install names and signs the copies it distributes. The dependency source archive contains the matching source and recipes.
