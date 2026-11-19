# DonutStudio video helper

The standalone application starts `arbit-video-helper` as a separate process for video decode, preview, rendering and export. This source includes the helper, its vendored dependencies, and the shared files consumed by its build. The repository root is the working directory for the commands below.

## Release correspondence

DonutStudio 1.0.0 uses the helper source in this tree for Linux and macOS. The matching application commits are `d1e1b84281c4bae3f1eebc7719b51dbcf219c7df` and `29846807ab8ff0ceafd7e3c5cc4453865aacb6b7`. The Linux self-extractor packaging change leaves the helper executable and its dependency files intact. Per-package binary hashes belong in the release correspondence receipt.

The files under `shared`, `plugin/Source`, `content/shader-packs` and `tools` are inputs to this helper. The renderer check reads the registration in `plugin/CMakeLists.txt`. The two contract test sources keep the helper project configurable. The commands build only the helper executable.

## Linux x86-64

Use Ubuntu 24.04 amd64 and the `distro` profile of `helper-build-materials/Dockerfile.linux-videohelper`. The release build image is `sha256:f3f7f76949f107a48d69ab1279a1f01107841e77be87f4a034fbd704f7458ce3`. The signed Ubuntu archive records in `helper-build-materials/linux-ubuntu-source-lock.json` name the exact source packages and hashes for the shipped ELF libraries. The separate nlohmann JSON lock records its installed header package.

```sh
cmake -S video-helper -B video-helper/build-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -g1' \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g1' \
  -DARBIT_REQUIRE_VIEWPORT=ON
cmake --build video-helper/build-linux --target arbit-video-helper
```

## macOS arm64

The arm64 helper builds with Apple clang 21.0.0 and the macOS 27.0 SDK, with deployment target 11.0. Configure pkg-config to find the Homebrew libraries listed in `helper-build-materials/macos-bundled-libraries.json`. The versioned recipes and source references are under `helper-build-materials/macos-homebrew`. The bundled copies have their library install names rewritten to paths relative to the helper, then they are signed along with the app.

```sh
cmake -S video-helper -B video-helper/build-macos -G Ninja \
  -DCMAKE_C_COMPILER=/Library/Developer/CommandLineTools/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/Library/Developer/CommandLineTools/usr/bin/clang++ \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DCMAKE_BUILD_TYPE=Release -DARBIT_WITH_NCNN=ON \
  '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -g1' \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g1'
cmake --build video-helper/build-macos --target arbit-video-helper
```

CMake pins its fetched dependency sources in `video-helper/CMakeLists.txt` and `video-helper/cmake`. See `NOTICE.md` for distribution notices.
