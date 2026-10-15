# Audited source prefix binding

`ARBIT_VIDEO_DEPENDENCY_PREFIX=/absolute/prefix` opts the helper into strict FFmpeg binding. With no prefix, the existing system pkg-config route remains active. The option does not change Lua, GLFW, NCNN or MoltenVK discovery.

The option consumes the existing staging recipes and their receipts. Linux admits `lgpl-linux-v1` for an x86_64 target at `<stage>/prefix`. Windows admits `lgpl-windows-mingw-v1` for x86_64 MinGW at `<stage>/sysroot/mingw64`. Mac admits `lgpl-macos-arm64-v1` for an arm64 helper at `<stage>/prefix`; a universal helper or Intel helper is refused. Admission uses the CMake target architecture rather than the build host architecture. Each stage must contain its recipe's `DEPENDENCY_BUILD.json` with PASS status and matching scope, source identities, current recipe input hashes and cache identity. Mac and Windows profile files must be integrated before those options can configure.

Configure verifies every installed prefix file against `artifactHashes` and refuses missing files, changed bytes, extra files and symlinks escaping the prefix. All seven public AV headers and pkg-config files must exist. The AV pkg-config query runs with only the prefix's `lib/pkgconfig` directory and an empty sysroot override. The caller's pkg-config environment is restored before other dependency discovery.

The helper's `PkgConfig::AV` target uses explicit library paths resolved inside the prefix without CMake's cross-compilation root rewriting. Admission requires every include, library and pkg-config directory inside the prefix and the complete seven-library shared FFmpeg closure. Bare library names, system fallback, static libav archives and additional compiler or linker options fail configuration. This checks source-prefix identity and link admission. It does not establish codec quality, runtime hardware support, installed-package behavior, dependency rights or a new helper license. Existing license and publication gates still apply.

## Private ten-bit export binding

`ARBIT_BUILD_KVAZAAR10_ADAPTER=ON` retains the admission rules in `cmake/Kvazaar10.cmake`. Only after that module provides `arbit-kvazaar10-adapter` does the helper link it privately and receive `ARBIT_HAVE_KVAZAAR10_EXPORT=1`. The adapter's header path and `KVZ_BIT_DEPTH` definition remain private to its own targets. The plugin receives neither. The existing GNU-compatible archive transformation admits ELF/COFF tools; Apple and MSVC remain refused.

The guarded exporter checkpoint depends on this helper definition and link. This change supplies that wiring without changing exporter behavior or its mux contract. The adapter remains opt-in, so an ordinary build does not enable private ten-bit export. Real Main10/HDR encode, independently decoded pixels and metadata, mux timestamps, cancellation, audio synchronization and packaged platform acceptance still need evidence.

## Prepared checks

After integration, the owner can run the receipt/binding fixtures and source guards without compiling:

```bash
python3 video-helper/tests/strict_dependency_prefix_binding_tests.py
python3 video-helper/tests/strict_dependency_prefix_source_contract.py
```

CMake registers them as `strict_dependency_prefix_binding` and `strict_dependency_prefix_source_contract`. The fixtures create synthetic recipe receipts in temporary directories for all three existing staging layouts. They exercise target admission, recipe and artifact drift, source identity, missing receipts, system path contamination, bare/static libraries and escaping symlinks. They do not prove a built prefix. This checkpoint prepares those checks; it does not claim they have run.
