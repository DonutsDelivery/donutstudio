#!/usr/bin/env python3
"""Admit existing source-staging receipts and the helper's actual libav binding."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

# These paths and layouts belong to the existing staging recipes.
PROFILES = {
    'Linux': ('lgpl-linux-v1', 'scripts/ffmpeg-lgpl-candidate/policy-6.1.6.json',
              'scripts/video-helper-dependencies/build-linux.py', 'prefix'),
    'Windows': ('lgpl-windows-mingw-v1', 'scripts/video-helper-dependencies/windows-profile.json',
                'scripts/video-helper-dependencies/build-windows.py', 'sysroot/mingw64'),
    'Darwin': ('lgpl-macos-arm64-v1', 'scripts/video-helper-dependencies/profile-macos-arm64-v1.json',
               'scripts/video-helper-dependencies/build-macos-source.py', 'prefix'),
}
RECIPE_INPUTS = {
    'Linux': ['scripts/video-helper-dependencies/audit.py', 'scripts/ffmpeg-lgpl-candidate/candidate.py',
              'Dockerfile.linux-videohelper', 'scripts/build-linux-local.sh',
              'scripts/build-linux-docker-videohelper.sh'],
    'Windows': ['scripts/video-helper-dependencies/windows.py', 'scripts/ffmpeg-lgpl-candidate/candidate.py',
                'scripts/ffmpeg-lgpl-candidate/policy-6.1.6.json', 'Dockerfile.windows-videohelper-lgpl',
                'scripts/mingw-w64-x86_64.cmake', 'scripts/build-windows-videohelper-local.sh',
                'scripts/build-windows-videohelper-docker.sh'],
    'Darwin': ['scripts/video-helper-dependencies/macos-policy.py', 'build.sh',
               'scripts/build-macos-vm-release.sh', 'video-helper/CMakeLists.txt'],
}
HEADERS = {'avformat': 'avformat.h', 'avcodec': 'avcodec.h', 'avutil': 'avutil.h',
           'avdevice': 'avdevice.h', 'avfilter': 'avfilter.h',
           'swscale': 'swscale.h', 'swresample': 'swresample.h'}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def contained(path, root):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('Dependency path must be absolute: ' + str(path))
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError('Dependency path escapes prefix: ' + str(path))
    return resolved


def admit_prefix(root, prefix, system, processor, osx_architectures='', mingw=''):
    if system not in PROFILES:
        raise ValueError('No reviewed source profile for target: ' + system)
    architecture = processor.lower()
    if system == 'Darwin':
        arches = osx_architectures.split(';') if osx_architectures else [architecture]
        if arches != ['arm64'] and arches != ['aarch64']:
            raise ValueError('Mac source prefix admits only an arm64 helper')
    elif architecture not in ('x86_64', 'amd64'):
        raise ValueError('Source prefix admits only an x86_64 target')
    if system == 'Windows' and mingw.upper() not in ('1', 'ON', 'TRUE', 'YES'):
        raise ValueError('Windows source prefix requires the reviewed MinGW target')
    root = Path(root).resolve(strict=True)
    prefix = Path(prefix)
    if not prefix.is_absolute():
        raise ValueError('Source prefix must be absolute')
    prefix = prefix.resolve(strict=True)
    profile_id, profile_path, builder_path, layout = PROFILES[system]
    stage = prefix.parents[1] if system == 'Windows' else prefix.parent
    if prefix != stage / layout:
        raise ValueError('Source prefix does not match the staging recipe layout')
    receipt = json.loads((stage / 'DEPENDENCY_BUILD.json').read_text())
    profile = json.loads((root / profile_path).read_text())
    scope = 'Mac-codec-source-prefix-only' if system == 'Darwin' else 'dependency-staging-only'
    if (receipt.get('schemaVersion') != 1 or receipt.get('status') != 'PASS'
            or receipt.get('scope') != scope or receipt.get('profile') != profile_id
            or profile.get('id') != profile_id):
        raise ValueError('Source prefix has no matching PASS profile receipt')
    if system != 'Windows' and receipt.get('prefix') != str(prefix):
        raise ValueError('Receipt identifies a different source prefix')
    if receipt.get('sources') != profile['sources']:
        raise ValueError('Receipt source identities differ from the reviewed profile')
    inputs = receipt.get('inputHashes', {})
    if not {profile_path, builder_path, *RECIPE_INPUTS[system]}.issubset(inputs):
        raise ValueError('Receipt omits a reviewed profile or source recipe input hash')
    for relative, expected in inputs.items():
        path = contained(root / relative, root)
        if sha256(path) != expected:
            raise ValueError('Source recipe hash mismatch: ' + relative)
    if system == 'Windows' and receipt.get('profileSha256') != sha256(root / profile_path):
        raise ValueError('Windows profile hash mismatch')
    if system == 'Darwin':
        if receipt.get('toolchain', {}).get('architecture') != 'arm64':
            raise ValueError('Receipt toolchain is not arm64')
        identity_inputs = {'recipe': inputs, 'toolchain': receipt['toolchain']}
    else:
        packages = (stage / 'evidence/installed-packages.txt').read_text()
        identity_inputs = ({'inputs': inputs, 'packages': packages, 'image': receipt['image']}
                           if system == 'Windows' else
                           {'recipe': inputs, 'installedPackages': packages,
                            'image': receipt['containerImageId']})
    identity = hashlib.sha256(json.dumps(identity_inputs, sort_keys=True).encode()).hexdigest()
    if stage.name != identity or (system == 'Darwin' and receipt.get('identity') != identity):
        raise ValueError('Source prefix cache identity mismatch')
    artifacts = receipt.get('artifactHashes', {})
    # Verify every installed file, not only the seven library files selected later.
    installed = list(prefix.rglob('*'))
    for path in installed:
        contained(path, prefix)
        if path.is_file():
            relative = str(path.relative_to(stage))
            if artifacts.get(relative) != sha256(path):
                raise ValueError('Installed artifact hash mismatch or missing receipt entry: ' + relative)
    for relative in artifacts:
        path = stage / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Receipt artifact path escapes stage')
        if path.is_relative_to(prefix) and not path.is_file():
            raise ValueError('Receipt artifact is missing: ' + relative)
    for name, header in HEADERS.items():
        contained(prefix / 'include' / ('lib' + name) / header, prefix)
        contained(prefix / 'lib/pkgconfig' / ('lib' + name + '.pc'), prefix)
    return prefix


def admit_binding(prefix, includes, library_dirs, pkg_dirs, libraries):
    if not includes or not library_dirs or len(pkg_dirs) != len(HEADERS):
        raise ValueError('Incomplete libav header/library/pkg-config binding')
    for path in [*includes, *library_dirs, *pkg_dirs]:
        if not contained(path, prefix).is_dir():
            raise ValueError('Dependency directory is missing: ' + path)
    found = set()
    for library in libraries:
        path = contained(library, prefix)
        if not path.is_file():
            raise ValueError('Dependency library is missing: ' + library)
        match = re.fullmatch(r'(?:lib)?(avformat|avcodec|avutil|avdevice|avfilter|swscale|swresample)'
                             r'(?:\.so(?:\.\d+)*|(?:\.\d+)*\.dylib|\.dll\.a)', Path(library).name)
        if not match:
            raise ValueError('Unreviewed libav link dependency: ' + library)
        found.add(match[1])
    if found != set(HEADERS):
        raise ValueError('Incomplete shared libav library binding')


def header_probe_command(row):
    """Keep actual compiler/SDK/include flags, replacing only compilation IO."""
    tokens = row.get('arguments') or shlex.split(row['command'])
    languages = []
    for index, token in enumerate(tokens):
        if token == '-x':
            if index + 1 == len(tokens):
                raise ValueError('Missing explicit compiler language in libav header admission')
            languages.append(tokens[index + 1])
        elif token.startswith('-x'):
            languages.append(token[2:])
    if len(languages) > 1 or (languages and languages[0] not in ('c', 'c++', 'objective-c', 'objective-c++')):
        raise ValueError('Unreviewed explicit compiler language in libav header admission')
    command = []
    skip = False
    source_count = 0
    for token in tokens:
        if skip:
            skip = False
            continue
        if token in ('-o', '-MF', '-MT', '-MQ', '-x'):
            skip = True
            continue
        if token in ('-c', '-MD', '-MMD', '-MP'):
            continue
        if token.startswith(('-MF', '-MT', '-MQ', '-x')):
            continue
        if token == row['file'] or (not token.startswith('-') and
                (Path(row['directory']) / token).resolve() == Path(row['file']).resolve()):
            source_count += 1
            continue
        if token.startswith('@') or token in ('&&', ';', '|', '>'):
            raise ValueError('Unreviewed compiler command in libav header admission')
        command.append(token)
    if skip or source_count != 1 or not command:
        raise ValueError('Missing or ambiguous compiler source in libav header admission')
    suffix = Path(row['file']).suffix
    language = languages[0] if languages else {'.mm': 'objective-c++', '.m': 'objective-c', '.c': 'c'}.get(suffix, 'c++')
    return command + ['-E', '-x', language, '-']


def admit_consumed_headers(prefix, preprocessed, artifacts):
    """Line markers identify the headers actually opened by the preprocessor."""
    found = {}
    for escaped in re.findall(r'^#\s+\d+\s+"((?:[^"\\]|\\.)*)"', preprocessed, re.M):
        name = json.loads('"' + escaped + '"')
        path = Path(name)
        if not set(path.parts).intersection('lib' + family for family in HEADERS):
            continue
        path = contained(path, prefix / 'include')
        relative = str(path.relative_to(prefix.parent))
        digest = sha256(path)
        if artifacts.get(relative) != digest:
            raise ValueError('Consumed libav header differs from admitted prefix: ' + str(path))
        found[str(path)] = digest
    for family, public in HEADERS.items():
        for header in (public, 'version.h'):
            expected = str((prefix / 'include' / ('lib' + family) / header).resolve(strict=True))
            if expected not in found:
                raise ValueError('Missing compiler-selected libav header: ' + expected)
    return found


def bound_probe_command(row, cache, build, root):
    if Path(row['directory']).resolve(strict=True) != build:
        raise ValueError('Compiler command directory differs from the generated build')
    source = Path(row['file']).resolve(strict=True)
    if not source.is_relative_to(root) and not source.is_relative_to(build):
        raise ValueError('Compiler source escapes the admitted helper/build roots')
    command = header_probe_command(row)
    language = {'c': 'C', 'c++': 'CXX', 'objective-c': 'OBJC', 'objective-c++': 'OBJCXX'}[command[-2]]
    compiler = cache.get('CMAKE_' + language + '_COMPILER')
    if not compiler:
        raise ValueError('Missing configured compiler identity for libav consumer')
    launcher = cache.get('CMAKE_' + language + '_COMPILER_LAUNCHER', '')
    execution = ([*launcher.split(';')] if launcher else []) + [compiler]
    # Ninja's ExportObjectCompileCommand omits the launcher. Execute that
    # compiler-only database form directly; do not reconstruct a cache-only tool.
    # A represented launcher must still match the complete configured prefix.
    if command[0] != compiler and command[:len(execution)] != execution:
        raise ValueError('Compiler/launcher differs from the generated CMake configuration')
    return command


def admit_compile_headers(prefix, build, root):
    build = Path(build).resolve(strict=True)
    cache_path = build / 'CMakeCache.txt'
    cache = dict(re.findall(r'^([^/#\s:]+):[^=\n]*=(.*)$', cache_path.read_text(), re.M))
    root = Path(root).resolve(strict=True)
    if Path(cache.get('CMAKE_HOME_DIRECTORY', '')).resolve() != root / 'video-helper':
        raise ValueError('Compiler configuration differs from the admitted helper source')
    if Path(cache.get('ARBIT_VIDEO_DEPENDENCY_PREFIX', '')).resolve() != prefix:
        raise ValueError('Compiler configuration differs from the admitted libav prefix')
    consumers = (build / 'arbit-prefix-av-consumers.txt').read_text().splitlines()
    if not consumers or any(not re.fullmatch(r'[A-Za-z0-9_.+-]+', name) for name in consumers):
        raise ValueError('Invalid libav consumer inventory')
    rows = json.loads((build / 'compile_commands.json').read_text())
    artifacts = json.loads((prefix.parent / 'DEPENDENCY_BUILD.json').read_text())['artifactHashes']
    probe = ''.join('#include <lib' + family + '/' + header + '>\n'
                    for family, public in HEADERS.items() for header in (public, 'version.h'))
    admitted = {}
    seen = set()
    for row in rows:
        # CMake identifies each target's compilation in its output object path.
        output = row.get('output') or row.get('command', '')
        target = next((name for name in consumers if 'CMakeFiles/' + name + '.dir/' in output), None)
        if target is None:
            continue
        command = bound_probe_command(row, cache, build, root)
        key = (row['directory'], tuple(command))
        if key not in admitted:
            result = subprocess.run(command, cwd=row['directory'], input=probe,
                                    text=True, capture_output=True)
            if result.returncode:
                raise ValueError('Libav header preprocessor failed for ' + target + ': ' + result.stderr)
            admitted[key] = admit_consumed_headers(prefix, result.stdout, artifacts)
        seen.add(target)
    if seen != set(consumers):
        raise ValueError('Missing actual compiler command for libav consumers: ' + ', '.join(sorted(set(consumers) - seen)))
    receipt = {'scope': 'compiler-selected-libav-headers-only', 'prefix': str(prefix),
               'consumers': sorted(seen), 'compileCommandsSha256': sha256(build / 'compile_commands.json'),
               'cmakeCacheSha256': sha256(cache_path),
               'configuredCompilerLaunchers': {language: cache.get('CMAKE_' + language + '_COMPILER_LAUNCHER', '')
                                              for language in ('C', 'CXX', 'OBJC', 'OBJCXX')},
               'launcherPolicy': 'Execute database argv; omitted launchers are not reconstructed', 'probes': [
                   {'directory': directory, 'command': list(command), 'headers': headers}
                   for (directory, command), headers in admitted.items()]}
    (build / 'arbit-prefix-av-header-admission.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'prefix', 'system', 'processor'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--osx-architectures', default='')
    parser.add_argument('--mingw', default='')
    parser.add_argument('--binding', action='store_true')
    parser.add_argument('--compile-build')
    for name in ('include-dir', 'library-dir', 'pkg-dir', 'library'):
        parser.add_argument('--' + name, action='append', default=[])
    args = parser.parse_args()
    try:
        prefix = admit_prefix(args.root, args.prefix, args.system, args.processor,
                              args.osx_architectures, args.mingw)
        if args.binding:
            admit_binding(prefix, args.include_dir, args.library_dir, args.pkg_dir, args.library)
        if args.compile_build:
            admit_compile_headers(prefix, args.compile_build, args.root)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
