# Preparing existing shader packs for publication

Use the repository archive builder for all five existing packs. It reads only
blobs from one committed Git revision. Working-tree edits, untracked files and
local private files cannot enter the archives. Review the selected revision
before using it. This prepares publisher inputs and does not upload anything.

```bash
python3 tools/build_shader_pack_archives.py \
  --revision 6b32b54d96e5d7cdaf1545ea78b04aa6b4483f22 \
  --hosting-base https://donutsdelivery.online/store/visuals/ \
  --output-dir .tmp/shader-pack-archives
```

The hosting base is required. The address above is the intended publisher
destination, not evidence that any generated URL exists or is downloadable.
Use a separate output directory for each verification run and compare outputs.

## Archive contents and reproducibility

The builder covers `arbit-essentials`, `arbit-isf-starters`,
`arbit-music-visualizers`, `arbit-3d-raymarch` and `vidvox-isf`.
It retains each pack's exact `pack.json`, `README.md`, `LICENSE` where present,
all `.fs` and `.vs` sources under `shaders/` or `ISF/`, and PNG thumbnails under
`thumbnails/`. Unknown committed payload paths fail closed for publisher review.
Vidvox's MIT LICENSE, source notices and attribution remain intact. First-party
CC0 declarations remain in the unchanged manifests and source documentation.
The builder excludes developer-only `PUBLISHING.md`. The repository-level
`categories.json` and `CONTRIBUTING.md` are not files in these pack directories.

Essentials' unlisted `shaders/apollonian_gasket.fs` is retained byte for byte.
It remains unlisted. Archiving it does not add a manifest entry or admit it to
the compiled shader trust catalog. Existing IDs, legacy digests and source bytes
are unchanged. This tooling does not select a new demo collection.

Every archive has exactly one top-level directory named for the source pack.
The current `SoundfontDownloader` flattens that directory after extraction,
leaving `pack.json` at the installed pack root. Files are sorted by relative
path, stamped 1980-01-01 00:00:00, and stored as Unix regular files with mode
0644. There are no explicit directory entries, comments or extra fields.
Compression is `ZIP_STORED`, deliberately uncompressed to avoid zlib-version
drift. This costs download bytes. Repeated builds with the same committed input
and tool produce identical ZIPs in the tested Python environment. Future Python
ZIP-writer changes are not covered by that claim; compare artifact hashes before
publishing from a different toolchain. Hosting base changes affect JSON only.

## Draft native entries

`draft-native-entries.json` is a receipt envelope with `status`, `inputCommit`,
`inputTree`, `compression`, `entries` and `artifacts`. `entries` contains draft
native-entry fragments. `artifacts` records exact per-file paths, sizes and
SHA-256 values. It is not a live manifest or a Store catalog replacement.

Each fragment copies the existing `id`, `name`, `version`, `license`, `author`,
`format` and `description`. It supplies the actual positive `sizeBytes`, ZIP
`sourceSha256`, revision-qualified `filename`, `directUrl`, `archiveFormat`
set to `zip`, `isZip` set to true, and `customInstallPath` set to
`~/.arbit/shader-packs/<directory>`. The archive digest uses the downloader's
supported `sourceSha256` field. Do not substitute `downloadUrl` or `installPath`.

`ArbitPaths::expandCustomPath` maps the `~/.arbit/` prefix to the canonical
platform data root, including Linux XDG data paths. There is no `media-machine`
segment. `{dataDir}` is documentation notation, not a supported literal path
expansion in this contract.

Pack format and redistribution license do not determine Free/Pro entitlement.
The fragments assign no category, access/price policy, installed/availability
state or app compatibility version. The publisher must review those decisions
and map the supported fields into the existing VPS catalog schema. A pack's
existing `version` is not an app compatibility assertion.

## Publisher review and pending delivery

Review the receipt, source revision, notices, layout, hashes and destination.
Follow [the Store authority and read-back procedure](../../../artifacts/store/README.md)
and [the Store feature guide](../../../artifacts/store/ADDING-STORE-FEATURES.md).
The authoritative catalog remains `/opt/donutsdelivery/store/catalog.json` on
the VPS. Do not create a local catalog beside those guides.

G03 still needs approved upload, temporary-name validation and atomic rename,
VPS catalog authoring and regeneration. Then read back the public ZIPs and both
native manifests. Verify actual ZIP bytes, sizes, SHA-256, content type, magic,
entry paths and the published link report. VPS-local success alone is not proof
of public delivery. Native download, installation, discovery and curated
admission still need acceptance. No publication or network check is part of
this builder. Keep bundled payloads until G03 delivery is established.
