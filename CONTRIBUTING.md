# Contributing

This repository accepts public documentation, examples, and scripting API
requests for Arbit.

## Good Contributions

- Small Lua scripts that solve one clear problem.
- Documentation fixes.
- API request issues with concrete examples.
- Reproducible scripting bugs.

## Script Guidelines

- Put scripts in a category folder under `examples/`.
- Use a first-line comment that explains the script.
- Prefer named-parameter calls, e.g. `arbit.move_note({ ... })`.
- Print a short summary when the script completes.
- Avoid deleting or overwriting user work unless the filename makes that
  behavior obvious.
- Do not assume Pro-only APIs unless the example says so.

## Do Not Submit

- Arbit application source code.
- Reverse-engineered private internals.
- License keys, license-server behavior, or auth bypasses.
- Proprietary sound libraries, voicebanks, samples, or model files.
- Large binary exports.

## Review Expectations

Examples should be readable by musicians who are not professional programmers.
Prefer boring Lua over clever Lua.
