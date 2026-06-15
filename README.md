<div align="center">

# Arbit

### Compose with harmonic relationships, not fixed scales

**A free harmonic composition workstation where every note is defined by its relationship.**
Use it standalone or inside your DAW — then upgrade to Arbit Pro for sampling, vocals, and audio transformation.

[**⬇ Download Arbit**](https://github.com/DonutsDelivery/arbit/releases/latest) ·
[Website](https://donutsdelivery.online/arbit) ·
[Arbit Pro](https://donutsdelivery.online/arbit/buy) ·
[Developer docs ↓](#for-developers--ai-agents)

</div>

---

## Download

**Latest release → [github.com/DonutsDelivery/arbit/releases/latest](https://github.com/DonutsDelivery/arbit/releases/latest)**

| Platform | Download | Notes |
|---|---|---|
| **Windows** 10+ | [Arbit-Windows-Setup.exe](https://github.com/DonutsDelivery/arbit/releases/latest/download/Arbit-Windows-Setup.exe) | Installer (VST3 · CLAP · standalone) |
| | [Arbit-Windows.zip](https://github.com/DonutsDelivery/arbit/releases/latest/download/Arbit-Windows.zip) | Portable |
| **macOS** 11+ <br>(Intel & Apple Silicon) | [Arbit-macOS.zip](https://github.com/DonutsDelivery/arbit/releases/latest/download/Arbit-macOS.zip) | |
| **Linux** | [Arbit-Linux-Installer.sh](https://github.com/DonutsDelivery/arbit/releases/latest/download/Arbit-Linux-Installer.sh) | Installer |
| | [Arbit-x86_64.AppImage](https://github.com/DonutsDelivery/arbit/releases/latest/download/Arbit-x86_64.AppImage) | Portable |
| | [Arbit-Linux.zip](https://github.com/DonutsDelivery/arbit/releases/latest/download/Arbit-Linux.zip) | Portable |

<details>
<summary><b>Install notes</b></summary>

- **Windows** — Run `Arbit-Windows-Setup.exe`. Choose VST3, CLAP, and standalone components in the wizard, then restart your DAW.
- **macOS** — Extract `Arbit-macOS.zip`, open Terminal in that folder, and run `chmod +x install-macos.sh && ./install-macos.sh`. If macOS still warns, right-click Arbit → **Open**.
- **Linux** — Run `chmod +x Arbit-Linux-Installer.sh && ./Arbit-Linux-Installer.sh`. AppImage users: `chmod +x Arbit-x86_64.AppImage`, then run it.

</details>

Arbit is in **open beta**. Downloading is free — the Arbit Free unlock happens inside the app on first launch (email), or enter an Arbit Pro license code. The same `.arbit` project files work across all three platforms.

---

## Why Arbit

Every other piano roll treats notes as fixed grid positions, each locked to one frequency. **Arbit treats notes as relationships.**

- **Every note's frequency comes from its link chain** — pure ratios from the harmonic series, any interval, any harmonic.
- **Harmonic context *is* the note** — move a master note and everything linked to it follows.
- **Linking is the tuning** — one automatic system, not manual per-note microtuning.
- **12-TET is just the starting point** — just intonation, EDO, and arbitrary ratios are all first-class.

## What's inside — Arbit Free

**Compose**
- Harmonic piano roll with just-intonation **link chains**
- JI-aware chord detection, auto-link from chords, progression-aware retuning
- 30+ note transforms — quantize, humanize, retrograde, and more
- Clip arranger with audio + MIDI clips, automation lanes, and tempo maps

**Sound**
- **Sybil-16** virtual-analog synth and **OP-7** FM synth (20,800+ presets)
- SF2 / SFZ soundfonts — 1,300+ instruments, GM-compatible, microtonal playback
- 480+ **Airwindows** effects plus **Donut Devices** clean utility effects
- In-app store for instruments and content

**Produce**
- A full standalone DAW — clip arranger, mixing, and stem export
- **VST3 / CLAP plugin mode** inside every major DAW
- MIDI import / export with microtuning preserved

## Arbit Pro

The transformation and media tier:

- Per-note audio **sampling** and audio-to-note transcription
- **Harmonic pitch correction** and vocal resynthesis
- **DiffSinger** and **UTAU** vocals
- A beat-synced **video editor** with **score-reactive visuals** and a music → image **modulation matrix**

→ [Compare Arbit and Pro](https://donutsdelivery.online/arbit/buy)

## Works in your DAW

VST3 in **Bitwig, Reaper, Ableton Live, FL Studio, Studio One, and Cubase**; CLAP in **Bitwig and Reaper**. Or run it fully standalone — no DAW required.

See screenshots and demos at **[donutsdelivery.online/arbit](https://donutsdelivery.online/arbit)**.

---

## For developers & AI agents

Arbit is scriptable and agent-controllable. This repository holds the public developer resources:

- **[`docs/lua-scripting.md`](docs/lua-scripting.md)** — Lua scripting overview and install location
- **[`docs/api-reference.md`](docs/api-reference.md)** — the public `arbit` scripting API surface
- **[`docs/skinning.md`](docs/skinning.md)** — skin format overview (preview)
- **[`examples/`](examples/)** — ready-to-edit scripts (transforms, generators, analysis) and skin packs
- **[`skills/`](skills/)** — agent skills for Claude Code, Cursor, and Codex

```lua
local notes = arbit.get_notes()
for _, note in ipairs(notes) do
  if note.velocity < 40 then
    arbit.set_note_velocity({ noteId = note.id, velocity = 80 })
  end
end
```

AI coding agents can drive Arbit directly through its MCP server or the port-9900 command server — import, beat-sync, effect, export, and then verify the result. See [`skills/README.md`](skills/README.md).

---

## About

Arbit is **closed source** — built by one person, in Denmark. This repository contains the public installers (Releases), documentation, examples, and integration resources. It does **not** contain the application source, build system, license server, DSP engines, or model integrations.

- 🌐 [donutsdelivery.online/arbit](https://donutsdelivery.online/arbit)
- 🛒 [Arbit Pro](https://donutsdelivery.online/arbit/buy)

## License

Documentation and example scripts in this repository are MIT licensed unless a file states otherwise. The Arbit application itself is proprietary; see the in-app EULA.
