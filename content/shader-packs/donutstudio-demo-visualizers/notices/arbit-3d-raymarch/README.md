# DonutStudio 3D Raymarch Worlds

Fifteen raymarched 3D scenes and materials for the Media Machine — the first fully 3D
shader pack. Each scene is a signed-distance-field world marched per pixel,
driven live by your project: sounding notes, exact JI tuning, harmonic links,
the transport clock, and mix analysis (RMS / peak / onset / 64-band spectrum).
Drop one into a shader generator clip and it renders in the viewport and
exports identically.

All shaders use the Block D camera contract (`arbitCameraRay`), so camera
position, target, and field of view stay automatable and mod-routable like any
other clip parameter even though each scene also drifts on its own.

## The scenes

| Shader | What drives it | The idea |
|---|---|---|
| Harmonic Cathedral | notes, beat, audio | Each note raises a glowing pillar; pitch sets height |
| Bass Nebula | bass bands, notes, onsets | Volumetric gas pumped by the low end; notes ignite stars |
| Fractal Bass Temple | beat, bass, notes | Menger temple folding open on every beat |
| Pitch Galaxy | bar phase, tempo, exact tuning | Spiral galaxy with one star per note lifted by cents offset |
| Chromatic Wormhole | 64-band spectrum, RMS, onsets | Infinite twisted tunnel with equalizer ribs in the walls |
| Resonance Ocean | exact JI ratios, bass | Standing-wave ocean; consonance calms it, dissonance foams it |
| Beat Crystal Bloom | downbeats, sustain, pitch class | Crystals grow from a mirror floor while their notes hold |
| Neon Bass Canyon | spectrum veins, bass squeeze | Canyon flythrough with EQ-striped walls and beat lanterns |
| Score Monolith | score rows, harmonic links | Black slab whose face is the score, lasers along links |
| Synaptic Bloom | notes + links | Neural garden; synapse pulses travel master to slave |
| Harmonic Wave Sculpture | exact JI ratios | Helix strands braided by consonance on a mirror pedestal |
| Chromatic Depth Pool | loudness level, note ripples | Liquid pool breathing with the mix; depth = frequency color |
| Gravity Well | full spectrum disc, notes | Black hole accretion disc mapped from the spectrum bands |
| Beat Reactor | bass, spectrum, onsets | Machined containment rig around a volatile plasma core |
| Holo Foil | view angle, loudness, onsets | Embossed trading card with bounded spectral diffraction |

Scenes marked "exact tuning" or "links" spread apart visibly when your tuning
is non-12-EDO — that is the point: microtonal movement becomes visible motion.

## Performance

Raymarching is per-pixel work. All scenes cap their march steps (44–100) and
note loops (max 8–12 active notes), and are tuned to hold 60 fps at 1080p on
integrated GPUs and 4K on discrete cards. If you need headroom, drop clip
resolution before dropping a scene.

## License

CC0-1.0. Original work for the Arbit Media Machine / DonutStudio.
