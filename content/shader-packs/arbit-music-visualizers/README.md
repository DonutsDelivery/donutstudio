# DonutStudio music visualizer templates

Seven CC0 templates turn project score and audio data into editable generator clips. They use the same shader contract in the live viewport and offline export.

- Neon Piano Roll and Ember Piano Roll share the falling-score renderer with different curated parameter defaults.
- Spectral History Ribbon uses a persistent ISF pass to scroll the mix spectrum through time.
- Harmonic Constellation plots exact prime exponents, roots, and harmonic links.
- Pitch Ribbons shows score history, lookahead, exact-frequency pitch, and harmonic-master relationships.
- Harmonic Oscilloscope draws one sampled, root-locked phosphor trace from the exact frequency ratios of all sounding notes.
- Resonant Strings maps exact ratios into visible wavelengths. Four root cycles make 5/4, 3/2, and 7/4 notes draw five, six, and seven cycles, with frame-timed motion trails and harmonic-link bridges.

Every ISF input becomes a `clip<id>/gen/<name>` parameter. Users can change it in the Video Editor, automate it, or route musical control to it. The source files are public-domain examples. Curated source executes through exact-byte catalog verification; editing source remains subject to the ordinary Shader Authoring policy.
