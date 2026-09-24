# calls

Bevy + egui explorer for the `python/calls` syrinx/CPG call model. The model
(synth, Wilson–Cowan gesture, 30-parameter genome) is ported to Rust and runs
live in the audio callback at 48 kHz with the same oversampling as the Python
renderer, so what you hear matches what `run_mapelites.py` scores.

```bash
cargo run --release
```

## Panels

- **Left — genome.** All 30 parameters in physical units, grouped as in
  `genome.py`. Instrument changes are heard immediately; the gesture and the
  spectrogram re-render in the background. Double-click a slider to reset it.
  The checkbox locks a parameter against Randomize / Mutate (σ is the
  normalised-space step, as `genome.mutate`).
- **Centre top — the call.** Spectrogram of the full offline render
  (100 Hz–10 kHz, log), with α (pressure, red) and β (tension, teal) overlaid,
  plus the phonation-onset line and the playhead. Click it to replay.
- **Centre bottom — XY pad + live output.** Drag on the pad to play the current
  instrument by hand (x = α, y = β; the shaded strip is below onset). `hold`
  keeps it sounding after release. The call's own (α, β) path is traced on the
  pad. The live spectrogram shows whatever is coming out of the speakers.
- **Right — sounds.** Save names the current genome and writes
  `presets/<name>.json`; click one to load and play it. Export wav writes the
  48 kHz render to `exports/<name>.wav`.
- **Right — MAP-Elites archive.** Choose any `python/calls/outputs/*/archive.npz`
  and click a cell to load and play that elite.

Keys: `space` play/stop · `m` mutate · `r` randomize · `ctrl+z` / `ctrl+shift+z`
undo/redo.

## Presets from Python

Presets store physical values keyed by parameter name:

```python
import json, numpy as np
from callsong import genome as gm
p = json.load(open("../../rust/calls/presets/foo.json"))["params"]
genome = gm.encode(np.array([p[n] for n in gm.NAMES]))
```
