"""callsong — generative animal-call synthesis.

A configurable nonlinear syrinx (Mindlin-family limit-cycle oscillator + a
suite of source-filter blocks) rendered in JAX, and a MAP-Elites loop that
illuminates the instrument space using the first two PCA axes of BirdNET
embeddings as behaviour descriptors.
"""

__all__ = ["genome", "synth", "gestures", "features", "birdnet", "archive", "audio_io"]
