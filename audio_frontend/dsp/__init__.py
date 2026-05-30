"""
Pure-Python DSP core for the audio frontend.

Modules:
  geometry    Mic-array geometry helpers (ring layout, steering vectors).
  srp_phat    SRP-PHAT sound-source-localization (azimuth) on a 6-mic ring.
  mvdr        MVDR beamformer in STFT domain.
  aec         SpeexDSP AEC wrapper (with a numpy NLMS fallback).
  dns_cpu     DeepFilterNet CPU denoiser wrapper (with passthrough fallback).
  pipeline    End-to-end: 8ch int16 -> mono float32 clean.
"""
