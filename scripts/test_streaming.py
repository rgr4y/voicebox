#!/usr/bin/env python3
"""Test: streaming vs non-streaming generation latency.

Run from voicebox/ with:
  backend/venv/bin/python scripts/test_streaming.py
"""

import sys
import time
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import warnings
warnings.filterwarnings("ignore", message="You are using a model of type")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
try:
    import transformers as _tf
    _tf.logging.set_verbosity_error()
except Exception:
    pass

import numpy as np
import mlx.core as mx
from mlx_audio.tts import load
from mlx_audio.utils import load_audio


MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE = None
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith((".wav", ".mp3", ".flac")):
            SAMPLE = os.path.join(root, f)
            break
    if SAMPLE:
        break

if len(sys.argv) > 1:
    SAMPLE = sys.argv[1]

if not SAMPLE:
    print("No audio sample found. Provide path as arg.")
    sys.exit(1)

print(f"Sample: {SAMPLE}")
print(f"Model:  {MODEL}")
print()

print("Loading model...")
model = load(MODEL)
print("Model loaded.\n")

ref_audio = load_audio(SAMPLE, sample_rate=24000)
ref_text = "This is a reference audio sample for voice cloning."
test_text = "The quick brown fox jumps over the lazy dog. This is a longer sentence to test streaming generation latency."

# --- Non-streaming ---
print("=" * 60)
print("NON-STREAMING (current behavior)")
print("=" * 60)
t0 = time.perf_counter()
total_audio = 0
for result in model.generate(test_text, ref_audio=ref_audio, ref_text=ref_text, stream=False):
    elapsed = time.perf_counter() - t0
    chunk_dur = result.audio.shape[0] / result.sample_rate
    total_audio += chunk_dur
    print(f"  Chunk: {chunk_dur:.2f}s audio at {elapsed*1000:.0f}ms")
print(f"  TOTAL: {total_audio:.2f}s audio, first sound at {elapsed*1000:.0f}ms")
print()

# --- Streaming (2s chunks, default) ---
print("=" * 60)
print("STREAMING (2s chunks, default interval)")
print("=" * 60)
t0 = time.perf_counter()
total_audio = 0
first_chunk_time = None
for result in model.generate(test_text, ref_audio=ref_audio, ref_text=ref_text,
                              stream=True, streaming_interval=2.0):
    elapsed = time.perf_counter() - t0
    chunk_dur = result.audio.shape[0] / result.sample_rate
    total_audio += chunk_dur
    if first_chunk_time is None:
        first_chunk_time = elapsed
    is_final = getattr(result, 'is_final_chunk', False)
    print(f"  Chunk: {chunk_dur:.2f}s audio at {elapsed*1000:.0f}ms {'(FINAL)' if is_final else ''}")
total_time = time.perf_counter() - t0
print(f"  TOTAL: {total_audio:.2f}s audio, first sound at {first_chunk_time*1000:.0f}ms, done at {total_time*1000:.0f}ms")
print()

# --- Streaming (1s chunks, aggressive) ---
print("=" * 60)
print("STREAMING (1s chunks, aggressive)")
print("=" * 60)
t0 = time.perf_counter()
total_audio = 0
first_chunk_time = None
for result in model.generate(test_text, ref_audio=ref_audio, ref_text=ref_text,
                              stream=True, streaming_interval=1.0):
    elapsed = time.perf_counter() - t0
    chunk_dur = result.audio.shape[0] / result.sample_rate
    total_audio += chunk_dur
    if first_chunk_time is None:
        first_chunk_time = elapsed
    is_final = getattr(result, 'is_final_chunk', False)
    print(f"  Chunk: {chunk_dur:.2f}s audio at {elapsed*1000:.0f}ms {'(FINAL)' if is_final else ''}")
total_time = time.perf_counter() - t0
print(f"  TOTAL: {total_audio:.2f}s audio, first sound at {first_chunk_time*1000:.0f}ms, done at {total_time*1000:.0f}ms")
print()

# --- Streaming (0.5s chunks, very aggressive) ---
print("=" * 60)
print("STREAMING (0.5s chunks, very aggressive)")
print("=" * 60)
t0 = time.perf_counter()
total_audio = 0
first_chunk_time = None
for result in model.generate(test_text, ref_audio=ref_audio, ref_text=ref_text,
                              stream=True, streaming_interval=0.5):
    elapsed = time.perf_counter() - t0
    chunk_dur = result.audio.shape[0] / result.sample_rate
    total_audio += chunk_dur
    if first_chunk_time is None:
        first_chunk_time = elapsed
    is_final = getattr(result, 'is_final_chunk', False)
    print(f"  Chunk: {chunk_dur:.2f}s audio at {elapsed*1000:.0f}ms {'(FINAL)' if is_final else ''}")
total_time = time.perf_counter() - t0
print(f"  TOTAL: {total_audio:.2f}s audio, first sound at {first_chunk_time*1000:.0f}ms, done at {total_time*1000:.0f}ms")
