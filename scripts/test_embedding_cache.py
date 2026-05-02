#!/usr/bin/env python3
"""Test: measure embedding computation time and verify determinism.

Run from voicebox/ with:
  backend/venv/bin/python scripts/test_embedding_cache.py
"""

import sys
import time
import os

# Suppress transformers noise
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

# Find a reference audio file
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE = None
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith((".wav", ".mp3", ".flac")):
            SAMPLE = os.path.join(root, f)
            break
    if SAMPLE:
        break

if not SAMPLE:
    print("No audio sample found in data/. Provide path as arg.")
    if len(sys.argv) > 1:
        SAMPLE = sys.argv[1]
    else:
        sys.exit(1)

if len(sys.argv) > 1:
    SAMPLE = sys.argv[1]

print(f"Sample: {SAMPLE}")
print(f"Model:  {MODEL}")
print()

# Load model
print("Loading model...")
t0 = time.perf_counter()
model = load(MODEL)
print(f"Model loaded in {time.perf_counter() - t0:.1f}s")
print()

# Load reference audio
ref_audio = load_audio(SAMPLE, sample_rate=24000)
print(f"Ref audio: {ref_audio.shape[0]} samples ({ref_audio.shape[0]/24000:.1f}s)")
print()

# --- Test 1: Speaker Embedding timing + determinism ---
print("=" * 60)
print("TEST 1: Speaker Embedding (ECAPA-TDNN)")
print("=" * 60)

embeddings = []
times = []
for i in range(3):
    t0 = time.perf_counter()
    emb = model.extract_speaker_embedding(ref_audio)
    mx.eval(emb)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    embeddings.append(np.array(emb))
    print(f"  Run {i+1}: {elapsed*1000:.1f}ms  shape={emb.shape}  norm={np.linalg.norm(embeddings[-1]):.4f}")

# Check determinism
for i in range(1, len(embeddings)):
    diff = np.max(np.abs(embeddings[0] - embeddings[i]))
    print(f"  Max diff run1 vs run{i+1}: {diff:.2e}  {'DETERMINISTIC' if diff < 1e-6 else 'STOCHASTIC!'}")
print()

# --- Test 2: Speech Tokenizer Encode timing + determinism ---
print("=" * 60)
print("TEST 2: Speech Tokenizer Encode (ref_codes)")
print("=" * 60)

if model.speech_tokenizer.has_encoder:
    ref_audio_3d = ref_audio[None, None, :]  # [1, 1, samples]
    codes_list = []
    times2 = []
    for i in range(3):
        t0 = time.perf_counter()
        codes = model.speech_tokenizer.encode(ref_audio_3d)
        mx.eval(codes)
        elapsed = time.perf_counter() - t0
        times2.append(elapsed)
        codes_list.append(np.array(codes))
        print(f"  Run {i+1}: {elapsed*1000:.1f}ms  shape={codes.shape}")

    for i in range(1, len(codes_list)):
        diff = np.max(np.abs(codes_list[0] - codes_list[i]))
        print(f"  Max diff run1 vs run{i+1}: {diff:.2e}  {'DETERMINISTIC' if diff < 1e-6 else 'STOCHASTIC!'}")
else:
    print("  SKIPPED — encoder not available (no ICL path for this model)")
    print("  Only speaker embedding is used for voice cloning")
    times2 = [0]
print()

# --- Test 3: Full generate x3 (for comparison) ---
print("=" * 60)
print("TEST 3: Full generate x3 (timing comparison)")
print("=" * 60)

ref_text = "This is a test reference."
test_text = "Hello world, this is a test of the voice cloning system."

gen_times = []
for i in range(3):
    t0 = time.perf_counter()
    for result in model.generate(test_text, ref_audio=ref_audio, ref_text=ref_text):
        elapsed = time.perf_counter() - t0
        dur = result.audio.shape[0] / result.sample_rate
        gen_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.0f}ms  audio={dur:.1f}s  rtf={dur/elapsed:.1f}x")
print()

# Summary
emb_avg = sum(times[1:]) * 1000 / len(times[1:])  # skip warmup
enc_avg = sum(times2[1:]) * 1000 / max(1, len(times2[1:])) if times2[0] > 0 else 0
gen_avg = sum(gen_times[1:]) * 1000 / len(gen_times[1:])  # skip warmup
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Speaker embedding avg:  {emb_avg:.0f}ms  (excl warmup)")
if times2[0] > 0:
    print(f"  Tokenizer encode avg:   {enc_avg:.0f}ms  (excl warmup)")
else:
    print(f"  Tokenizer encode:       N/A (no encoder in this model)")
print(f"  Combined per-call cost: {emb_avg + enc_avg:.0f}ms")
print(f"  Full generate avg:      {gen_avg:.0f}ms  (excl warmup)")
print(f"  Embedding % of total:   {(emb_avg + enc_avg) / gen_avg * 100:.1f}%")
print()
if (emb_avg + enc_avg) < 50:
    print("  VERDICT: Caching embeddings saves <50ms per call.")
    print("           Probably not worth the complexity unless")
    print("           you're doing rapid-fire generation.")
else:
    print("  VERDICT: Caching embeddings saves meaningful time.")
