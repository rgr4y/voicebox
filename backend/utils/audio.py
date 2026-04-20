"""
Audio processing utilities.

Includes EBU R128 loudness normalization with true-peak limiting,
matching broadcast standards. Uses pyloudnorm for LUFS measurement
and normalization — pure Python, no ffmpeg dependency.
"""

import logging
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def normalize_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    target_lufs: float = -16.0,
    true_peak_limit_db: float = -2.0,
) -> np.ndarray:
    """
    Normalize audio to target loudness (EBU R128) with true-peak limiting.

    Matches the behavior of ffmpeg's loudnorm filter:
        loudnorm=I=-16:TP=-2:LRA=11,alimiter=limit=-2dB

    Falls back to simple RMS normalization if pyloudnorm is unavailable
    or audio is too short for LUFS measurement.

    Args:
        audio: Input audio array (mono, float32)
        sample_rate: Audio sample rate
        target_lufs: Target integrated loudness in LUFS (default: -16)
        true_peak_limit_db: True-peak ceiling in dBTP (default: -2)

    Returns:
        Normalized audio array
    """
    import warnings
    audio = audio.astype(np.float32)

    if len(audio) == 0:
        return audio

    # True-peak limit as linear amplitude
    peak_limit = 10 ** (true_peak_limit_db / 20)

    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)

        # pyloudnorm requires at least 0.4s of audio for LUFS measurement
        min_samples = int(sample_rate * 0.4)
        if len(audio) < min_samples:
            logger.debug("Audio too short for LUFS normalization, using RMS fallback")
            return _normalize_rms(audio, target_db=target_lufs, peak_limit=peak_limit)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            current_lufs = meter.integrated_loudness(audio)

        # If audio is essentially silent, LUFS returns -inf
        if not np.isfinite(current_lufs) or current_lufs < -70:
            logger.debug("Audio too quiet for LUFS normalization (%.1f LUFS)", current_lufs)
            return audio

        # Apply loudness normalization (suppress clipping warnings — we clip intentionally below)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Possible clipping.*")
            audio = pyln.normalize.loudness(audio, current_lufs, target_lufs)

        # True-peak limiting via clipping (simple but effective for TTS output)
        audio = np.clip(audio, -peak_limit, peak_limit)

        return audio

    except ImportError:
        logger.warning("pyloudnorm not installed, falling back to RMS normalization")
        return _normalize_rms(audio, target_db=target_lufs, peak_limit=peak_limit)


def _normalize_rms(
    audio: np.ndarray,
    target_db: float = -20.0,
    peak_limit: float = 0.85,
) -> np.ndarray:
    """
    Simple RMS-based normalization fallback.

    Used when pyloudnorm is unavailable or audio is too short for LUFS.
    """
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 10 ** (target_db / 20)

    if rms > 0:
        gain = target_rms / rms
        audio = audio * gain

    audio = np.clip(audio, -peak_limit, peak_limit)
    return audio


def load_audio(
    path: str,
    sample_rate: int = 24000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with normalization.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return audio, sr


def trim_leading_silence(
    audio: np.ndarray,
    sample_rate: int = 24000,
    threshold_rms: float = 0.03,
    window_ms: float = 30.0,
    pad_ms: float = 30.0,
) -> np.ndarray:
    """
    Remove leading silence from generated audio.

    Uses RMS energy over sliding windows to find speech onset,
    then trims to pad_ms before that point.

    Args:
        audio: Audio samples (1D float array)
        sample_rate: Sample rate in Hz
        threshold_rms: RMS level that counts as speech
        window_ms: Analysis window size in ms
        pad_ms: Silence to preserve before speech onset

    Returns:
        Trimmed audio array
    """
    if len(audio) == 0:
        return audio

    window = max(1, int(sample_rate * window_ms / 1000))
    hop = window // 2

    for start in range(0, len(audio) - window, hop):
        chunk = audio[start:start + window]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > threshold_rms:
            pad_samples = int(sample_rate * pad_ms / 1000)
            cut = max(0, start - pad_samples)
            if cut > 0:
                trimmed_ms = cut * 1000 / sample_rate
                logger.debug(f"Trimmed {trimmed_ms:.0f}ms leading silence")
            return audio[cut:]

    return audio


def save_audio(
    audio: np.ndarray,
    path: str,
    sample_rate: int = 24000,
    normalize: bool = True,
) -> None:
    """
    Save audio file, optionally with loudness normalization.

    When normalize=True (the default), applies EBU R128 loudness
    normalization to -16 LUFS with -2 dBTP true-peak limiting before
    saving. This ensures consistent output volume across generations.

    Args:
        audio: Audio array
        path: Output path
        sample_rate: Sample rate
        normalize: Apply loudness normalization before saving (default: True)
    """
    audio = trim_leading_silence(audio, sample_rate=sample_rate)
    if normalize:
        audio = normalize_audio(audio, sample_rate=sample_rate)
    sf.write(path, audio, sample_rate)


def trim_tts_output(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    min_silence_ms: int = 200,
    max_internal_silence_ms: int = 1000,
    fade_ms: int = 30,
) -> np.ndarray:
    """
    Trim trailing silence and post-silence hallucination from TTS output.

    Chatterbox sometimes produces ``[speech][silence][hallucinated noise]``.
    This detects internal silence gaps longer than *max_internal_silence_ms*
    and cuts the audio at that boundary, then trims trailing silence and
    applies a short cosine fade-out.

    Args:
        audio: Input audio array (mono float32)
        sample_rate: Sample rate in Hz
        frame_ms: Frame size for RMS energy calculation
        silence_threshold_db: dB threshold below which a frame is silence
        min_silence_ms: Minimum trailing silence to keep
        max_internal_silence_ms: Cut after any silence gap longer than this
        fade_ms: Cosine fade-out duration in ms

    Returns:
        Trimmed audio array
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return audio

    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)

    # Compute per-frame RMS
    rms = np.array(
        [
            np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ]
    )
    is_speech = rms >= threshold_linear

    # Find first speech frame
    first_speech = 0
    for i, s in enumerate(is_speech):
        if s:
            first_speech = max(0, i - 1)  # keep 1 frame padding
            break

    # Walk forward from first speech; cut at long internal silence gaps
    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    consecutive_silence = 0
    cut_frame = n_frames

    for i in range(first_speech, n_frames):
        if is_speech[i]:
            consecutive_silence = 0
        else:
            consecutive_silence += 1
            if consecutive_silence >= max_silence_frames:
                cut_frame = i - consecutive_silence + 1
                break

    # Trim trailing silence from the cut point
    min_silence_frames = int(min_silence_ms / frame_ms)
    end_frame = cut_frame
    while end_frame > first_speech and not is_speech[end_frame - 1]:
        end_frame -= 1
    # Keep a short tail
    end_frame = min(end_frame + min_silence_frames, cut_frame)

    # Convert frames back to samples
    start_sample = first_speech * frame_len
    end_sample = min(end_frame * frame_len, len(audio))

    trimmed = audio[start_sample:end_sample].copy()

    # Cosine fade-out
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and len(trimmed) > fade_samples:
        fade = np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        trimmed[-fade_samples:] *= fade

    return trimmed


def preprocess_reference_audio(
    audio: np.ndarray,
    sample_rate: int,
    peak_target: float = 0.95,
    trim_top_db: float = 40.0,
    edge_padding_ms: int = 100,
) -> np.ndarray:
    """
    Clean up a reference-audio sample before validation/storage.

    Removes DC offset, trims leading/trailing silence, and caps the peak so a
    slightly-hot recording doesn't get rejected downstream as "clipping". The
    goal is to accept reasonable real-world recordings — not to repair badly
    distorted ones. True clipping artifacts inside the waveform can't be
    recovered by peak scaling and will still sound bad.

    Args:
        audio: Mono audio array.
        sample_rate: Sample rate of ``audio`` in Hz.
        peak_target: Peak amplitude cap in [0, 1]. Applied only if the input
            peak exceeds this value.
        trim_top_db: Silence threshold for edge trimming, in dB below peak.
            40 dB sits below normal speech dynamic range (≈30 dB) so soft
            trailing syllables are preserved, while still catching obvious
            leading/trailing silence. Lower values are more aggressive;
            librosa's own default is 60.
        edge_padding_ms: Milliseconds of padding to add back at each edge
            *only if* trimming shortened the waveform, so TTS engines have a
            brief silence to anchor on without ever making the output longer
            than the input.

    Returns:
        Preprocessed audio array (float32).
    """
    audio = audio.astype(np.float32, copy=False)

    if audio.size == 0:
        return audio

    audio = audio - float(np.mean(audio))

    trimmed, _ = librosa.effects.trim(audio, top_db=trim_top_db)
    if 0 < trimmed.size < audio.size:
        pad_each = int(sample_rate * edge_padding_ms / 1000)
        # Never pad past the original length — for near-max-duration uploads
        # an unconditional pad would push them over the 30 s ceiling and
        # trigger a spurious "too long" rejection.
        headroom = (audio.size - trimmed.size) // 2
        pad = min(pad_each, max(headroom, 0))
        if pad > 0:
            trimmed = np.pad(trimmed, (pad, pad), mode="constant")
        audio = trimmed

    peak = float(np.abs(audio).max())
    if peak > peak_target and peak > 0:
        audio = audio * (peak_target / peak)

    return audio


def validate_and_load_reference_audio(
    audio_path: str,
    min_duration: float = 2.0,
    max_duration: float = 30.0,
    trim_threshold: float = 45.0,
    min_rms: float = 0.01,
) -> Tuple[bool, Optional[str], Optional[np.ndarray], Optional[int]]:
    """
    Validate and load reference audio in a single pass.

    Applies preprocessing before checks so slightly-hot recordings pass
    without changing existing loudness-normalization behavior elsewhere.
    """
    try:
        audio, sr = load_audio(audio_path)
        audio = preprocess_reference_audio(audio, sr)
        duration = len(audio) / sr

        if duration < min_duration:
            return False, f"Audio too short ({duration:.1f}s, minimum {min_duration}s)", None, None

        if duration > max_duration:
            if duration <= trim_threshold:
                max_samples = int(max_duration * sr)
                audio = audio[:max_samples]
                duration = max_duration
            else:
                return (
                    False,
                    f"Audio too long ({duration:.1f}s, maximum {max_duration}s). "
                    f"Clips up to {trim_threshold:.0f}s are auto-trimmed.",
                    None,
                    None,
                )

        rms = np.sqrt(np.mean(audio**2))
        if rms < min_rms:
            return False, "Audio is too quiet or silent", None, None

        return True, None, audio, sr
    except Exception as e:
        return False, f"Error processing audio: {str(e)}", None, None


def validate_reference_audio(
    audio_path: str,
    min_duration: float = 2.0,
    max_duration: float = 30.0,
    trim_threshold: float = 45.0,
    min_rms: float = 0.01,
) -> Tuple[bool, Optional[str]]:
    """
    Validate reference audio for voice cloning.

    Args:
        audio_path: Path to audio file (will be overwritten with processed version)
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds (inclusive - 30.0s is allowed)
        trim_threshold: Auto-trim clips up to this length to max_duration.
            Clips longer than this are rejected outright.
        min_rms: Minimum RMS level (below this = silence)

    Returns:
        Tuple of (is_valid, error_message)
    """
    ok, err, audio, sr = validate_and_load_reference_audio(
        audio_path=audio_path,
        min_duration=min_duration,
        max_duration=max_duration,
        trim_threshold=trim_threshold,
        min_rms=min_rms,
    )
    if not ok or audio is None or sr is None:
        return ok, err

    # Keep existing behavior: normalize + overwrite validated reference input.
    audio = normalize_audio(audio, sample_rate=sr)
    sf.write(audio_path, audio, sr)
    return True, None


# Keep old name as alias for backward compatibility
validate_and_normalize_reference_audio = validate_reference_audio
