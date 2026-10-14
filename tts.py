# tts.py
# Offline text-to-speech using Piper neural voices (natural human-like
# speech). The voice model is downloaded ONCE into backend/tts_voices/ and
# then works fully offline. If Piper is unavailable for any reason, we fall
# back to espeak-ng so the speaker button always has something to play.

import io
import os
import subprocess
import tempfile
import threading
import wave

# Piper voice used for bot replies: a clear, natural American male voice.
# Other good English options (download with `python -m piper.download_voices
# <name> --download-dir tts_voices`): en_US-lessac-high (female),
# en_GB-alan-medium (British male), en_GB-alba-medium (British female).
VOICE_NAME = "en_US-ryan-high"
VOICE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tts_voices"
)
MODEL_PATH = os.path.join(VOICE_DIR, f"{VOICE_NAME}.onnx")

# 1.0 = the voice's natural speed; lower = slower, higher = faster.
LENGTH_SCALE = 1.0

# Cap the input so a single request can't tie up the CPU for long.
MAX_CHARS = 4000

# Loaded lazily on first use (the first load takes a moment) and reused
# afterwards. The lock makes concurrent first requests safe.
_voice = None
_voice_lock = threading.Lock()


def _load_voice():
    """Load the Piper voice model exactly once (thread-safe)."""
    global _voice
    with _voice_lock:
        if _voice is None:
            from piper import PiperVoice
            _voice = PiperVoice.load(MODEL_PATH)
        return _voice


def warmup_tts():
    """
    Load the TTS voice model in a background thread so the first
    /speak request is fast. Failures are logged, not raised -
    synthesis will retry (and fall back to espeak-ng) on demand.
    """
    def _run():
        try:
            _load_voice()
            print("TTS voice loaded:", VOICE_NAME)
        except Exception as e:
            print(f"TTS warmup failed (will retry on first request): {e}")

    threading.Thread(target=_run, daemon=True).start()


def _synthesize_piper(text: str) -> bytes:
    """Render text with the Piper neural voice and return WAV bytes."""
    from piper import SynthesisConfig

    voice = _load_voice()
    syn_config = SynthesisConfig(length_scale=LENGTH_SCALE)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        voice.synthesize_wav(text, wav, syn_config)
    return buf.getvalue()


def _synthesize_espeak(text: str) -> bytes:
    """
    Emergency fallback: render text with espeak-ng (robotic but always
    available since it is a system package on this machine).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = os.path.join(tmpdir, "text.txt")
        wav_path = os.path.join(tmpdir, "speech.wav")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        subprocess.run(
            [
                "espeak-ng",
                "-v", "en-us",
                "-s", "160",
                "-f", txt_path,
                "-w", wav_path,
            ],
            check=True,
            capture_output=True,
        )

        with open(wav_path, "rb") as f:
            return f.read()


def synthesize_wav(text: str) -> bytes:
    """
    Render `text` to a WAV audio clip and return the bytes.
    Uses the Piper neural voice; falls back to espeak-ng if needed.
    """
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        raise ValueError("No text to speak.")

    try:
        return _synthesize_piper(text)
    except Exception as e:
        print(f"Piper TTS failed ({e}); falling back to espeak-ng.")
        return _synthesize_espeak(text)
