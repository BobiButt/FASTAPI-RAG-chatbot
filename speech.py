# speech.py
# Offline speech-to-text using faster-whisper.
# The Whisper model runs locally on the CPU (int8) and is downloaded
# only once on first use - after that everything works fully offline.

import io
import threading

from faster_whisper import WhisperModel

# "base.en" is a good balance of speed + accuracy for English questions.
# Bigger models (small.en, medium) are more accurate but slower on CPU.
MODEL_SIZE = "base.en"

# Loading the model is expensive, so we keep a single shared instance.
_model: WhisperModel | None = None
_model_lock = threading.Lock()


def get_model() -> WhisperModel:
    """Load the Whisper model once (thread-safe, lazy)."""
    global _model
    with _model_lock:
        if _model is None:
            _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        return _model


def warmup() -> None:
    """Load the model in a background thread so the first request is fast."""

    def _load() -> None:
        try:
            get_model()
            print(f"STT model '{MODEL_SIZE}' ready (offline speech-to-text).")
        except Exception as e:
            # Never crash the server here; the model retries on first request.
            print(f"STT model warmup failed (will retry on first request): {e}")

    threading.Thread(target=_load, daemon=True).start()


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Convert an audio clip (WAV recorded in the browser) to text.
    Runs 100% locally - no internet needed once the model is downloaded.
    """
    model = get_model()

    # BytesIO is decoded through faster-whisper's bundled PyAV decoder,
    # so browser WAV files work without any system ffmpeg setup.
    segments, _info = model.transcribe(io.BytesIO(audio_bytes), language="en")
    return " ".join(segment.text.strip() for segment in segments).strip()
