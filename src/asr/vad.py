"""Voice Activity Detection using Silero VAD.

Detects speech vs silence in audio stream. Emits events when
an utterance ends (silence after speech), signaling the orchestrator
to check intent.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# Silero VAD expects 16kHz mono audio in chunks of 512 samples (32ms)
VAD_CHUNK_SIZE = 512
VAD_SAMPLE_RATE = 16000


class VoiceActivityDetector:
    """Silero VAD wrapper with utterance boundary detection."""

    def __init__(
        self,
        silence_timeout: float = 1.5,  # seconds of silence to mark end of utterance
        speech_threshold: float = 0.5,  # VAD probability threshold
    ):
        self.silence_timeout = silence_timeout
        self.speech_threshold = speech_threshold

        self._model = None
        self._is_speaking = False
        self._silence_start: float | None = None
        self._speech_start: float | None = None
        self._buffer = np.array([], dtype=np.float32)

        self._on_speech_start: list = []
        self._on_speech_end: list = []

    def on_speech_start(self, callback):
        """Register callback for when speech begins."""
        self._on_speech_start.append(callback)

    def on_speech_end(self, callback):
        """Register callback for when speech ends (silence detected)."""
        self._on_speech_end.append(callback)

    def load(self):
        """Load the Silero VAD model."""
        import torch
        logger.info("Loading Silero VAD model...")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model.eval()
        logger.info("Silero VAD loaded")

    def reset(self):
        """Reset VAD state for a new session."""
        if self._model is not None:
            self._model.reset_states()
        self._is_speaking = False
        self._silence_start = None
        self._speech_start = None
        self._buffer = np.array([], dtype=np.float32)

    def process(self, audio: np.ndarray) -> list[dict]:
        """Process audio chunk through VAD.

        Args:
            audio: Float32 mono 16kHz audio chunk.

        Returns:
            List of events: {"type": "speech_start"|"speech_end", "time": float}
        """
        if self._model is None:
            raise RuntimeError("VAD model not loaded. Call load() first.")

        events = []
        self._buffer = np.concatenate([self._buffer, audio])

        # Process in 512-sample chunks as required by Silero
        while len(self._buffer) >= VAD_CHUNK_SIZE:
            chunk = self._buffer[:VAD_CHUNK_SIZE]
            self._buffer = self._buffer[VAD_CHUNK_SIZE:]

            import torch
            tensor = torch.from_numpy(chunk).float()
            prob = self._model(tensor, VAD_SAMPLE_RATE).item()

            now = time.monotonic()

            if prob >= self.speech_threshold:
                # Speech detected
                self._silence_start = None
                if not self._is_speaking:
                    self._is_speaking = True
                    self._speech_start = now
                    events.append({"type": "speech_start", "time": now})
                    for cb in self._on_speech_start:
                        try:
                            cb()
                        except Exception:
                            logger.exception("speech_start callback error")
            else:
                # Silence detected
                if self._is_speaking:
                    if self._silence_start is None:
                        self._silence_start = now
                    elif now - self._silence_start >= self.silence_timeout:
                        # Utterance ended
                        self._is_speaking = False
                        duration = now - (self._speech_start or now)
                        events.append({
                            "type": "speech_end",
                            "time": now,
                            "duration": duration,
                        })
                        self._speech_start = None
                        self._silence_start = None
                        for cb in self._on_speech_end:
                            try:
                                cb()
                            except Exception:
                                logger.exception("speech_end callback error")

        return events

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
