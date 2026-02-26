"""Audio playback to output devices."""

import asyncio
import logging

import numpy as np
import sounddevice as sd

from .devices import find_device

logger = logging.getLogger(__name__)


class AudioPlayback:
    """Non-blocking audio playback."""

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self._playing = False
        self._current_stream: sd.OutputStream | None = None

    async def play(self, audio: np.ndarray, blocking: bool = True):
        """Play audio array.

        Args:
            audio: Float32 audio samples.
            blocking: If True, wait until playback finishes.
        """
        if audio.size == 0:
            return

        # Ensure correct shape
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        self._playing = True
        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(None, self._play_sync, audio)
        finally:
            self._playing = False

    def _play_sync(self, audio: np.ndarray):
        """Synchronous playback (runs in executor)."""
        sd.play(audio, samplerate=self.sample_rate, device=self.device)
        sd.wait()

    async def stop(self):
        """Stop any current playback."""
        sd.stop()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing
