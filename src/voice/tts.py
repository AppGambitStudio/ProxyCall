"""TTS client for VoiceBox — generates speech from text using cloned voice."""

import json
import logging
import time
import urllib.request
import wave

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:17493"


class VoiceBoxTTS:
    """Synchronous TTS client for VoiceBox API.

    Uses urllib (sync) to avoid aiohttp event loop conflicts.
    Called via run_in_executor from the orchestrator.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        profile_id: str = "",
        language: str = "en",
    ):
        self.base_url = base_url.rstrip("/")
        self.profile_id = profile_id
        self.language = language

    async def start(self):
        """Auto-detect profile if needed."""
        if not self.profile_id:
            self._detect_profile()

    def _detect_profile(self):
        """Fetch profiles from VoiceBox and use the first one."""
        req = urllib.request.Request(f"{self.base_url}/profiles")
        with urllib.request.urlopen(req, timeout=5) as resp:
            profiles = json.loads(resp.read())
        if profiles:
            self.profile_id = profiles[0]["id"]
            logger.info(
                "Auto-detected voice profile: %s (%s)",
                profiles[0]["name"],
                self.profile_id,
            )
        else:
            raise RuntimeError("No voice profiles found in VoiceBox")

    def synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Generate speech audio from text (synchronous).

        Returns:
            Tuple of (audio_float32, sample_rate).
        """
        start = time.monotonic()

        payload = json.dumps({
            "profile_id": self.profile_id,
            "text": text,
            "language": self.language,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())

        audio_path = result["audio_path"]
        duration = result.get("duration", 0)

        # Load the generated WAV file
        audio, sample_rate = self._load_wav(audio_path)

        elapsed = time.monotonic() - start
        logger.info(
            "TTS synthesized %.2fs audio in %.2fs (%.1fx realtime): %s",
            duration,
            elapsed,
            duration / elapsed if elapsed > 0 else 0,
            text[:60],
        )

        return audio, sample_rate

    def _load_wav(self, path: str) -> tuple[np.ndarray, int]:
        """Load WAV file as float32 numpy array."""
        with wave.open(path, "r") as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sample_rate

    async def stop(self):
        """No-op for sync client."""
        pass
