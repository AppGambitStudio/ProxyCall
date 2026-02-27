"""Voxtral ASR wrapper — runs voxtral.c as subprocess with --stdin.

Feeds raw s16le 16kHz mono audio to stdin, reads transcript tokens from stdout.
"""

import asyncio
import logging
import signal
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BINARY = "./vendor/voxtral.c/voxtral"
DEFAULT_MODEL = "./vendor/voxtral.c/voxtral-model"


class VoxtralASR:
    """Async wrapper around voxtral.c subprocess."""

    def __init__(
        self,
        binary_path: str = DEFAULT_BINARY,
        model_path: str = DEFAULT_MODEL,
        processing_interval: float = 2.0,
    ):
        self.binary_path = str(Path(binary_path).resolve())
        self.model_path = str(Path(model_path).resolve())
        self.processing_interval = processing_interval

        self._process: asyncio.subprocess.Process | None = None
        self._running = False
        self._on_transcript: list = []  # callbacks: (text: str) -> None
        self._write_buffer = bytearray()  # batch small chunks before writing
        self._write_threshold = 16000 * 2  # 1 second of s16le audio (16kHz * 2 bytes)
        self._warming_up = False
        self._model_ready = asyncio.Event()

    def on_transcript(self, callback):
        """Register a callback for new transcript text."""
        self._on_transcript.append(callback)

    async def start(self):
        """Launch voxtral.c subprocess."""
        if self._running:
            return

        cmd = [
            self.binary_path,
            "-d", self.model_path,
            "--stdin",
            "-I", str(self.processing_interval),
        ]
        logger.info("Starting voxtral: %s", " ".join(cmd))

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._write_buffer.clear()  # clear stale audio from before stop
        self._model_ready.clear()
        self._warming_up = True  # ignore audio during model load
        self._running = True

        # Start reading stdout in background
        asyncio.create_task(self._read_stdout())
        asyncio.create_task(self._read_stderr())

        logger.info("Voxtral ASR started (pid=%d), waiting for model load...", self._process.pid)

    async def wait_ready(self):
        """Wait until the model is fully loaded. Call after start()."""
        await self._model_ready.wait()
        logger.info("Voxtral ASR ready")

    async def _read_stdout(self):
        """Read transcript tokens from stdout."""
        try:
            while self._running and self._process:
                data = await self._process.stdout.read(4096)
                if not data:
                    break
                text = data.decode("utf-8", errors="replace")
                if text.strip():
                    logger.debug("ASR output: %r", text)
                    for cb in self._on_transcript:
                        try:
                            cb(text)
                        except Exception:
                            logger.exception("Transcript callback error")
        except Exception:
            if self._running:
                logger.exception("Error reading voxtral stdout")

    async def _read_stderr(self):
        """Log stderr output from voxtral, detect model loaded."""
        try:
            while self._running and self._process:
                data = await self._process.stderr.read(4096)
                if not data:
                    break
                text = data.decode("utf-8", errors="replace").strip()
                if text:
                    logger.debug("Voxtral stderr: %s", text)
                    if "Model loaded" in text:
                        self._warming_up = False
                        self._model_ready.set()
        except Exception:
            pass

    async def feed_audio(self, audio: np.ndarray):
        """Feed float32 mono 16kHz audio to voxtral stdin.

        Batches small chunks into ~1s writes to reduce pipe overhead.
        Converts to s16le format expected by voxtral --stdin.
        """
        if not self._running or not self._process or self._process.stdin.is_closing():
            return
        if self._warming_up:
            return  # discard audio during model load

        # Convert float32 [-1, 1] to int16
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        self._write_buffer.extend(audio_int16.tobytes())

        # Flush when we have enough data (~1s of audio)
        if len(self._write_buffer) >= self._write_threshold:
            await self._flush_buffer()

    async def _flush_buffer(self):
        """Write buffered audio to voxtral stdin."""
        if not self._write_buffer or not self._process or self._process.stdin.is_closing():
            return
        try:
            self._process.stdin.write(bytes(self._write_buffer))
            self._write_buffer.clear()
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Voxtral stdin pipe broken")
            self._running = False

    async def pause(self):
        """Pause voxtral subprocess (SIGSTOP) to free GPU for TTS."""
        if self._process and self._running:
            try:
                self._process.send_signal(signal.SIGSTOP)
                logger.info("Voxtral ASR paused (SIGSTOP)")
            except (ProcessLookupError, OSError):
                pass

    async def resume(self):
        """Resume voxtral subprocess (SIGCONT) after TTS."""
        if self._process and self._running:
            try:
                self._process.send_signal(signal.SIGCONT)
                logger.info("Voxtral ASR resumed (SIGCONT)")
            except (ProcessLookupError, OSError):
                pass

    async def stop(self):
        """Stop voxtral subprocess."""
        self._running = False
        if self._process:
            try:
                self._process.stdin.close()
                await self._process.stdin.wait_closed()
            except Exception:
                pass
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            logger.info("Voxtral ASR stopped")
            self._process = None

    @property
    def is_running(self) -> bool:
        return self._running
