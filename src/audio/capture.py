"""Audio capture from BlackHole or any input device.

Captures at the device's native sample rate (typically 48kHz) and
resamples to the target rate (16kHz) for downstream ASR consumption.
Bridges audio thread to asyncio consumers through an asyncio.Queue.
"""

import asyncio
import collections
import logging
import threading
from typing import AsyncIterator

import numpy as np
import sounddevice as sd

from .devices import find_blackhole

logger = logging.getLogger(__name__)

# 2 minutes of audio at target sample rate
RING_BUFFER_SECONDS = 120


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Simple linear interpolation resampling."""
    if from_rate == to_rate:
        return audio
    ratio = to_rate / from_rate
    n_samples = int(len(audio) * ratio)
    indices = np.arange(n_samples) / ratio
    indices = np.clip(indices, 0, len(audio) - 1)
    idx_floor = indices.astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
    frac = (indices - idx_floor).astype(np.float32)
    return audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac


class AudioCapture:
    """Continuous audio capture with async streaming."""

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 480,  # 30ms at 16kHz
    ):
        self.device = device
        self.sample_rate = sample_rate  # target (output) rate
        self.channels = channels  # output channels (1=mono)
        self.block_size = block_size

        self._native_rate: int = 0  # set at start() from device info
        self._native_channels: int = 0  # set at start() from device info
        self._native_block_size: int = 0

        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[np.ndarray] | None = None

        # Ring buffer: stores last N seconds at target sample rate
        max_blocks = (RING_BUFFER_SECONDS * sample_rate) // block_size
        self._ring_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=max_blocks
        )
        self._lock = threading.Lock()
        self._running = False

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Called from audio thread for each block at native rate/channels."""
        if status:
            logger.warning("Audio capture status: %s", status)

        # Downmix to mono: average all channels
        if indata.ndim > 1 and indata.shape[1] > 1:
            raw = np.mean(indata, axis=1)
        elif indata.ndim > 1:
            raw = indata[:, 0].copy()
        else:
            raw = indata.copy().flatten()

        # Resample from native rate to target rate
        chunk = _resample(raw, self._native_rate, self.sample_rate)

        with self._lock:
            self._ring_buffer.append(chunk)

        # Push to async queue (non-blocking)
        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(self._queue_put, chunk)

    def _queue_put(self, chunk: np.ndarray):
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            # Drop oldest if consumer is slow
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

    async def start(self):
        """Start capturing audio."""
        if self._running:
            return

        if self.device is None:
            try:
                self.device = find_blackhole("input")
                logger.info("Using BlackHole device index: %d", self.device)
            except RuntimeError:
                logger.warning(
                    "BlackHole not found — falling back to default input device (mic)"
                )
                self.device = sd.default.device[0]

        # Query native sample rate and channels from device
        info = sd.query_devices(self.device)
        self._native_rate = int(info["default_samplerate"])
        self._native_channels = int(info["max_input_channels"])
        # Scale block size to native rate (keep same duration per block)
        block_duration = self.block_size / self.sample_rate
        self._native_block_size = int(block_duration * self._native_rate)

        logger.info(
            "Device: %dHz %dch, capturing at native and resampling to %dHz mono",
            self._native_rate,
            self._native_channels,
            self.sample_rate,
        )

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=500)

        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self._native_rate,
            channels=self._native_channels,
            blocksize=self._native_block_size,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        logger.info(
            "Audio capture started: device=%d, native_sr=%d, target_sr=%d, block=%d",
            self.device,
            self._native_rate,
            self.sample_rate,
            self._native_block_size,
        )

    async def stop(self):
        """Stop capturing audio."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        logger.info("Audio capture stopped")

    async def stream(self) -> AsyncIterator[np.ndarray]:
        """Async generator yielding audio chunks as they arrive."""
        if not self._running:
            raise RuntimeError("Capture not started. Call start() first.")
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield chunk
            except asyncio.TimeoutError:
                continue

    def get_recent_audio(self, seconds: float) -> np.ndarray:
        """Return the last N seconds of captured audio from the ring buffer."""
        blocks_needed = int((seconds * self.sample_rate) / self.block_size)
        with self._lock:
            blocks = list(self._ring_buffer)[-blocks_needed:]
        if not blocks:
            return np.array([], dtype=np.float32)
        return np.concatenate(blocks)

    @property
    def is_running(self) -> bool:
        return self._running
