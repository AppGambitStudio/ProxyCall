"""Rolling transcript buffer with timestamped segments."""

import time
from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    timestamp: float  # seconds since session start
    text: str
    speaker: str | None = None  # Phase 4: diarization
    confidence: float = 1.0


class TranscriptBuffer:
    """Accumulates transcript text into timestamped segments."""

    def __init__(self):
        self._segments: list[TranscriptSegment] = []
        self._start_time: float = 0.0
        self._current_text: str = ""  # accumulates tokens until flush

    def start_session(self):
        """Mark the start of a transcription session."""
        self._start_time = time.monotonic()
        self._segments.clear()
        self._current_text = ""

    def add_text(self, text: str):
        """Add raw text from ASR (may be partial tokens).

        Accumulates text and creates segments on sentence boundaries.
        """
        self._current_text += text

        # Split on sentence boundaries
        while True:
            # Look for sentence-ending punctuation followed by space or end
            for i, ch in enumerate(self._current_text):
                if ch in ".!?" and (i + 1 >= len(self._current_text) or self._current_text[i + 1] == " "):
                    sentence = self._current_text[: i + 1].strip()
                    self._current_text = self._current_text[i + 1 :].lstrip()
                    if sentence:
                        self._segments.append(
                            TranscriptSegment(
                                timestamp=time.monotonic() - self._start_time,
                                text=sentence,
                            )
                        )
                    break
            else:
                break

    def flush(self):
        """Force-flush any accumulated text as a segment."""
        text = self._current_text.strip()
        if text:
            self._segments.append(
                TranscriptSegment(
                    timestamp=time.monotonic() - self._start_time,
                    text=text,
                )
            )
            self._current_text = ""

    def get_recent(self, seconds: float) -> list[TranscriptSegment]:
        """Return segments from the last N seconds."""
        cutoff = time.monotonic() - self._start_time - seconds
        return [s for s in self._segments if s.timestamp >= cutoff]

    def get_recent_text(self, seconds: float) -> str:
        """Return concatenated text from the last N seconds."""
        segments = self.get_recent(seconds)
        parts = [s.text for s in segments]
        # Include any pending text
        if self._current_text.strip():
            parts.append(self._current_text.strip())
        return " ".join(parts)

    def get_all_text(self) -> str:
        """Return full transcript as text."""
        parts = [s.text for s in self._segments]
        if self._current_text.strip():
            parts.append(self._current_text.strip())
        return " ".join(parts)

    @property
    def segments(self) -> list[TranscriptSegment]:
        return list(self._segments)

    @property
    def pending_text(self) -> str:
        return self._current_text
