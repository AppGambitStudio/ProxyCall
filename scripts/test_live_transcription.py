"""Phase 2 Test: Live transcription from BlackHole via voxtral.c.

Captures audio from BlackHole, runs VAD to detect speech, feeds audio
to voxtral.c for transcription, displays live transcript in terminal.

Usage:
    # Set system output to BlackHole (or Meet + Agent), play audio, then:
    python scripts/test_live_transcription.py
    python scripts/test_live_transcription.py --no-vad    # skip VAD, transcribe everything
    python scripts/test_live_transcription.py -I 1.0      # lower latency
"""

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.capture import AudioCapture
from src.asr.voxtral import VoxtralASR
from src.asr.vad import VoiceActivityDetector
from src.transcript.buffer import TranscriptBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run(args):
    print("\n=== Live Transcription Test ===\n")

    # Initialize components
    capture = AudioCapture(sample_rate=16000)
    asr = VoxtralASR(processing_interval=args.interval)
    transcript = TranscriptBuffer()

    vad = None
    if not args.no_vad:
        print("Loading VAD model...")
        vad = VoiceActivityDetector(silence_timeout=1.5)
        vad.load()
        vad.reset()
        print("VAD loaded.\n")

    # Wire up ASR output to transcript buffer
    def on_text(text):
        transcript.add_text(text)
        # Print new text inline
        sys.stdout.write(text)
        sys.stdout.flush()

    asr.on_transcript(on_text)

    # Start components
    transcript.start_session()
    await capture.start()
    await asr.start()

    print("Listening... (Ctrl+C to stop)\n")
    print("-" * 60)

    start_time = time.monotonic()
    chunk_count = 0
    vad_speaking = False

    try:
        async for chunk in capture.stream():
            chunk_count += 1

            # Run VAD
            if vad is not None:
                events = vad.process(chunk)
                for event in events:
                    if event["type"] == "speech_start" and not vad_speaking:
                        vad_speaking = True
                        elapsed = time.monotonic() - start_time
                        sys.stdout.write(f"\n[{elapsed:.1f}s] ")
                        sys.stdout.flush()
                    elif event["type"] == "speech_end" and vad_speaking:
                        vad_speaking = False

            # Always feed audio to ASR (voxtral handles silence fine)
            await asr.feed_audio(chunk)

    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("\nStopping...")

    # Cleanup
    await asr.stop()
    await capture.stop()

    # Flush remaining transcript
    transcript.flush()

    # Summary
    elapsed = time.monotonic() - start_time
    print(f"\n=== Session Summary ===")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Audio chunks processed: {chunk_count}")
    print(f"  Transcript segments: {len(transcript.segments)}")
    print(f"\nFull transcript:")
    print(f"  {transcript.get_all_text()}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Live transcription test")
    parser.add_argument("-I", "--interval", type=float, default=2.0,
                        help="Voxtral processing interval in seconds")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable VAD (transcribe all audio)")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
