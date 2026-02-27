"""Phase 2 Test: Live transcription via voxtral.c.

Captures audio from microphone (or BlackHole if available), feeds to
voxtral.c for transcription, displays live transcript in terminal.

Usage:
    # Use built-in mic (simplest — just speak or play Meet through speakers):
    python scripts/test_live_transcription.py --no-vad

    # Specify a device by name:
    python scripts/test_live_transcription.py --no-vad --device mic

    # Lower latency:
    python scripts/test_live_transcription.py --no-vad -I 1.0
"""

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.capture import AudioCapture
from src.audio.devices import find_device, find_blackhole, list_devices
from src.asr.voxtral import VoxtralASR
from src.transcript.buffer import TranscriptBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_device(device_arg: str | None) -> int | None:
    """Resolve --device argument to a device index."""
    if device_arg is None:
        # Auto: try BlackHole, fall back to default mic
        return None

    if device_arg.lower() == "mic":
        idx = find_device("MacBook Pro Microphone", kind="input")
        if idx is None:
            # Fall back to any built-in mic
            idx = find_device("Microphone", kind="input")
        if idx is None:
            print("ERROR: Could not find microphone device")
            sys.exit(1)
        return idx

    if device_arg.lower() == "blackhole":
        return find_blackhole("input")

    # Try as device index
    try:
        return int(device_arg)
    except ValueError:
        pass

    # Try as name substring
    idx = find_device(device_arg, kind="input")
    if idx is None:
        print(f"ERROR: No input device matching '{device_arg}'")
        print("\nAvailable devices:")
        for d in list_devices():
            if d["max_input_channels"] > 0:
                print(f"  {d['index']}: {d['name']}")
        sys.exit(1)
    return idx


async def run(args):
    print("\n=== Live Transcription Test ===\n")

    # Resolve audio device
    device_id = resolve_device(args.device)
    if device_id is not None:
        import sounddevice as sd
        info = sd.query_devices(device_id)
        print(f"Audio device: [{device_id}] {info['name']}")
    else:
        print("Audio device: auto (BlackHole or default mic)")

    # Initialize components
    capture = AudioCapture(device=device_id, sample_rate=16000)
    asr = VoxtralASR(processing_interval=args.interval)
    transcript = TranscriptBuffer()

    vad = None
    if not args.no_vad:
        print("Loading VAD model...")
        from src.asr.vad import VoiceActivityDetector
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

    # Start ASR first (model load takes ~30s)
    transcript.start_session()
    await asr.start()
    print("Waiting for voxtral model to load (~30s)...")
    await asr.wait_ready()
    print("Model loaded! Starting audio capture...\n")
    await capture.start()

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
    parser.add_argument("--device", type=str, default="mic",
                        help="Audio device: 'mic', 'blackhole', device index, or name substring (default: mic)")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
