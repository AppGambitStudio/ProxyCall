"""Phase 1 Test: Audio Pipeline — capture from BlackHole, save WAV, play back.

Usage:
    # Make sure audio is playing through BlackHole (e.g. YouTube in browser
    # with system output set to "Meet + Agent" multi-output device)

    python scripts/test_audio_pipeline.py              # capture 5s, save, play back
    python scripts/test_audio_pipeline.py --duration 10  # capture 10s
    python scripts/test_audio_pipeline.py --list-devices  # show all audio devices
"""

import argparse
import asyncio
import logging
import sys
import os
import wave

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.devices import list_devices, find_blackhole, find_device
from src.audio.capture import AudioCapture
from src.audio.playback import AudioPlayback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WAV_OUTPUT = "test_capture.wav"


def show_devices():
    print("\n=== Audio Devices ===\n")
    for d in list_devices():
        direction = []
        if d["max_input_channels"] > 0:
            direction.append(f"IN:{d['max_input_channels']}ch")
        if d["max_output_channels"] > 0:
            direction.append(f"OUT:{d['max_output_channels']}ch")
        print(f"  [{d['index']:2d}] {d['name']:<40} {', '.join(direction)}")
    print()

    try:
        bh = find_blackhole("input")
        print(f"  BlackHole input device index: {bh}")
    except RuntimeError as e:
        print(f"  BlackHole: {e}")


def save_wav(filename: str, audio: np.ndarray, sample_rate: int):
    """Save float32 audio to 16-bit WAV file."""
    # Clip and convert to int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


async def test_capture_and_playback(duration: float):
    print(f"\n=== Audio Pipeline Test ===\n")

    # Step 1: Find devices
    print("Step 1: Finding devices...")
    show_devices()

    capture_idx = find_blackhole("input")
    print(f"  Capture device: BlackHole 2ch (index {capture_idx})")

    # Step 2: Capture audio
    print(f"\nStep 2: Capturing {duration}s of audio from BlackHole...")
    print("  (Make sure audio is playing through BlackHole right now)")

    capture = AudioCapture(device=capture_idx, sample_rate=SAMPLE_RATE)
    await capture.start()

    chunks = []
    samples_needed = int(duration * SAMPLE_RATE)
    samples_captured = 0

    async for chunk in capture.stream():
        chunks.append(chunk)
        samples_captured += len(chunk)
        # Print progress
        elapsed = samples_captured / SAMPLE_RATE
        sys.stdout.write(f"\r  Captured: {elapsed:.1f}s / {duration:.1f}s")
        sys.stdout.flush()
        if samples_captured >= samples_needed:
            break

    await capture.stop()
    print()

    # Combine and check
    audio = np.concatenate(chunks)[:samples_needed]
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"  Samples: {len(audio)}")
    print(f"  Peak amplitude: {peak:.4f}")
    print(f"  RMS level: {rms:.4f}")

    if peak < 0.001:
        print("\n  WARNING: Audio is nearly silent! Check that:")
        print("  1. System output is set to 'Meet + Agent' (or includes BlackHole)")
        print("  2. Something is actually playing (YouTube, music, etc)")
        print("  3. BlackHole is correctly installed")

    # Step 3: Save WAV
    print(f"\nStep 3: Saving to {WAV_OUTPUT}...")
    save_wav(WAV_OUTPUT, audio, SAMPLE_RATE)
    file_size = os.path.getsize(WAV_OUTPUT)
    print(f"  Saved: {WAV_OUTPUT} ({file_size:,} bytes)")

    # Step 4: Also test ring buffer
    print("\nStep 4: Testing ring buffer...")
    recent = capture.get_recent_audio(2.0)
    print(f"  Ring buffer last 2s: {len(recent)} samples")

    # Step 5: Play back
    print(f"\nStep 5: Playing back captured audio...")
    print("  (You should hear what was just captured)")
    playback = AudioPlayback(sample_rate=SAMPLE_RATE)
    await playback.play(audio)
    print("  Playback complete.")

    # Summary
    print(f"\n=== Results ===")
    print(f"  Capture: {'OK' if peak > 0.001 else 'SILENT — check setup'}")
    print(f"  WAV save: OK ({WAV_OUTPUT})")
    print(f"  Ring buffer: OK")
    print(f"  Playback: OK (verify by ear)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Test audio pipeline")
    parser.add_argument("--duration", type=float, default=5.0, help="Capture duration in seconds")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        show_devices()
        return

    asyncio.run(test_capture_and_playback(args.duration))


if __name__ == "__main__":
    main()
