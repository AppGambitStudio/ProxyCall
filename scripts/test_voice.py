"""Phase 4 Test: Voice clone TTS — generate speech and play back.

Usage:
    python scripts/test_voice.py
    python scripts/test_voice.py --text "Custom text to speak"
"""

import argparse
import asyncio
import logging
import os
import sys

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice.tts import VoiceBoxTTS
from src.voice.profile import list_profiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TEST_PHRASES = [
    "I think we should go with token-based rate limiting. It gives us better per-user control and scales horizontally.",
    "Let me get back to you on that, I want to double-check the numbers first.",
    "Yeah, the auth module is complete and all integration tests are passing. We're in good shape.",
]


async def run(args):
    print("\n=== Voice Clone TTS Test ===\n")

    # List profiles
    profiles = await list_profiles()
    print(f"Available profiles: {len(profiles)}")
    for p in profiles:
        print(f"  - {p['name']} ({p['id'][:8]}...)")

    # Init TTS
    tts = VoiceBoxTTS()
    await tts.start()
    print(f"\nUsing profile: {tts.profile_id[:8]}...\n")

    # Find speakers
    devs = sd.query_devices()
    spk = next(i for i, d in enumerate(devs) if "MacBook Pro Speakers" in d["name"])

    phrases = [args.text] if args.text else TEST_PHRASES

    for i, text in enumerate(phrases):
        print(f"--- Phrase {i+1} ---")
        print(f"  Text: {text}")

        audio, sr = await tts.synthesize(text)
        duration = len(audio) / sr
        print(f"  Audio: {duration:.2f}s @ {sr}Hz")
        print(f"  Playing...")

        sd.play(audio, samplerate=sr, device=spk)
        sd.wait()
        print(f"  Done.\n")

    await tts.stop()
    print("=== Test Complete ===\n")


def main():
    parser = argparse.ArgumentParser(description="Test voice clone TTS")
    parser.add_argument("--text", type=str, help="Custom text to speak")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
