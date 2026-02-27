"""Test if BlackHole loopback is working.

BlackHole routes audio between processes, not within a single process.
This test uses playrec (full-duplex) for self-test, and also spawns
a separate process to verify cross-process routing (the real use case).
"""
import subprocess
import sys
import time

import numpy as np
import sounddevice as sd

SR = 48000
DURATION = 1.5  # seconds


def find_blackhole_2ch() -> int:
    """Find BlackHole 2ch device index."""
    for i, dev in enumerate(sd.query_devices()):
        if "BlackHole 2ch" in dev["name"]:
            return i
    raise RuntimeError("BlackHole 2ch not found")


def make_tone(freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, DURATION, int(SR * DURATION), False)
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_fullduplex(dev: int) -> float:
    """Self-loopback via playrec (single process, full-duplex)."""
    tone = make_tone()
    rec = sd.playrec(tone, samplerate=SR, channels=2, device=dev, dtype="float32")
    sd.wait()
    return float(np.max(np.abs(rec)))


def test_cross_process(dev: int) -> float:
    """Play from a subprocess, record in this process."""
    player_code = f"""
import numpy as np, sounddevice as sd, time
sr = {SR}; t = np.linspace(0, 1, sr, False)
tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
time.sleep(0.3)
sd.play(tone, samplerate=sr, device={dev})
sd.wait()
"""
    # Start recording
    rec = sd.rec(int(SR * 2), samplerate=SR, channels=2, dtype="float32", device=dev)
    # Launch player subprocess
    proc = subprocess.Popen([sys.executable, "-c", player_code])
    sd.wait()
    proc.wait()
    return float(np.max(np.abs(rec)))


def main():
    try:
        dev = find_blackhole_2ch()
    except RuntimeError:
        print("BlackHole 2ch not found! Install it: brew install blackhole-2ch")
        sys.exit(1)

    info = sd.query_devices(dev)
    print(f"BlackHole 2ch: device={dev}, sr={int(info['default_samplerate'])}Hz")

    # Test 1: full-duplex self-loopback
    peak1 = test_fullduplex(dev)
    status1 = "PASS" if peak1 > 0.01 else "FAIL"
    print(f"  Full-duplex loopback: peak={peak1:.4f} [{status1}]")

    # Test 2: cross-process (the real Chrome -> Python scenario)
    peak2 = test_cross_process(dev)
    status2 = "PASS" if peak2 > 0.01 else "FAIL"
    print(f"  Cross-process loopback: peak={peak2:.4f} [{status2}]")

    if peak1 > 0.01 or peak2 > 0.01:
        print("\nBlackHole is WORKING")
    else:
        print("\nBlackHole is BROKEN — check Audio MIDI Setup volume levels")
        sys.exit(1)


if __name__ == "__main__":
    main()
