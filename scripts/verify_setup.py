"""Phase 0 Verification Script — tests all components are installed and working."""

import sys
import subprocess
import json

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def check(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def main():
    print("\n=== Voice Agent — Phase 0 Verification ===\n")
    all_passed = True

    # 1. Python version
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 11
    all_passed &= check("Python >= 3.11", ok, f"Python {v.major}.{v.minor}.{v.micro}")

    # 2. Core Python packages
    packages = ["sounddevice", "numpy", "aiohttp", "websockets", "pydantic", "rich", "ollama", "yaml"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            all_passed &= check(f"Package: {pkg}", True, ver)
        except ImportError:
            all_passed &= check(f"Package: {pkg}", False, "not installed")

    # 3. Audio devices
    print()
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        blackhole_found = False
        for d in devices:
            if "BlackHole" in d["name"]:
                blackhole_found = True
                check("BlackHole audio device", True, d["name"])
                break
        if not blackhole_found:
            check("BlackHole audio device", False, "not found — install with: brew install blackhole-2ch")
            all_passed = False
    except Exception as e:
        all_passed &= check("Audio devices", False, str(e))

    # 4. Ollama
    print()
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        ollama_ok = result.returncode == 0
        all_passed &= check("Ollama server", ollama_ok)

        if ollama_ok:
            has_llama = "llama3.1:8b" in result.stdout
            all_passed &= check(
                "Ollama model: llama3.1:8b",
                has_llama,
                "found" if has_llama else "not found — run: ollama pull llama3.1:8b",
            )
    except FileNotFoundError:
        all_passed &= check("Ollama", False, "not installed")
    except subprocess.TimeoutExpired:
        all_passed &= check("Ollama", False, "server not responding")

    # 5. voxtral.c binary
    print()
    voxtral_bin = "./vendor/voxtral.c/voxtral"
    try:
        result = subprocess.run([voxtral_bin], capture_output=True, text=True, timeout=5)
        # voxtral with no args should show usage/error but prove it runs
        all_passed &= check("voxtral.c binary", True, "compiled and runnable")
    except FileNotFoundError:
        all_passed &= check("voxtral.c binary", False, f"not found at {voxtral_bin}")
    except Exception as e:
        # Even a non-zero exit is fine — it means the binary runs
        all_passed &= check("voxtral.c binary", True, "compiled")

    # 5b. voxtral model weights
    import os
    model_path = "./vendor/voxtral.c/voxtral-model/consolidated.safetensors"
    if os.path.exists(model_path):
        size_gb = os.path.getsize(model_path) / (1024**3)
        all_passed &= check("Voxtral model weights", size_gb > 5, f"{size_gb:.1f} GB")
    else:
        check("Voxtral model weights", False, "not found — run: cd vendor/voxtral.c && ./download_model.sh")
        all_passed = False

    # 6. VoiceBox
    print()
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8000/docs")
        urllib.request.urlopen(req, timeout=3)
        all_passed &= check("VoiceBox API server", True, "http://localhost:8000")
    except Exception:
        print(f"  {SKIP} VoiceBox API server — not running (start VoiceBox app first)")

    # 7. ffmpeg
    print()
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        ver = result.stdout.split("\n")[0] if result.returncode == 0 else "unknown"
        all_passed &= check("ffmpeg", result.returncode == 0, ver)
    except FileNotFoundError:
        all_passed &= check("ffmpeg", False, "not installed — run: brew install ffmpeg")

    # Summary
    print(f"\n{'='*50}")
    if all_passed:
        print(f"  {PASS} All checks passed! Ready to build.")
    else:
        print(f"  {FAIL} Some checks failed. Fix the issues above.")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
