"""Audio device discovery and validation."""

import sounddevice as sd


def list_devices() -> list[dict]:
    """Return all audio devices with their properties."""
    devices = sd.query_devices()
    return [
        {
            "index": i,
            "name": d["name"],
            "max_input_channels": d["max_input_channels"],
            "max_output_channels": d["max_output_channels"],
            "default_samplerate": d["default_samplerate"],
        }
        for i, d in enumerate(devices)
    ]


def find_device(name: str, kind: str = "input") -> int | None:
    """Find device index by name substring.

    Args:
        name: Substring to match in device name (case-insensitive).
        kind: "input" or "output".

    Returns:
        Device index or None if not found.
    """
    devices = sd.query_devices()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(devices):
        if name.lower() in d["name"].lower() and d[channel_key] > 0:
            return i
    return None


def find_blackhole(kind: str = "input") -> int:
    """Find BlackHole 2ch device index. Raises if not found."""
    idx = find_device("BlackHole 2ch", kind)
    if idx is None:
        raise RuntimeError(
            "BlackHole 2ch not found. Install with: brew install blackhole-2ch"
        )
    return idx


def validate_device(index: int, sample_rate: int = 16000, channels: int = 1) -> bool:
    """Check if a device supports the requested configuration."""
    try:
        info = sd.query_devices(index)
        if channels > info["max_input_channels"] and channels > info["max_output_channels"]:
            return False
        # sounddevice will raise if sample rate isn't supported
        sd.check_input_settings(device=index, samplerate=sample_rate, channels=channels)
        return True
    except Exception:
        return False
