# Hardware Setup Guide

## Your Hardware

| Machine | Chip | RAM | Role |
|---------|------|-----|------|
| MacBook Pro | M4 | 24GB | Primary runtime — runs full agent stack |
| Mac Mini | M4 | 16GB | Secondary — offload TTS server (optional) |
| Future MBP | M4 Pro/Max | 48-64GB | Future primary — larger models, better perf |

---

## Configuration A: Single Machine (24GB MBP)

This is the recommended starting configuration. Everything runs on one machine.

### Memory Allocation

| Component | RAM | Notes |
|-----------|-----|-------|
| macOS + system | ~4GB | Baseline OS overhead |
| voxtral.c (ASR) | ~2GB | KV cache + working buffers |
| Ollama (Llama 3.1 8B Q4) | ~5GB | Quantized model in unified memory |
| VoiceBox (Qwen3-TTS 0.6B) | ~3GB | MLX backend |
| Python agent + buffers | ~1GB | Orchestrator, audio buffers |
| **Total** | **~15GB** | |
| **Available headroom** | **~9GB** | Comfortable margin |

### Audio Routing

```
Google Meet (Chrome/Safari)
    │
    │ System audio output
    ▼
┌─────────────────────────────┐
│  Multi-Output Device        │
│  (created in Audio MIDI)    │
│                             │
│  ├── Your Headphones ◄──── You hear the meeting
│  └── BlackHole 2ch ◄────── Agent hears the meeting
└─────────────────────────────┘
    │
    │ BlackHole input stream
    ▼
┌─────────────────────────────┐
│  Voice Agent                │
│  (captures from BlackHole)  │
│                             │
│  Response audio ──► Speakers (for now)
│                     OR
│                     BlackHole output → Meet mic (advanced)
└─────────────────────────────┘
```

#### Initial Setup (Phase 1-4): Agent speaks through speakers
- You hear the agent's response through your speakers
- Other meeting attendees do NOT hear the agent
- You can then relay or the agent's audio naturally gets picked up by your mic
- **Simplest setup, good for development and testing**

#### Advanced Setup (Phase 5+): Agent speaks into Meet directly
- Route agent TTS output to a second BlackHole device (BlackHole 16ch)
- Set Google Meet's microphone input to BlackHole 16ch
- Agent's voice goes directly into the meeting
- **Requires careful echo cancellation to prevent feedback loops**

### Setup Steps

#### 1. Install BlackHole
```bash
brew install blackhole-2ch
```

#### 2. Create Multi-Output Device
1. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup")
2. Click **"+"** button at bottom left → **Create Multi-Output Device**
3. Check both:
   - Your headphones/speakers
   - BlackHole 2ch
4. Right-click → **Use This Device For Sound Output** (optional, can also set in System Settings)
5. Rename it to "Meet + Agent" for clarity

#### 3. Verify
```bash
# List audio devices from Python
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```
You should see "BlackHole 2ch" in the list.

#### 4. Test Capture
```python
import sounddevice as sd
import numpy as np
import wave

# Record 5 seconds from BlackHole
DURATION = 5
SAMPLE_RATE = 16000

# Find BlackHole device
devices = sd.query_devices()
blackhole_id = None
for i, d in enumerate(devices):
    if "BlackHole" in d["name"] and d["max_input_channels"] > 0:
        blackhole_id = i
        break

print(f"Recording from device {blackhole_id}: {devices[blackhole_id]['name']}")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
               channels=1, dtype='float32', device=blackhole_id)
sd.wait()

# Save to WAV
with wave.open("test_capture.wav", "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SAMPLE_RATE)
    f.writeframes((audio * 32767).astype(np.int16).tobytes())

print("Saved test_capture.wav — play it back to verify")
```

---

## Configuration B: Two-Machine Split (24GB MBP + 16GB Mac Mini)

Splits the workload over LAN for better performance.

### Machine Allocation

**MacBook Pro 24GB (Primary)**
| Component | RAM |
|-----------|-----|
| macOS | ~4GB |
| voxtral.c (ASR) | ~2GB |
| Ollama (Llama 3.1 8B Q4) | ~5GB |
| Python agent | ~1GB |
| **Total** | **~12GB** |
| **Headroom** | **~12GB** |

**Mac Mini 16GB (TTS Server)**
| Component | RAM |
|-----------|-----|
| macOS | ~4GB |
| VoiceBox (Qwen3-TTS 0.6B) | ~3GB |
| **Total** | **~7GB** |
| **Headroom** | **~9GB** |

### Network Setup

Both machines on same LAN (WiFi or Ethernet). Ethernet preferred for <1ms latency.

```
MBP (agent) ──── LAN ──── Mac Mini (TTS)
                           http://<mini-ip>:8000/generate
```

In `config.yaml`:
```yaml
tts:
  provider: "voicebox"
  base_url: "http://192.168.1.XX:8000"  # Mac Mini's IP
```

### When to Use This Config
- When you notice memory pressure on the MBP (swap usage, slowdowns)
- When you want faster TTS (Mini is dedicated to just TTS)
- When you want to keep MBP responsive for other work during meetings

---

## Configuration C: Future 48-64GB Machine

With more RAM, you can run larger/better models:

| Component | Model | RAM |
|-----------|-------|-----|
| ASR | Voxtral 4B Realtime (full, via vLLM) | ~10GB |
| LLM | Llama 3.1 70B Q4 | ~40GB |
| TTS | Qwen3-TTS 1.7B Pro | ~6GB |
| Diarization | pyannote full pipeline | ~2GB |

Benefits:
- 70B model gives dramatically better reasoning and response quality
- Full Voxtral 4B gives better accuracy than 3B
- Larger TTS model gives better voice quality
- Can add full speaker diarization

---

## Google Meet Audio Setup

### For Development/Testing (Recommended Start)
1. Set system output to "Meet + Agent" (Multi-Output Device)
2. Join Google Meet normally
3. Agent captures audio from BlackHole
4. Agent's responses play through your speakers
5. You're the "relay" — other attendees hear you, not the agent directly

### For Production (Advanced)
1. Install BlackHole 16ch (in addition to 2ch)
2. Set system output to Multi-Output Device (speakers + BlackHole 2ch)
3. In Google Meet settings → Microphone → BlackHole 16ch
4. Agent outputs TTS audio to BlackHole 16ch
5. Meet picks up agent's voice as "your microphone"
6. **Warning**: You'll need to mute your real mic or the agent needs to handle your real voice vs its output

### Echo Prevention
When agent speaks into Meet directly:
- Agent must temporarily stop listening while speaking (half-duplex)
- Or implement echo cancellation to filter out its own voice from the capture
- Simplest approach: State machine — when SPEAKING, ignore all ASR input

---

## Troubleshooting

### "BlackHole not showing up as a device"
```bash
# Check if kernel extension is loaded
kextstat | grep BlackHole
# If not, may need to allow in System Settings → Privacy & Security
```

### "No audio captured from BlackHole"
- Verify system output is set to the Multi-Output Device
- Check that BlackHole 2ch is checked in the Multi-Output Device
- Test: Play music → run capture script → verify WAV has audio

### "Audio is choppy or has gaps"
- Increase buffer size in sounddevice (try 1024 or 2048)
- Check CPU usage — if ML models are hogging CPU, audio callbacks can be delayed
- Consider running audio capture in a separate high-priority process

### "Meet doesn't see BlackHole as microphone"
- Refresh the page after connecting BlackHole
- Some browsers cache audio device lists
- Try: Chrome → Settings → Privacy → Site Settings → Microphone → allow
