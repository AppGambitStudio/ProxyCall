# Tech Stack Decisions

## Summary Table

| Layer | Choice | Alternative Considered | Rationale |
|-------|--------|----------------------|-----------|
| Audio Capture | sounddevice + BlackHole | PyAudio | sounddevice has better macOS/CoreAudio support, simpler API |
| ASR | Voxtral Mini 4B Realtime | Whisper, faster-whisper | Native streaming, <500ms latency, outperforms Whisper on benchmarks |
| Speaker Diarization | pyannote-audio 4.0 | Simple energy-based VAD | Best open-source diarization, runs locally, 2.5% real-time factor |
| LLM | Ollama (Llama 3.1 8B Q4) | Mistral 7B, GPT-4o | Local, fast, good reasoning at 8B, fits in memory alongside other components |
| TTS + Voice Clone | VoiceBox (Qwen3-TTS) | Coqui TTS, Bark, XTTS v2 | Best open-source voice cloning quality, REST API built-in, MLX optimized |
| Audio Output | sounddevice → BlackHole | PyAudio | Same rationale as capture |
| Orchestration | Python asyncio | Node.js, Go | Ecosystem compatibility (all ML libs are Python), async fits event-driven model |
| Language | Python 3.11+ | - | Required by ML stack |

---

## Layer-by-Layer Details

### 1. Audio Capture & Routing — BlackHole + sounddevice

**BlackHole** (https://github.com/ExistentialAudio/BlackHole)
- macOS virtual audio loopback driver
- Zero additional latency
- Creates a virtual audio device that bridges apps
- We use **BlackHole 2ch** (stereo is sufficient)

**Setup**: Create a macOS Multi-Output Device in Audio MIDI Setup:
- Output 1: Your speakers/headphones (so you still hear the meeting)
- Output 2: BlackHole 2ch (so the agent also hears the meeting)
- Set this Multi-Output as system default when in meetings

**sounddevice** (Python library)
- Wraps PortAudio with clean Python API
- Supports callback-based streaming (low latency)
- Can enumerate devices and select BlackHole programmatically
- `pip install sounddevice`

```python
# Example: Capture from BlackHole
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
BLOCK_SIZE = 480  # 30ms chunks

def audio_callback(indata, frames, time, status):
    # indata is numpy array of audio samples
    audio_queue.put(indata.copy())

stream = sd.InputStream(
    device="BlackHole 2ch",
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    blocksize=BLOCK_SIZE,
    callback=audio_callback
)
```

### 2. ASR — Voxtral Mini 4B Realtime

**Why Voxtral over Whisper:**
- **Streaming**: Native real-time streaming with configurable 480ms delay. Whisper is batch-only (you have to hack streaming with overlapping windows).
- **Accuracy**: Outperforms Whisper large-v3 on FLEURS, TEDLIUM, GigaSpeech benchmarks.
- **Understanding**: Not just transcription — has semantic comprehension built in. Can potentially do intent detection directly.
- **License**: Apache 2.0, fully open.

**Deployment options (pick one based on hardware):**

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **voxtral.c** (antirez) | Zero deps, Metal GPU, 2.5x realtime on M3 Max, ~2GB RAM | Less mature, no WebSocket API | Single-machine, lowest overhead |
| **vLLM server** | Production-grade, WebSocket streaming, official support | Heavier, needs ~16GB VRAM | Multi-machine or 48GB+ RAM |
| **transformers** | Familiar API, easy to prototype | No streaming, batch only | Prototyping phase |

**Recommendation**: Start with **voxtral.c** for the 24GB MBP. It's lean (~2GB RAM for KV cache), uses Metal natively, and achieves 2.5x faster than real-time on Apple Silicon. Has `--from-mic` flag for direct mic capture too.

**Performance on Apple Silicon (M3 Max benchmarks, expect similar on M4):**
- Encoder (3.6s audio): 284ms
- Decoder: ~23.5ms per token
- Overall: ~2.5x faster than real-time

### 3. Speaker Diarization — pyannote-audio (Optional, Phase 4)

**pyannote-audio 4.0** with community-1 model
- State-of-the-art open-source speaker diarization
- Runs locally on CPU/GPU
- 2.5% real-time factor on GPU
- Can identify and track different speakers

**Why optional initially**: For a meeting where you know the participants, simple name-based detection ("Dhaval, what do you think?") covers 80% of cases. Diarization becomes important when you want to:
- Know *who* asked the question (to personalize response)
- Handle cases where someone refers to you indirectly
- Build a per-speaker transcript

**Simpler alternative for Phase 1-2**: Voice Activity Detection (VAD) with silero-vad — just detect when someone stops talking (silence > 1.5s), then check if the last utterance was a question directed at you.

### 4. LLM — Ollama with Llama 3.1 8B

**Ollama** (https://ollama.com)
- Already optimized for Apple Silicon
- Simple REST API (`POST /api/generate`, `POST /api/chat`)
- Model management built in
- Streaming responses

**Model: Llama 3.1 8B Q4_K_M**
- ~5GB RAM footprint with Q4 quantization
- Fast inference on M4 (~30-50 tokens/sec)
- Good enough for intent classification + short response generation
- Can upgrade to 70B Q4 on 48/64GB machine later

**Two LLM calls per interaction:**

1. **Intent Classification** (fast, short output):
   ```
   System: You classify if speech is directed at Dhaval. Reply ONLY with JSON.
   User: [last 30s of transcript]
   → {"directed_at_me": true, "confidence": 0.92, "question": "What's the timeline for the API?"}
   ```

2. **Response Generation** (if directed_at_me && confidence > 0.8):
   ```
   System: You are Dhaval's meeting assistant. [meeting context]. Keep responses to 2-3 sentences.
   User: [meeting transcript + detected question]
   → "We're targeting end of Q1 for the API launch. The core endpoints are already built, we're finalizing auth and rate limiting."
   ```

### 5. TTS + Voice Cloning — VoiceBox (Qwen3-TTS)

**VoiceBox** (https://voicebox.sh)
- Open source (MIT), local-first
- Powered by Qwen3-TTS from Alibaba
- REST API: `POST /generate` with text + voice profile
- MLX backend with Metal acceleration on Apple Silicon (4-5x faster)
- Voice cloning from 5-10 seconds of clean audio
- Can run as headless server

**Voice Profile Setup:**
1. Record 5-10 seconds of clean speech (multiple samples improve quality)
2. Create a voice profile in VoiceBox
3. Use the profile_id in API calls

**Alternative: mlx-audio**
- `pip install mlx-audio`
- Supports Qwen3-TTS natively
- Pure Python, no separate server process
- Good fallback if VoiceBox's server mode has issues

**Performance expectations (M4, Qwen3-TTS 0.6B via MLX):**
- ~3GB RAM
- RTF ~1.8-2.2 (real-time factor, meaning 1s of speech takes ~2s to generate)
- For a 2-3 sentence response (~5s of speech): ~10s generation time
- With streaming output: first audio chunk in ~1-2s

### 6. Orchestration — Python asyncio

The orchestrator is the central brain that ties everything together:

```python
# Simplified event loop
async def main():
    audio_stream = AudioCapture(device="BlackHole 2ch")
    asr = VoxtralASR()
    llm = OllamaClient(model="llama3.1:8b")
    tts = VoiceBoxTTS(profile_id="dhaval-voice")

    transcript_buffer = TranscriptBuffer(max_seconds=120)
    meeting_context = load_meeting_context("meeting_brief.md")

    async for audio_chunk in audio_stream:
        # ASR: audio → text
        text = await asr.transcribe(audio_chunk)
        transcript_buffer.append(text)

        # Detect silence (end of utterance)
        if detect_silence(audio_chunk, threshold=1.5):
            recent = transcript_buffer.last(30)  # last 30s

            # Intent classification
            intent = await llm.classify_intent(recent)

            if intent.directed_at_me and intent.confidence > 0.8:
                # Generate response
                response = await llm.generate_response(
                    meeting_context, transcript_buffer.full(), intent.question
                )

                # Speak response
                audio = await tts.synthesize(response)
                await play_audio(audio, device="BlackHole Virtual Mic")
```

---

## Memory Budget (24GB MacBook Pro M4)

| Component | RAM Estimate |
|-----------|-------------|
| macOS + system | ~4GB |
| Voxtral (voxtral.c) | ~2GB |
| Ollama (Llama 3.1 8B Q4) | ~5GB |
| VoiceBox / mlx-audio (Qwen3-TTS 0.6B) | ~3GB |
| pyannote (optional) | ~1GB |
| Python runtime + buffers | ~1GB |
| **Total** | **~16GB** |
| **Headroom** | **~8GB** |

This fits comfortably on 24GB with room for macOS to breathe.

---

## Dependencies Summary

```
# Core
python >= 3.11
sounddevice          # Audio I/O
numpy                # Audio processing
aiohttp              # Async HTTP client
websockets           # WebSocket client (for vLLM if used)

# ASR
voxtral.c            # Built from source (C + Metal)
# OR
vllm                 # If using vLLM server
transformers >= 5.2  # If using transformers

# LLM
ollama               # Server (installed separately)
ollama-python        # Python client

# TTS
voicebox             # Installed separately (desktop app with API)
# OR
mlx-audio            # pip install, pure Python alternative

# Diarization (Phase 4)
pyannote-audio       # Speaker diarization
torch                # Required by pyannote

# Utilities
silero-vad           # Voice Activity Detection
pydantic             # Config and data models
rich                 # Terminal UI
```
