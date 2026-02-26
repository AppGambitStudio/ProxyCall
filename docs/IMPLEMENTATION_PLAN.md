# Implementation Plan

## Phase Overview

| Phase | Name | Goal | Estimated Effort |
|-------|------|------|-----------------|
| 0 | Environment Setup | Install all tools, verify hardware | Half day |
| 1 | Audio Pipeline | Capture Meet audio, play back audio | 1-2 days |
| 2 | Real-time Transcription | Live transcript from meeting audio | 1-2 days |
| 3 | Brain — Intent + Response | Detect questions for you, generate answers | 2-3 days |
| 4 | Voice — Clone + Speak | Respond in your cloned voice | 1-2 days |
| 5 | Orchestration | Wire everything together, add controls | 2-3 days |
| 6 | Polish | Reliability, latency tuning, edge cases | Ongoing |

---

## Phase 0: Environment Setup

### 0.1 Install BlackHole
```bash
brew install blackhole-2ch
```
Then in **Audio MIDI Setup** (macOS):
1. Click "+" → Create Multi-Output Device
2. Add: Your speakers/headphones + BlackHole 2ch
3. Name it "Meet + Agent"
4. When joining a meeting, set system output to "Meet + Agent"

### 0.2 Install Ollama
```bash
brew install ollama
ollama pull llama3.1:8b
# Verify
ollama run llama3.1:8b "Hello, are you working?"
```

### 0.3 Build voxtral.c
```bash
git clone https://github.com/antirez/voxtral.c
cd voxtral.c
make mps           # Build with Metal GPU support
./download_model.sh  # Downloads ~8.9GB model weights
# Test
./voxtral -d voxtral-model -i test_audio.wav
```

### 0.4 Install VoiceBox
- Download from https://voicebox.sh/ (macOS ARM build)
- Or build from source: https://github.com/jamiepine/voicebox
- Launch, create a voice profile with your voice samples
- Verify API: `curl http://localhost:8000/docs`

### 0.5 Python Environment
```bash
cd voiceagent
python3 -m venv .venv
source .venv/bin/activate
pip install sounddevice numpy aiohttp websockets pydantic rich silero-vad ollama
```

### 0.6 Verification Checklist
- [ ] BlackHole appears as audio device in System Settings
- [ ] Multi-Output Device created and working
- [ ] Ollama responds to prompts
- [ ] voxtral.c transcribes a test WAV file
- [ ] VoiceBox generates speech from text via API
- [ ] Python venv activated with all deps

---

## Phase 1: Audio Pipeline

### Goal
Capture system audio from Google Meet via BlackHole, and play audio back through a virtual device.

### 1.1 Audio Capture Module

**File: `src/audio/capture.py`**

Core responsibilities:
- Discover and connect to BlackHole device
- Continuous capture at 16kHz mono float32
- Ring buffer storing last 2 minutes of audio
- Expose async generator for downstream consumers
- Handle device disconnection gracefully

Key implementation details:
```
- Use sounddevice.InputStream with callback
- Buffer size: 480 samples (30ms at 16kHz)
- Thread-safe queue between callback and async consumer
- Auto-detect BlackHole device ID by name matching
```

### 1.2 Audio Playback Module

**File: `src/audio/playback.py`**

Core responsibilities:
- Play generated speech audio to output device
- Support both direct speaker output (for testing) and BlackHole routing
- Non-blocking async playback
- Volume control

### 1.3 Device Manager

**File: `src/audio/devices.py`**

Core responsibilities:
- List available audio devices
- Find BlackHole input/output devices
- Validate device configuration
- Helper to create/verify Multi-Output Device setup

### 1.4 Test: Audio Loopback
Record 5 seconds from BlackHole → save to WAV → play back → verify it matches what was playing in the meeting.

### Deliverable
- Can capture live meeting audio programmatically
- Can play audio files to a specific device
- Audio quality verified (no distortion, correct sample rate)

---

## Phase 2: Real-time Transcription

### Goal
Live, streaming transcription of meeting audio with <1s latency.

### 2.1 Voxtral ASR Wrapper

**File: `src/asr/voxtral.py`**

Two modes:

**Mode A: voxtral.c subprocess (recommended for 24GB)**
```
- Launch voxtral.c as subprocess with --stdin flag
- Pipe audio chunks to stdin
- Read transcript lines from stdout
- Parse timestamps and text
```

**Mode B: vLLM WebSocket (for future 48/64GB machine)**
```
- Connect to vLLM Realtime API at /v1/realtime
- Stream audio chunks over WebSocket
- Receive transcript tokens in real-time
```

### 2.2 Transcript Buffer

**File: `src/transcript/buffer.py`**

Core responsibilities:
- Append timestamped transcript segments
- Query last N seconds of transcript
- Full transcript history for the session
- Export to text/markdown

Data structure:
```python
@dataclass
class TranscriptSegment:
    timestamp: float      # seconds since meeting start
    text: str
    speaker: str | None   # from diarization (Phase 4)
    confidence: float
```

### 2.3 Voice Activity Detection (VAD)

**File: `src/asr/vad.py`**

Using silero-vad:
- Detect speech vs silence in audio stream
- Trigger "end of utterance" event after 1.5s silence
- This event signals the orchestrator to check intent
- Prevents cutting off speakers mid-sentence

### 2.4 Test: Live Transcription
Play a YouTube video / podcast → capture via BlackHole → see live transcript in terminal → verify accuracy.

### Deliverable
- Real-time transcript appearing in terminal as people speak
- <1s delay from speech to text
- Utterance boundary detection working

---

## Phase 3: Brain — Intent Detection + Response Generation

### Goal
Determine if speech is directed at you. If yes, generate an appropriate response.

### 3.1 Meeting Context Manager

**File: `src/brain/context.py`**

Core responsibilities:
- Load meeting brief from markdown file
- Structure: meeting topic, attendees, your role, key points, things to avoid
- Provide context window for LLM prompts

Meeting brief format:
```markdown
# Meeting: Weekly Engineering Sync
## Date: 2026-02-26
## Attendees: Dhaval (you), Sarah (PM), Mike (Backend Lead), Lisa (Frontend)
## Your Role: Tech Lead
## Key Context:
- API v2 launch targeting end of Q1
- Auth module is done, rate limiting in progress
- Frontend migration 60% complete
## Your Positions:
- We should use token-based rate limiting, not IP-based
- Migration timeline is realistic if we get the design system done first
## Things to Avoid:
- Don't commit to dates without checking with the team
- Don't discuss the acquisition rumors
```

### 3.2 Intent Classifier

**File: `src/brain/intent.py`**

Two-tier detection:

**Tier 1 — Fast Pattern Match (< 10ms)**
```python
TRIGGER_PATTERNS = [
    r'\bdhaval\b',                    # Direct name mention
    r'\bhey dhaval\b',
    r'what do you think\s*[,?]?\s*dhaval',
    r'dhaval\s*,?\s*can you',
    r'dhaval\s*,?\s*what',
    r'over to you\s*,?\s*dhaval',
]
```
If any pattern matches → proceed to Tier 2.

**Tier 2 — LLM Classification (~1-2s)**
```
System prompt: You analyze meeting transcripts to determine if the last
utterance is directed at or requires a response from Dhaval.

Consider:
- Direct address by name
- Questions following a topic Dhaval owns
- "What do you think?" after Dhaval's area of expertise was discussed
- Round-robin questions ("and Dhaval?")

Reply with JSON only:
{
  "directed_at_me": bool,
  "confidence": float (0-1),
  "question_summary": str,
  "urgency": "immediate" | "can_wait" | "fyi_only"
}
```

### 3.3 Response Generator

**File: `src/brain/responder.py`**

System prompt structure:
```
You are responding on behalf of Dhaval in a meeting.
[Meeting context from brief]

Rules:
- Keep responses to 2-3 sentences maximum
- Match Dhaval's communication style: [direct, technical, friendly]
- If unsure, say "Let me get back to you on that" rather than guessing
- Never make commitments about timelines without hedging
- Reference specific technical details from the meeting context when relevant

Recent transcript:
[Last 2 minutes of conversation]

Question directed at Dhaval:
[Extracted question]

Generate Dhaval's response:
```

### 3.4 Confidence Gate

**File: `src/brain/gate.py`**

Decision logic:
```
IF confidence >= 0.9 AND urgency == "immediate":
    → Respond automatically
IF confidence >= 0.8 AND urgency == "immediate":
    → Respond automatically (but log for review)
IF confidence >= 0.7:
    → Show notification, wait for manual approval (3s timeout)
IF confidence < 0.7:
    → Stay silent, log the detection for review
```

### 3.5 Test: Simulated Meeting
Feed pre-recorded meeting audio with known questions → verify correct intent detection and reasonable responses.

### Deliverable
- Agent correctly identifies questions directed at you (>90% accuracy)
- Generates contextually appropriate responses
- Stays silent when it should

---

## Phase 4: Voice — Clone + Speak

### Goal
Respond in your cloned voice with natural-sounding speech.

### 4.1 Voice Profile Setup

**One-time setup:**
1. Record 3-5 voice samples (5-10 seconds each) of you speaking naturally
2. Vary the samples: questions, statements, excited, calm
3. Create voice profile in VoiceBox
4. Test with various phrases, adjust if needed

**File: `src/voice/profile.py`**
- Manage voice profile references
- Store profile_id for API calls
- Utility to record new samples

### 4.2 TTS Client

**File: `src/voice/tts.py`**

Primary: VoiceBox REST API
```python
async def synthesize(text: str, profile_id: str) -> np.ndarray:
    async with aiohttp.ClientSession() as session:
        resp = await session.post("http://localhost:8000/generate", json={
            "text": text,
            "profile_id": profile_id,
            "language": "en"
        })
        audio_data = await resp.read()
        return np.frombuffer(audio_data, dtype=np.float32)
```

Fallback: mlx-audio (in-process)
```python
from mlx_audio.tts.utils import load_model
model = load_model("mlx-community/Qwen3-TTS-0.6B-bf16")
# Use with voice cloning reference audio
```

### 4.3 Speech Naturalness

Post-processing for natural meeting speech:
- Add slight pause (200-400ms) before speaking (thinking time)
- Normalize volume to match meeting audio levels
- Optional: Add filler word at start ("So," / "Yeah," / "Right,") for naturalness

### 4.4 Test: Voice Quality
Generate responses to 10 sample questions → listen → rate quality → iterate on profile.

### Deliverable
- Voice cloning working with your voice
- Speech output sounds natural in meeting context
- <3s from text to first audio output

---

## Phase 5: Orchestration — Wire Everything Together

### Goal
A single command to start the agent, with hotkey controls and status display.

### 5.1 Main Orchestrator

**File: `src/orchestrator.py`**

The central event loop that connects all components:
```
1. Start audio capture
2. Start ASR streaming
3. Start transcript buffer
4. Main loop:
   a. Receive transcript updates
   b. On silence detection → check intent
   c. On intent match → generate response
   d. On response ready → synthesize and play
   e. Handle hotkey events
```

### 5.2 State Machine

```
IDLE → LISTENING → DETECTING → THINKING → SPEAKING → IDLE
  ↑                                                    │
  └────────────────────────────────────────────────────┘

MUTED (toggle with hotkey, overrides everything)
```

States:
- **IDLE**: Agent is passive, no speech detected recently
- **LISTENING**: Active speech detected, building transcript
- **DETECTING**: Silence detected, analyzing if question was for me
- **THINKING**: Question confirmed, generating response via LLM
- **SPEAKING**: Playing TTS audio output
- **MUTED**: Agent is silenced, still transcribing but won't respond

### 5.3 Terminal UI

**File: `src/ui/terminal.py`**

Using `rich` library:
```
┌─ Voice Agent ─────────────────────────────────┐
│ Status: 🟢 LISTENING          [M]ute [Q]uit  │
├───────────────────────────────────────────────┤
│ Transcript:                                    │
│ [Sarah] So the API launch is looking good...   │
│ [Mike] Yeah, the auth module passed all tests  │
│ [Sarah] Dhaval, what's your take on the        │
│         rate limiting approach?                 │
│                                                │
│ ► Detected question for you (confidence: 0.94) │
│ ► Generating response...                       │
│ ► Speaking: "I think we should go with token-  │
│   based rate limiting. It gives us better       │
│   control per-user and is easier to scale."    │
├───────────────────────────────────────────────┤
│ Meeting: Weekly Engineering Sync               │
│ Context loaded: 12 key points                  │
│ Latency: ASR 480ms | LLM 2.1s | TTS 1.8s     │
└───────────────────────────────────────────────┘
```

### 5.4 Hotkey Controls

| Key | Action |
|-----|--------|
| `m` | Toggle mute/unmute |
| `f` | Force respond to last utterance |
| `s` | Skip current response (stop speaking) |
| `c` | Show/edit meeting context |
| `t` | Show full transcript |
| `q` | Quit agent |

### 5.5 Configuration

**File: `config.yaml`**
```yaml
agent:
  name: "Dhaval"
  trigger_names: ["Dhaval", "dhaval"]
  confidence_threshold: 0.8
  max_response_sentences: 3
  silence_timeout_ms: 1500

audio:
  capture_device: "BlackHole 2ch"
  playback_device: "MacBook Pro Speakers"  # or BlackHole for routing to Meet
  sample_rate: 16000

asr:
  engine: "voxtral.c"  # or "vllm"
  model_path: "./models/voxtral"
  delay_ms: 480

llm:
  provider: "ollama"
  model: "llama3.1:8b"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 200

tts:
  provider: "voicebox"  # or "mlx-audio"
  base_url: "http://localhost:8000"
  voice_profile_id: "dhaval-main"
  language: "en"

meeting:
  context_file: "./meetings/current.md"
```

### 5.6 CLI Entry Point

**File: `src/main.py`**
```bash
# Start agent with default config
python -m src.main

# Start with specific meeting context
python -m src.main --meeting meetings/weekly_sync.md

# Start in listen-only mode (transcribe but don't respond)
python -m src.main --listen-only

# Start with specific audio devices
python -m src.main --capture-device "BlackHole 2ch" --playback-device "Speakers"
```

### Deliverable
- Single command starts the full agent
- Hotkeys for control
- Live terminal UI showing status and transcript
- Clean shutdown

---

## Phase 6: Polish & Hardening

### 6.1 Latency Optimization
- Profile each component, find bottlenecks
- Pre-warm LLM with meeting context on startup
- Overlap ASR and intent detection (pipeline)
- Stream TTS output (start playing first chunk while generating rest)

### 6.2 Reliability
- Auto-reconnect on audio device changes
- Handle Ollama/VoiceBox server crashes
- Graceful degradation (text notification if TTS fails)
- Session logging for post-meeting review

### 6.3 Edge Cases
- Multiple people talking at once (overlapping speech)
- Background noise handling
- Agent being asked to respond while already speaking
- Very long questions that span multiple utterances
- Questions in languages other than English

### 6.4 Testing
- Unit tests for each component
- Integration tests with recorded meeting audio
- Latency benchmarks under load
- Voice quality A/B testing

---

## Project Structure

```
voiceagent/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TECH_STACK.md
│   ├── IMPLEMENTATION_PLAN.md
│   └── HARDWARE_SETUP.md
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── orchestrator.py         # Central event loop
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py          # BlackHole audio capture
│   │   ├── playback.py         # Audio output
│   │   └── devices.py          # Device discovery
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── voxtral.py          # Voxtral ASR wrapper
│   │   └── vad.py              # Voice Activity Detection
│   ├── transcript/
│   │   ├── __init__.py
│   │   └── buffer.py           # Rolling transcript
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── context.py          # Meeting context manager
│   │   ├── intent.py           # Intent classification
│   │   ├── responder.py        # Response generation
│   │   └── gate.py             # Confidence gating
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── tts.py              # TTS client (VoiceBox/mlx-audio)
│   │   └── profile.py          # Voice profile management
│   └── ui/
│       ├── __init__.py
│       └── terminal.py         # Rich terminal UI
├── meetings/                   # Meeting context files
│   └── example.md
├── voice_samples/              # Your voice recordings
├── config.yaml
├── requirements.txt
└── README.md
```
