# ProxyCall

A local AI agent that attends your Google Meet calls, listens to conversations, and responds in your cloned voice. All AI inference runs locally on Apple Silicon — no cloud APIs for transcription, LLM, or voice synthesis. The only thing that leaves your machine is the audio going through Google Meet (obviously).

> **Disclaimer:** This is a fun experimental project built and tested only on Apple Silicon Macs (M4). It's not production-ready, will occasionally say odd things, and the latency is noticeable. Use it to amuse yourself, not to fool your boss.

## How It Works

```
Google Meet Audio → BlackHole → Voxtral ASR → Intent Classifier → LLM Response → VoiceBox TTS → Speakers
```

1. **Audio Capture** — Routes Google Meet audio through [BlackHole](https://existential.audio/blackhole/) virtual audio device
2. **Speech-to-Text** — [voxtral.c](https://github.com/nicholasgasior/voxtral.c) (Mistral's Voxtral Realtime 4B) transcribes speech in real-time
3. **Intent Detection** — Local LLM via [Ollama](https://ollama.com) decides if the utterance needs a response
4. **Response Generation** — Same LLM generates a contextual response using your meeting prep notes
5. **Voice Synthesis** — [VoiceBox](https://voicebox.sh) clones your voice and speaks the response through your speakers

## Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  BlackHole   │────▶│  voxtral.c    │────▶│  Transcript  │
│  (audio in)  │     │  (ASR)        │     │  Buffer      │
└──────────────┘     └───────────────┘     └──────┬───────┘
                                                  │
                     ┌───────────────┐            │
                     │  Meeting      │            │
                     │  Context (.md)│────┐       │
                     └───────────────┘    │       │
                                          ▼       ▼
┌──────────────┐     ┌──────────────┐  ┌──────────────┐
│  Speakers    │◀────│  VoiceBox    │◀─│  Qwen3 LLM   │
│  (audio out) │     │  (TTS)       │  │  (brain)     │
└──────────────┘     └──────────────┘  └──────────────┘
```

The meeting context file feeds your prep notes (status updates, positions, communication style) into the LLM so responses are relevant to the actual conversation.

The orchestrator manages a state machine: `IDLE → LISTENING → DETECTING → THINKING → SPEAKING → IDLE`

## Requirements

- **macOS** with Apple Silicon (M4)
- **32GB+ RAM recommended** (24GB works but tight — ASR and TTS share GPU memory)
- **Python 3.11+**
- **BlackHole 2ch** — virtual audio driver
- **Ollama** — local LLM server
- **VoiceBox** — voice cloning TTS app
- **voxtral.c** — compiled from source (see below)

### Optional: Multi-Machine Setup

If you're RAM-constrained (24GB), you can offload Ollama to a second Mac on your local network:
- **Primary Mac**: voxtral.c (ASR) + VoiceBox (TTS) + audio capture
- **Secondary Mac**: Ollama (LLM)

On the secondary Mac:
```bash
# Install and start Ollama
brew install ollama
ollama pull qwen3:8b

# Ollama binds to localhost by default. To expose it on your network:
OLLAMA_HOST=0.0.0.0 ollama serve
```

Then in your `config.yaml` on the primary Mac, point to the secondary machine's IP:
```yaml
llm:
  base_url: "http://192.168.1.x:11434"  # your secondary Mac's local IP
```

## Setup

### 1. Install Dependencies

```bash
# BlackHole virtual audio driver
brew install blackhole-2ch

# Ollama
brew install ollama
ollama pull qwen3:8b  # or: llama3.1:8b, phi4-mini, gemma3:4b

# VoiceBox — download from https://voicebox.sh
# After install, create a voice profile by recording a ~30s sample of your voice

# Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build voxtral.c (ASR Engine)

```bash
# Clone into vendor directory
mkdir -p vendor
git clone https://github.com/nicholasgasior/voxtral.c.git vendor/voxtral.c
cd vendor/voxtral.c

# Build (requires Xcode Command Line Tools)
make

# Download model weights (~2GB)
./download_model.sh

# Verify it works
./voxtral -d voxtral-model samples/jfk.wav

cd ../..
```

> See the [voxtral.c README](https://github.com/nicholasgasior/voxtral.c) for detailed build instructions and troubleshooting.

### 3. Configure Audio Routing

1. Set your **macOS system sound output** to `BlackHole 2ch`
2. In **Google Meet**, Chrome will use the system default output (BlackHole)
3. The agent captures from BlackHole and plays responses through your actual speakers

> **Important:** Chrome latches audio devices at launch. If you change the output device, quit Chrome fully and reopen it.

### 4. Configure the Agent

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`:
- `agent.name` — your name
- `agent.trigger_names` — add common ASR misrecognitions of your name
- `audio.playback_device` — your speaker device name
- `llm.base_url` — Ollama URL (localhost or remote machine)
- `tts.voice_profile_id` — your VoiceBox voice profile ID (leave empty to auto-detect)

### 5. Prepare Meeting Context

The meeting context file is **the most important part** — it's what makes the agent give relevant answers instead of generic ones. Think of it as your meeting prep notes that the agent reads before the call.

```bash
cp meetings/example.md meetings/current.md
```

Edit `meetings/current.md` **before every call** with:

| Section | What to put | Why it matters |
|---------|-------------|----------------|
| **Your Key Context** | Facts, status updates, what you shipped, what's in progress | The agent uses this to answer "what's the status?" questions |
| **Your Positions** | Pre-loaded answers for expected questions (timeline, risks, blockers) | Direct control over what the agent says — "If asked about X, say Y" |
| **Communication Style** | How you speak (direct, casual, formal) | Keeps responses sounding like you |
| **Things to Avoid** | Topics, numbers, or commitments the agent should never make up | Prevents hallucination and overpromising |

**Example — before a weekly sync:**
```markdown
## Your Key Context:
- Finished the API refactor, pushed to staging on Wednesday
- Production is stable, no incidents this week
- Blocked on design review for the dashboard redesign

## Your Positions:
- If asked about timeline: API refactor ships Monday, dashboard depends on design team
- If asked about risks: only risk is the design dependency
- If asked about production: all green, no issues
```

The better your prep, the better the responses. The agent will say "let me get back to you on that" for anything not covered — which is the right thing to do.

## Usage

```bash
source .venv/bin/activate

# Run with debug logging (recommended for first run)
python -m src.main --no-ui --debug

# Watch the pipeline in another terminal
tail -f voiceagent.log

# Run with terminal UI
python -m src.main

# Listen-only mode (transcribe but don't respond)
python -m src.main --listen-only

# Custom meeting context
python -m src.main --meeting meetings/my-standup.md
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `M` | Toggle mute (still transcribes but won't respond) |
| `F` | Force respond to the last utterance |
| `S` | Skip/stop current response playback |
| `Q` | Quit |

## Project Structure

```
src/
├── main.py              # CLI entry point
├── orchestrator.py      # Central state machine
├── audio/
│   ├── capture.py       # BlackHole audio capture (48kHz→16kHz resampling)
│   ├── devices.py       # Audio device discovery
│   └── playback.py      # Speaker output
├── asr/
│   └── voxtral.py       # voxtral.c subprocess wrapper
├── brain/
│   ├── context.py       # Meeting markdown parser
│   ├── intent.py        # "Does this need a response?" classifier
│   ├── gate.py          # Confidence threshold gate
│   └── responder.py     # Response generator
├── transcript/
│   └── buffer.py        # Rolling transcript buffer
├── voice/
│   ├── tts.py           # VoiceBox TTS client
│   └── profile.py       # Voice profile management
└── ui/
    └── terminal.py      # Rich terminal dashboard
```

## Test Scripts

```bash
# Verify all components are installed
python scripts/verify_setup.py

# Test audio capture from BlackHole
python scripts/test_audio_pipeline.py

# Test live transcription
python scripts/test_live_transcription.py

# Test intent classification and response generation
python scripts/test_brain.py

# Test voice synthesis
python scripts/test_voice.py
```

## Known Limitations

- **Latency** — End-to-end response takes 15-30s on 24GB machines (intent ~5s + response ~5s + TTS ~15s). More RAM helps significantly.
- **ASR restarts** — On memory-constrained machines, voxtral.c must stop during TTS to free GPU memory, causing a few seconds of deaf time after each response.
- **Single speaker** — Optimized for 1-on-1 calls. Group calls would need diarization (not yet implemented).
- **English only** — ASR and TTS are configured for English.
- **macOS only** — Depends on BlackHole, Metal GPU acceleration, and macOS audio APIs.

## Tech Stack

| Component | Technology | License |
|-----------|-----------|---------|
| ASR | [voxtral.c](https://github.com/nicholasgasior/voxtral.c) (Voxtral Realtime 4B) | Apache 2.0 |
| LLM | [Ollama](https://ollama.com) + Qwen3 8B | Apache 2.0 |
| TTS | [VoiceBox](https://voicebox.sh) (Qwen3-TTS) | MIT |
| Audio | [BlackHole](https://existential.audio/blackhole/) + sounddevice | MIT |
| Language | Python 3.11+ with asyncio | — |

## License

MIT
