# Voice Agent Architecture

## Overview

A local AI agent that listens to Google Meet calls in real-time, detects when questions are directed at you, and responds using your cloned voice. Designed to run entirely on local hardware (Apple Silicon Macs).

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Google Meet (Browser)                        │
│                                                                     │
│   Speaker Audio ──► BlackHole (Virtual Audio) ──► Agent Input       │
│   Agent Output  ◄── Virtual Mic ◄────────────── Agent TTS Output   │
└─────────────────────────────────────────────────────────────────────┘
                              │                          ▲
                              ▼                          │
┌──────────────────────────────────────────────────────────────────────┐
│                         Voice Agent Core                             │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐   │
│  │ Audio Capture │    │  ASR Engine  │    │  Speaker Diarization │   │
│  │  (PyAudio /   │───►│  (Voxtral    │───►│  (pyannote /         │   │
│  │  sounddevice) │    │   Realtime)  │    │   energy-based)      │   │
│  └──────────────┘    └──────┬───────┘    └──────────┬────────────┘   │
│                              │                       │               │
│                              ▼                       ▼               │
│                    ┌──────────────────────────────────────┐          │
│                    │        Orchestrator / Brain          │          │
│                    │                                      │          │
│                    │  - Rolling transcript buffer         │          │
│                    │  - Intent detection (is this for me?)│          │
│                    │  - Meeting context (pre-loaded)      │          │
│                    │  - Response decision engine          │          │
│                    │  - Confidence gating                 │          │
│                    └──────────────┬───────────────────────┘          │
│                                   │                                  │
│                                   ▼                                  │
│                    ┌──────────────────────────────────┐              │
│                    │       LLM (Ollama)               │              │
│                    │  - Intent classification         │              │
│                    │  - Response generation           │              │
│                    │  - Context-aware reasoning       │              │
│                    └──────────────┬──────────────────┘               │
│                                   │                                  │
│                                   ▼                                  │
│                    ┌─────────────────────────────────┐               │
│                    │   TTS / Voice Clone             │               │
│                    │   (VoiceBox / Qwen3-TTS         │               │
│                    │    via mlx-audio)               │               │
│                    └──────────────┬──────────────────┘               │
│                                   │                                  │
│                                   ▼                                  │
│                    ┌─────────────────────────────────┐               │
│                    │    Audio Output                 │               │
│                    │    (Play to virtual mic)        │               │
│                    └─────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Audio Capture Pipeline
```
Meet Audio → BlackHole → sounddevice (16kHz PCM) → Audio Buffer (ring buffer)
```
- Continuous capture at 16kHz mono
- Ring buffer holds last 30s of audio for context
- Chunks sent to ASR every ~480ms

### 2. Transcription Pipeline
```
Audio Chunks → Voxtral Realtime (WebSocket) → Timestamped Transcript
```
- Streaming ASR with 480ms configurable delay
- Returns word-level timestamps
- Feeds into rolling transcript buffer

### 3. Intent Detection Pipeline
```
Transcript → Name Detection + Question Pattern → Ollama (classify) → Decision
```
- **Fast path**: Regex/keyword detection for your name ("Dhaval", "hey Dhaval")
- **Slow path**: LLM classification for indirect references ("what do you think?", "can you explain...")
- Confidence threshold: Only respond if confidence > 0.8
- Silence detection: Wait for speaker to finish (1.5s pause) before responding

### 4. Response Generation Pipeline
```
Meeting Context + Transcript + Question → Ollama → Response Text
```
- System prompt includes: meeting brief, your role, your communication style
- Rolling transcript provides conversation context
- Response limited to 2-3 sentences for natural meeting flow

### 5. Voice Synthesis Pipeline
```
Response Text → VoiceBox API (Qwen3-TTS) → PCM Audio → Virtual Mic → Meet
```
- Uses pre-created voice profile from your voice samples
- Streaming TTS output for lower perceived latency
- Audio routed to BlackHole virtual mic input

## Key Design Principles

### Confidence Gating
The agent should **stay silent when unsure** rather than answering incorrectly. A wrong answer in a meeting is far worse than a delayed response. The confidence threshold is configurable.

### Graceful Degradation
- If ASR fails: Log and continue, don't interrupt
- If LLM is slow: Queue the response, don't drop it
- If TTS fails: Fall back to text notification on screen

### Manual Override
- **Hotkey to mute agent**: Spacebar or configurable key
- **Hotkey to force response**: Agent generates answer for last question
- **Hotkey to correct**: "Actually, what I meant was..." override
- **Visual indicator**: Terminal/overlay showing agent status (listening/thinking/speaking)

## Component Boundaries

| Component | Process | Port | Protocol |
|-----------|---------|------|----------|
| Audio Capture | Main process | - | In-process |
| Voxtral ASR | vLLM server OR voxtral.c | 8001 | WebSocket / stdout |
| Ollama LLM | Ollama server | 11434 | HTTP REST |
| VoiceBox TTS | VoiceBox server | 8000 | HTTP REST |
| Orchestrator | Main process | - | In-process |
| Control UI | Main process | - | Terminal / stdin |

## Multi-Machine Deployment (Optional)

For better performance, split across Mac Mini (16GB) + MacBook Pro (24GB):

```
MacBook Pro 24GB (Primary)          Mac Mini 16GB (Secondary)
├── Audio Capture                   ├── VoiceBox TTS Server
├── Voxtral ASR (voxtral.c)        │   (Qwen3-TTS via MLX)
├── Orchestrator                    │   Port 8000
├── Ollama (Llama 3.1 8B)          └── Accessible via LAN
└── Audio Output
```
