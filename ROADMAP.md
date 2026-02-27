# ProxyCall Roadmap

Where we are, where we're going, and what would make this actually useful.

## Current State (v0.1 — "It Works, Barely")

The end-to-end pipeline is functional: ASR captures speech, LLM classifies intent and generates responses, TTS speaks in your cloned voice. It handles 1-on-1 calls with a static meeting context file.

**What works:**
- Real-time transcription via voxtral.c
- Intent detection ("does this need a response?")
- Contextual response generation from meeting prep notes
- Voice-cloned TTS playback
- Multi-machine Ollama support for memory-constrained setups

**What hurts:**
- 15-30s latency per response (intent + LLM + TTS)
- ASR goes deaf during TTS on 24GB machines (GPU memory juggling)
- Meeting context is a static markdown file you manually write before each call
- No UI — terminal only, tail -f voiceagent.log is the dashboard
- 1-on-1 calls only, no speaker diarization

---

## Phase 1: Smarter Context

The meeting context file is the single biggest lever for response quality. Right now it's a manually written markdown file. This phase makes it richer and less manual.

### 1.1 — Knowledge Base Integration
- **RAG over documents**: Index Confluence, Notion, Google Docs pages relevant to recurring meetings. When the LLM generates a response, it pulls from your actual project docs — not just what you remembered to write in the meeting file.
- **Git-aware context**: Auto-summarize recent commits, PRs, and deployments. "What did we ship this week?" gets answered from real data.
- **Ticket/issue integration**: Pull status from Jira, Linear, or GitHub Issues. "What's the status of PROJ-123?" gets a real answer.

### 1.2 — Live Context Updates
- **Hot-reload**: Edit the meeting context file while the agent is running, changes take effect immediately (no restart).
- **In-call learning**: If you correct the agent or manually respond, it updates its context for the rest of the call.
- **Post-meeting context generation**: After each call, auto-generate a context file from the transcript for future reference.

---

## Phase 2: Desktop App

Replace the terminal with a proper desktop application.

### 2.1 — Core App (Tauri/Electron)
- **Meeting manager**: Create, edit, and switch between meeting contexts with a proper editor. Templates for standups, 1-on-1s, sprint reviews, client calls.
- **Start/stop controls**: One-click to start listening, mute, force respond, or stop. Visual state indicator (IDLE/LISTENING/THINKING/SPEAKING).
- **Live transcript**: Real-time scrolling transcript with speaker labels and the agent's responses highlighted.
- **System tray**: Runs in background, shows status in menu bar. Quick mute/unmute from tray icon.

### 2.2 — Configuration UI
- **Audio device picker**: Visual selection of capture (BlackHole) and playback devices.
- **LLM settings**: Model picker, temperature slider, Ollama URL config.
- **Voice profile manager**: Record, preview, and switch between voice profiles.
- **Confidence threshold tuner**: Slider to control how eagerly the agent responds.

### 2.3 — Meeting History
- **Session logs**: Browsable history of past calls with full transcripts, agent responses, and latency metrics.
- **Response review**: See what the agent said, mark responses as good/bad, refine for next time.
- **Export**: Markdown or PDF meeting summaries from transcripts.

---

## Phase 3: Performance

Bring latency down from "awkward pause" to "natural conversation."

### 3.1 — Faster Intent Classification
- **Local classifier**: Fine-tune a tiny model (< 1B params) specifically for intent classification. Skip the full LLM round-trip for obvious cases.
- **Heuristic fast-path**: Questions ending in "?" or containing "can you", "what's the" — classify locally in < 100ms, only use LLM for ambiguous cases.
- **Streaming intent**: Start classifying while the person is still talking, not just after silence.

### 3.2 — Streaming TTS
- **Chunk-based synthesis**: Generate TTS for the first sentence while the LLM is still producing the second. Start speaking immediately.
- **Sentence pipelining**: LLM streams tokens → split into sentences → TTS processes each sentence independently → play sequentially.
- **Pre-warm TTS model**: Keep VoiceBox model loaded (requires more RAM or a dedicated TTS machine).

### 3.3 — Memory Optimization
- **Persistent ASR**: On 32GB+ machines, keep voxtral.c running during TTS. No more stop/start cycle, no deaf time.
- **MLX models**: Use MLX-native models for LLM inference — better Apple Silicon utilization than Ollama's llama.cpp backend.
- **Quantization tuning**: Test Q4 vs Q6 vs Q8 tradeoffs for each component.

---

## Phase 4: Multi-Speaker & Group Calls

Move beyond 1-on-1 to handle real team meetings.

### 4.1 — Speaker Diarization
- **Who's talking**: Integrate pyannote-audio to identify different speakers in the audio stream.
- **Speaker profiles**: Learn to recognize recurring speakers (Mike always sounds like this).
- **Directed detection**: In a 5-person call, only respond when someone is clearly talking to you — not during side conversations.

### 4.2 — Conversation Tracking
- **Multi-turn memory**: Remember what was discussed earlier in the call. If someone asks "going back to what Mike said about the timeline" — the agent knows what that was.
- **Topic segmentation**: Detect when the conversation shifts topics. Associate responses with the right agenda item.
- **Action item detection**: Flag commitments and follow-ups during the call.

---

## Phase 5: Intelligence & Safety

Its not Agentic Conversation at the moment. Convert the interactions into Agentic Conversation and make it more intelligent based on historical transcripts.

### 5.1 — Response Modes
- **Auto mode** (current): Agent speaks automatically when confident.
- **Suggest mode**: Agent drafts a response and shows it on screen. You approve, edit, or dismiss before it speaks. Safer for important calls.
- **Copilot mode**: Agent never speaks, but shows real-time suggested responses as you talk. Like a teleprompter.

### 5.2 — Tone Awareness
- **Sentiment detection**: Is the caller frustrated, confused, or in a hurry? Adjust response style accordingly.
- **Formality matching**: Casual call → casual responses. Executive review → more polished language.
- **Interruption handling**: If someone starts talking while the agent is speaking, stop immediately.

### 5.3 — Guardrails
- **Confidence display**: Always show the agent's confidence level. Low confidence → don't speak, show a warning.
- **Fact verification**: Cross-reference responses against the meeting context. If the agent is about to say something not grounded in the context, flag it.
- **Audit log**: Full record of every decision the agent made and why, for post-call review.

---

## Phase 6: Platform Support

### 6.1 — Beyond Google Meet
- **Zoom, Microsoft Teams, Slack Huddles**: Abstract the audio routing layer to support other platforms.
- **System audio capture**: Platform-agnostic audio capture that works with any video call app.

---

## Contributing

This is a fun side project. If something on this roadmap interests you, open an issue to discuss before sending a PR. The codebase is straightforward — start with `src/orchestrator.py` to understand the flow, then pick a component to improve.

Highest-impact contributions right now:
1. Streaming TTS (Phase 3.2) — biggest latency win
2. Heuristic intent fast-path (Phase 3.1) — skip LLM for obvious questions
3. Hot-reload meeting context (Phase 1.3) — quality of life
4. Suggest mode (Phase 5.1) — safer for real meetings
