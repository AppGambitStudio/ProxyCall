"""Central orchestrator — wires audio, ASR, brain, and voice together.

State machine:
  IDLE → LISTENING → DETECTING → THINKING → SPEAKING → IDLE
  MUTED overrides everything (still transcribes but won't respond)
"""

import asyncio
import logging
import time
from enum import Enum

import numpy as np
import sounddevice as sd
import yaml

from .audio.capture import AudioCapture
from .audio.devices import find_device
from .asr.voxtral import VoxtralASR
from .transcript.buffer import TranscriptBuffer
from .brain.context import load_meeting_context, format_context_for_llm
from .brain.intent import IntentClassifier
from .brain.responder import Responder
from .brain.gate import ConfidenceGate, Action
from .voice.tts import VoiceBoxTTS

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    DETECTING = "DETECTING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    MUTED = "MUTED"


class Orchestrator:
    """Central event loop connecting all components."""

    def __init__(self, config_path: str = "config.yaml", meeting_path: str | None = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.state = State.IDLE
        self._muted = False
        self._running = False

        # Callbacks for UI updates
        self._on_state_change: list = []
        self._on_transcript: list = []
        self._on_detection: list = []
        self._on_response: list = []
        self._on_latency: list = []
        self._on_status: list = []

        # Components (initialized in start())
        self.capture: AudioCapture | None = None
        self.asr: VoxtralASR | None = None
        self.transcript: TranscriptBuffer | None = None
        self.classifier: IntentClassifier | None = None
        self.responder: Responder | None = None
        self.gate: ConfidenceGate | None = None
        self.tts: VoiceBoxTTS | None = None

        # Meeting context
        meeting_file = meeting_path or self.config["meeting"]["context_file"]
        self.meeting_ctx = load_meeting_context(meeting_file)
        self.formatted_context = format_context_for_llm(
            self.meeting_ctx, self.config["agent"]["name"]
        )

        # Playback device
        self._playback_device: int | None = None

        # Latency tracking
        self.latency = {"asr": 0.0, "intent": 0.0, "llm": 0.0, "tts": 0.0}

        # Silence detection for intent checking
        self._last_speech_time = 0.0
        self._silence_check_interval = self.config["agent"]["silence_timeout_ms"] / 1000
        self._pending_check = False
        self._last_checked_speech_time = 0.0  # track when we last checked

    # --- Event callbacks ---

    def on_state_change(self, cb):
        self._on_state_change.append(cb)

    def on_transcript(self, cb):
        self._on_transcript.append(cb)

    def on_detection(self, cb):
        self._on_detection.append(cb)

    def on_response(self, cb):
        self._on_response.append(cb)

    def on_latency(self, cb):
        self._on_latency.append(cb)

    def on_status(self, cb):
        self._on_status.append(cb)

    def _emit_status(self, msg: str):
        for cb in self._on_status:
            try:
                cb(msg)
            except Exception:
                pass

    def _set_state(self, state: State):
        if self._muted and state != State.MUTED:
            return
        old = self.state
        self.state = state
        if old != state:
            for cb in self._on_state_change:
                try:
                    cb(state)
                except Exception:
                    pass

    # --- Controls ---

    def toggle_mute(self):
        self._muted = not self._muted
        if self._muted:
            self._set_state(State.MUTED)
            sd.stop()
        else:
            self.state = State.IDLE  # bypass _set_state guard
            self._set_state(State.IDLE)
        return self._muted

    def force_respond(self):
        """Force respond to the last utterance."""
        if self.state in (State.SPEAKING, State.THINKING):
            return
        asyncio.get_event_loop().create_task(self._check_and_respond(force=True))

    def skip_response(self):
        """Stop current speech playback."""
        sd.stop()
        self._set_state(State.IDLE)

    # --- Lifecycle ---

    async def start(self):
        """Initialize and start all components."""
        cfg = self.config
        logger.info("Starting orchestrator...")

        # Audio capture
        capture_device = find_device(cfg["audio"]["capture_device"], "input")
        self.capture = AudioCapture(
            device=capture_device,
            sample_rate=cfg["audio"]["sample_rate"],
            block_size=cfg["audio"]["block_size"],
        )

        # Playback device
        self._playback_device = find_device(cfg["audio"]["playback_device"], "output")

        # ASR
        self.asr = VoxtralASR(
            binary_path=cfg["asr"]["binary_path"],
            model_path=cfg["asr"]["model_path"],
            processing_interval=cfg["asr"]["processing_interval"],
        )

        # Transcript buffer
        self.transcript = TranscriptBuffer()

        # Brain
        self.classifier = IntentClassifier(
            trigger_names=cfg["agent"]["trigger_names"],
            ollama_model=cfg["llm"]["model"],
            ollama_base_url=cfg["llm"]["base_url"],
            intent_temperature=cfg["llm"]["intent_temperature"],
            skip_tier1=cfg["agent"].get("skip_name_check", False),
        )
        self.responder = Responder(
            user_name=cfg["agent"]["name"],
            ollama_model=cfg["llm"]["model"],
            ollama_base_url=cfg["llm"]["base_url"],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_tokens"],
            max_sentences=cfg["agent"]["max_response_sentences"],
        )
        self.gate = ConfidenceGate(
            auto_threshold=cfg["agent"]["confidence_threshold"],
        )

        # TTS
        self.tts = VoiceBoxTTS(
            base_url=cfg["tts"]["base_url"],
            profile_id=cfg["tts"]["voice_profile_id"],
            language=cfg["tts"]["language"],
        )

        # Wire ASR output to transcript
        def on_asr_text(text):
            self.transcript.add_text(text)
            self._last_speech_time = time.monotonic()
            if self.state == State.IDLE:
                self._set_state(State.LISTENING)
            for cb in self._on_transcript:
                try:
                    cb(text)
                except Exception:
                    pass

        self.asr.on_transcript(on_asr_text)

        # Start everything — ASR first (model load takes ~30s)
        self.transcript.start_session()
        await self.asr.start()
        logger.info("Waiting for ASR model to load...")
        await self.asr.wait_ready()
        await self.capture.start()
        await self.tts.start()

        self._running = True
        self._set_state(State.IDLE)
        self._emit_status("All systems ready — waiting for colleague to speak...")
        logger.info("Orchestrator started — all components running")

    async def stop(self):
        """Stop all components."""
        self._running = False
        await self.asr.stop()
        await self.capture.stop()
        await self.tts.stop()
        self.transcript.flush()
        logger.info("Orchestrator stopped")

    async def run(self):
        """Main event loop — feed audio and check for intent on silence."""
        try:
            async for chunk in self.capture.stream():
                if not self._running:
                    break

                # Feed audio to ASR
                await self.asr.feed_audio(chunk)

                # Check for silence → trigger intent detection
                now = time.monotonic()
                time_since_speech = now - self._last_speech_time if self._last_speech_time > 0 else 0
                has_new_speech = self._last_speech_time > self._last_checked_speech_time

                if (
                    has_new_speech
                    and time_since_speech > self._silence_check_interval
                    and not self._pending_check
                    and self.state in (State.IDLE, State.LISTENING)
                ):
                    logger.info(
                        "Silence after new speech (%.1fs), checking intent...",
                        time_since_speech,
                    )
                    self._emit_status("Silence detected, analyzing intent...")
                    self._pending_check = True
                    self._last_checked_speech_time = self._last_speech_time
                    self._set_state(State.DETECTING)
                    asyncio.create_task(self._check_and_respond())

        except asyncio.CancelledError:
            pass

    async def _check_and_respond(self, force: bool = False):
        """Check intent and generate response if appropriate."""
        try:
            self._set_state(State.DETECTING)

            recent_text = self.transcript.get_recent_text(60)
            if not recent_text.strip():
                self._set_state(State.IDLE)
                self._pending_check = False
                return

            # Intent classification (sync Ollama call — run in executor)
            loop = asyncio.get_running_loop()
            self._emit_status("Classifying intent via LLM...")
            t0 = time.monotonic()
            intent = await loop.run_in_executor(
                None, self.classifier.classify, recent_text, self.formatted_context
            )
            self.latency["intent"] = time.monotonic() - t0
            self._emit_status(f"Intent done ({self.latency['intent']:.1f}s)")

            # Gate decision
            if force:
                from .brain.gate import GateDecision
                decision = GateDecision(action=Action.RESPOND, reason="Forced", intent=intent)
            else:
                decision = self.gate.decide(intent)

            for cb in self._on_detection:
                try:
                    cb(decision)
                except Exception:
                    pass

            if decision.action != Action.RESPOND or self._muted:
                self._emit_status("Ready — you can speak")
                self._set_state(State.IDLE)
                self._pending_check = False
                return

            # Generate response
            self._set_state(State.THINKING)
            self._emit_status("Generating response via LLM...")

            style = "\n".join(f"- {s}" for s in self.meeting_ctx.communication_style)
            avoid = "\n".join(f"- {a}" for a in self.meeting_ctx.avoid)

            t0 = time.monotonic()
            response_text = await loop.run_in_executor(
                None,
                lambda: self.responder.generate(
                    question_summary=intent.question_summary,
                    recent_transcript=recent_text,
                    meeting_context=self.formatted_context,
                    communication_style=style,
                    avoid=avoid,
                ),
            )
            self.latency["llm"] = time.monotonic() - t0
            self._emit_status(f"Response ready ({self.latency['llm']:.1f}s)")

            for cb in self._on_response:
                try:
                    cb(response_text)
                except Exception:
                    pass

            # Synthesize and play
            self._set_state(State.SPEAKING)

            try:
                logger.info("Synthesizing TTS for: %s", response_text[:60])
                # Stop ASR to free GPU memory for TTS and prevent feedback loop
                self._emit_status("Stopping ASR for TTS...")
                logger.info("Stopping ASR to free GPU for TTS...")
                await self.asr.stop()
                await asyncio.sleep(2)  # wait for GPU memory release
                self._emit_status("Synthesizing voice...")
                t0 = time.monotonic()
                audio, sr = await loop.run_in_executor(
                    None, self.tts.synthesize_sync, response_text
                )
                self.latency["tts"] = time.monotonic() - t0
                self._emit_status(f"TTS done ({self.latency['tts']:.1f}s), speaking...")
                logger.info("TTS done in %.1fs, playing audio (%d samples, %dHz)",
                            self.latency["tts"], len(audio), sr)

                # Update latency callbacks
                for cb in self._on_latency:
                    try:
                        cb(self.latency)
                    except Exception:
                        pass

                # Play through speakers (ASR stays off to prevent feedback loop)
                if not self._muted:
                    try:
                        await loop.run_in_executor(None, self._play_audio, audio, sr)
                        logger.info("Playback complete")
                    except Exception:
                        logger.exception("Playback failed")

            except Exception:
                logger.exception("TTS synthesis failed")
                self._set_state(State.IDLE)
                return
            finally:
                # Restart ASR only after playback is done
                self._emit_status("Restarting ASR...")
                logger.info("Restarting ASR...")
                await self.asr.start()
                await self.asr.wait_ready()
                # Mark current speech as checked to avoid re-triggering
                self._last_checked_speech_time = self._last_speech_time

            self._emit_status("Waiting for colleague to speak...")
            self._set_state(State.IDLE)

        except Exception:
            logger.exception("Error in check_and_respond")
            self._set_state(State.IDLE)
        finally:
            self._pending_check = False

    def _play_audio(self, audio: np.ndarray, sr: int):
        """Play audio synchronously (runs in executor)."""
        sd.play(audio, samplerate=sr, device=self._playback_device)
        sd.wait()
