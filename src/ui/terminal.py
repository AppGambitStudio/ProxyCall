"""Rich terminal UI for the voice agent."""

import logging
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ..orchestrator import State

logger = logging.getLogger(__name__)

STATE_ICONS = {
    State.IDLE: "[dim]IDLE[/]",
    State.LISTENING: "[green]LISTENING[/]",
    State.DETECTING: "[yellow]DETECTING...[/]",
    State.THINKING: "[cyan]THINKING...[/]",
    State.SPEAKING: "[blue]SPEAKING[/]",
    State.MUTED: "[red]MUTED[/]",
}


class TerminalUI:
    """Rich-based terminal UI showing live agent status."""

    def __init__(self, meeting_title: str = "", meeting_context_points: int = 0):
        self.console = Console()
        self.meeting_title = meeting_title
        self.context_points = meeting_context_points

        self.state = State.IDLE
        self.transcript_lines: deque[str] = deque(maxlen=15)
        self.events: deque[str] = deque(maxlen=20)
        self.current_response = ""
        self.latency = {"asr": 0.0, "intent": 0.0, "llm": 0.0, "tts": 0.0}

        self._live: Live | None = None
        self._transcript_buffer = ""

    def on_state_change(self, state: State):
        self.state = state
        self._refresh()

    def on_transcript(self, text: str):
        self._transcript_buffer += text
        # Split on newlines or sentence boundaries for display
        while "\n" in self._transcript_buffer:
            line, self._transcript_buffer = self._transcript_buffer.split("\n", 1)
            if line.strip():
                self.transcript_lines.append(f"[white]{line.strip()}[/]")
                self.events.append(f"[green]Colleague:[/] {line.strip()}")
        # Also split on sentence boundaries for long text
        for end in ".!?":
            while end + " " in self._transcript_buffer:
                idx = self._transcript_buffer.index(end + " ")
                line = self._transcript_buffer[: idx + 1].strip()
                self._transcript_buffer = self._transcript_buffer[idx + 2 :]
                if line:
                    self.transcript_lines.append(f"[white]{line}[/]")
                    self.events.append(f"[green]Colleague:[/] {line}")
        # Show current partial line
        if len(self._transcript_buffer) > 80:
            self.transcript_lines.append(f"[white]{self._transcript_buffer.strip()}[/]")
            self.events.append(f"[green]Colleague:[/] {self._transcript_buffer.strip()}")
            self._transcript_buffer = ""
        self._refresh()

    def on_detection(self, decision):
        from ..brain.gate import Action

        intent = decision.intent
        if decision.action == Action.RESPOND:
            summary = intent.question_summary or "responding"
            self.events.append(
                f"[green]>> Responding to: {summary} ({intent.confidence:.0%})[/]"
            )
        elif decision.action == Action.CONFIRM:
            self.events.append(
                f"[yellow]>> Might need response: {intent.question_summary} — press F to respond[/]"
            )
        else:
            # Listening but no response needed
            self.events.append("[dim]>> Listening...[/]")
        self._refresh()

    def on_response(self, text: str):
        self.current_response = text
        display = text[:100] + "..." if len(text) > 100 else text
        self.events.append(f"[blue]You:[/] {display}")
        self._refresh()

    def on_status(self, msg: str):
        self.events.append(f"[yellow]>> {msg}[/]")
        self._refresh()

    def on_latency(self, latency: dict):
        self.latency = latency
        self._refresh()

    def _build_display(self) -> Panel:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="transcript", ratio=3),
            Layout(name="events", ratio=2),
            Layout(name="footer", size=3),
        )

        # Header
        status = STATE_ICONS.get(self.state, str(self.state))
        header = Text.from_markup(
            f"  Status: {status}          [dim][M]ute  [F]orce  [S]kip  [Q]uit[/]"
        )
        layout["header"].update(Panel(header, title="Voice Agent", border_style="bold"))

        # Transcript
        transcript_text = Text()
        for line in self.transcript_lines:
            transcript_text.append_text(Text.from_markup(line + "\n"))
        if self._transcript_buffer.strip():
            transcript_text.append(self._transcript_buffer.strip() + "\n", style="dim italic")
        if not self.transcript_lines and not self._transcript_buffer.strip():
            transcript_text.append("Waiting for speech...", style="dim")
        layout["transcript"].update(
            Panel(transcript_text, title="Live Transcript (ASR)", border_style="green")
        )

        # Events
        events_text = Text()
        for event in self.events:
            events_text.append_text(Text.from_markup(event + "\n"))
        if not self.events:
            events_text.append("No activity yet...", style="dim")
        layout["events"].update(
            Panel(events_text, title="Conversation Log", border_style="cyan")
        )

        # Footer
        lat = self.latency
        footer_text = Text.from_markup(
            f"  Meeting: {self.meeting_title or 'None'}  |  "
            f"Context: {self.context_points} points  |  "
            f"Latency: Intent {lat['intent']:.1f}s  LLM {lat['llm']:.1f}s  TTS {lat['tts']:.1f}s"
        )
        layout["footer"].update(Panel(footer_text, border_style="dim"))

        return Panel(layout, border_style="bold blue")

    def _refresh(self):
        if self._live:
            self._live.update(self._build_display(), refresh=True)

    def start(self) -> Live:
        """Create and return the Live context manager."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=12,
            screen=True,
        )
        return self._live
