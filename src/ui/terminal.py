"""Rich terminal UI for the voice agent."""

import logging
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..orchestrator import State

logger = logging.getLogger(__name__)

STATE_COLORS = {
    State.IDLE: "dim",
    State.LISTENING: "green",
    State.DETECTING: "yellow",
    State.THINKING: "cyan",
    State.SPEAKING: "blue",
    State.MUTED: "red",
}

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
        self.transcript_lines: deque[str] = deque(maxlen=12)
        self.events: deque[str] = deque(maxlen=6)
        self.current_response = ""
        self.latency = {"intent": 0.0, "llm": 0.0, "tts": 0.0}

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
                self.transcript_lines.append(line.strip())
        # Also split on sentence boundaries for long text
        for end in ".!?":
            while end + " " in self._transcript_buffer:
                idx = self._transcript_buffer.index(end + " ")
                line = self._transcript_buffer[: idx + 1].strip()
                self._transcript_buffer = self._transcript_buffer[idx + 2 :]
                if line:
                    self.transcript_lines.append(line)
        # Show current partial line
        if len(self._transcript_buffer) > 80:
            self.transcript_lines.append(self._transcript_buffer.strip())
            self._transcript_buffer = ""
        self._refresh()

    def on_detection(self, decision):
        from ..brain.gate import Action
        intent = decision.intent
        if intent.directed_at_me:
            self.events.append(
                f"[yellow]>> Detected question (confidence: {intent.confidence:.0%})[/]"
            )
            self.events.append(f"   [dim]{intent.question_summary}[/]")
            if decision.action == Action.RESPOND:
                self.events.append("[green]>> Generating response...[/]")
            elif decision.action == Action.CONFIRM:
                self.events.append("[yellow]>> Awaiting confirmation (press F)[/]")
        self._refresh()

    def on_response(self, text: str):
        self.current_response = text
        self.events.append(f"[blue]>> Speaking:[/] {text[:70]}...")
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
            transcript_text.append(line + "\n")
        if self._transcript_buffer.strip():
            transcript_text.append(self._transcript_buffer.strip(), style="dim")
        layout["transcript"].update(
            Panel(transcript_text, title="Transcript", border_style="green")
        )

        # Events
        events_text = Text()
        for event in self.events:
            events_text.append_text(Text.from_markup(event + "\n"))
        layout["events"].update(
            Panel(events_text, title="Agent Activity", border_style="cyan")
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
            self._live.update(self._build_display())

    def start(self) -> Live:
        """Create and return the Live context manager."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            screen=True,
        )
        return self._live
