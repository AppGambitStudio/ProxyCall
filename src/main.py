"""Voice Agent — main entry point.

Usage:
    python -m src.main
    python -m src.main --meeting meetings/example.md
    python -m src.main --listen-only
    python -m src.main --no-ui
"""

import argparse
import asyncio
import logging
import sys
import termios
import tty

from .orchestrator import Orchestrator
from .ui.terminal import TerminalUI

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler("voiceagent.log"), logging.StreamHandler(sys.stderr)],
    )
    # Keep agent logs at INFO/DEBUG, silence noisy HTTP internals
    logging.getLogger("src").setLevel(logging.INFO if not debug else logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def read_keys(orchestrator: Orchestrator, ui: TerminalUI | None):
    """Read keyboard input in raw mode."""
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        while True:
            key = await loop.run_in_executor(None, lambda: sys.stdin.read(1))

            if key.lower() == "q":
                return  # quit

            elif key.lower() == "m":
                muted = orchestrator.toggle_mute()
                if ui:
                    ui.events.append(
                        f"[red]>> MUTED[/]" if muted else "[green]>> UNMUTED[/]"
                    )

            elif key.lower() == "f":
                orchestrator.force_respond()
                if ui:
                    ui.events.append("[yellow]>> Force responding...[/]")

            elif key.lower() == "s":
                orchestrator.skip_response()
                if ui:
                    ui.events.append("[dim]>> Skipped response[/]")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


async def run(args):
    setup_logging(args.debug)

    orchestrator = Orchestrator(
        config_path=args.config,
        meeting_path=args.meeting,
    )

    # Setup UI
    ui = None
    if not args.no_ui:
        ui = TerminalUI(
            meeting_title=orchestrator.meeting_ctx.title,
            meeting_context_points=len(orchestrator.meeting_ctx.key_context),
        )
        orchestrator.on_state_change(ui.on_state_change)
        orchestrator.on_transcript(ui.on_transcript)
        orchestrator.on_detection(ui.on_detection)
        orchestrator.on_response(ui.on_response)
        orchestrator.on_latency(ui.on_latency)

    # Start orchestrator
    await orchestrator.start()

    if args.listen_only:
        orchestrator._muted = True
        orchestrator.state = orchestrator.state  # trigger UI update

    try:
        if ui:
            live = ui.start()
            with live:
                # Run main loop and key reader concurrently
                main_task = asyncio.create_task(orchestrator.run())
                key_task = asyncio.create_task(read_keys(orchestrator, ui))

                # Wait for either to finish (key_task finishes on 'q')
                done, pending = await asyncio.wait(
                    [main_task, key_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
        else:
            # No UI — just print transcript to stdout
            def print_transcript(text):
                sys.stdout.write(text)
                sys.stdout.flush()

            orchestrator.on_transcript(print_transcript)

            print(f"Voice Agent running (meeting: {orchestrator.meeting_ctx.title})")
            print("Press Ctrl+C to stop.\n")

            await orchestrator.run()

    except KeyboardInterrupt:
        pass
    finally:
        await orchestrator.stop()

    # Print session summary
    if orchestrator.transcript:
        full_text = orchestrator.transcript.get_all_text()
        if full_text:
            print(f"\n--- Session Transcript ---\n{full_text}\n")


def main():
    parser = argparse.ArgumentParser(description="Voice Agent")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--meeting", type=str, help="Meeting context file")
    parser.add_argument("--listen-only", action="store_true", help="Transcribe only, don't respond")
    parser.add_argument("--no-ui", action="store_true", help="Disable Rich terminal UI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
