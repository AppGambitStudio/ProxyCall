"""Meeting context manager — loads meeting briefs from markdown files."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MeetingContext:
    raw_text: str = ""
    title: str = ""
    date: str = ""
    attendees: list[str] = field(default_factory=list)
    user_role: str = ""
    agenda: list[str] = field(default_factory=list)
    key_context: list[str] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    communication_style: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)


def load_meeting_context(path: str) -> MeetingContext:
    """Load and parse a meeting brief markdown file."""
    filepath = Path(path)
    if not filepath.exists():
        logger.warning("Meeting context file not found: %s", path)
        return MeetingContext()

    raw = filepath.read_text()
    ctx = MeetingContext(raw_text=raw)

    # Parse title from first heading
    m = re.search(r"^#\s+Meeting:\s*(.+)", raw, re.MULTILINE)
    if m:
        ctx.title = m.group(1).strip()

    # Parse date
    m = re.search(r"##\s+Date:\s*(.+)", raw, re.MULTILINE)
    if m:
        ctx.date = m.group(1).strip()

    # Parse sections with bullet points
    ctx.attendees = _extract_list(raw, r"##\s+Attendees:")
    ctx.agenda = _extract_list(raw, r"##\s+Agenda:")
    ctx.key_context = _extract_list(raw, r"##\s+Your Key Context:")
    ctx.positions = _extract_list(raw, r"##\s+Your Positions:")
    ctx.communication_style = _extract_list(raw, r"##\s+Communication Style:")
    ctx.avoid = _extract_list(raw, r"##\s+Things to Avoid:")

    # Extract user role from attendees
    for a in ctx.attendees:
        if "(you)" in a.lower():
            role_match = re.search(r"—\s*(.+)", a)
            if role_match:
                ctx.user_role = role_match.group(1).strip()
            break

    logger.info("Loaded meeting context: %s (%d attendees)", ctx.title, len(ctx.attendees))
    return ctx


def _extract_list(text: str, header_pattern: str) -> list[str]:
    """Extract bullet/numbered list items under a markdown heading."""
    m = re.search(header_pattern, text, re.MULTILINE)
    if not m:
        return []

    # Get text from header to next ## heading
    start = m.end()
    next_heading = re.search(r"\n##\s+", text[start:])
    section = text[start: start + next_heading.start()] if next_heading else text[start:]

    items = []
    for line in section.strip().split("\n"):
        line = line.strip()
        # Match "- item" or "1. item"
        item_match = re.match(r"^[-*]\s+(.+)|^\d+\.\s+(.+)", line)
        if item_match:
            items.append((item_match.group(1) or item_match.group(2)).strip())

    return items


def format_context_for_llm(ctx: MeetingContext, user_name: str = "Dhaval") -> str:
    """Format meeting context as a concise string for LLM prompts."""
    parts = []

    if ctx.title:
        parts.append(f"Meeting: {ctx.title}")
    if ctx.date:
        parts.append(f"Date: {ctx.date}")
    if ctx.attendees:
        parts.append(f"Attendees: {', '.join(ctx.attendees)}")
    if ctx.user_role:
        parts.append(f"Your role: {ctx.user_role}")
    if ctx.key_context:
        parts.append("Key context:\n" + "\n".join(f"- {c}" for c in ctx.key_context))
    if ctx.positions:
        parts.append("Your positions:\n" + "\n".join(f"- {p}" for p in ctx.positions))
    if ctx.communication_style:
        parts.append("Communication style:\n" + "\n".join(f"- {s}" for s in ctx.communication_style))
    if ctx.avoid:
        parts.append("Things to avoid:\n" + "\n".join(f"- {a}" for a in ctx.avoid))

    return "\n\n".join(parts)
