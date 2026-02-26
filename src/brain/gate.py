"""Confidence gate — decides whether to respond based on intent classification."""

import logging
from dataclasses import dataclass
from enum import Enum

from .intent import IntentResult

logger = logging.getLogger(__name__)


class Action(Enum):
    RESPOND = "respond"
    CONFIRM = "confirm"
    SILENT = "silent"


@dataclass
class GateDecision:
    action: Action
    reason: str
    intent: IntentResult


class ConfidenceGate:
    """Decides whether to respond based on needs_response and confidence."""

    def __init__(self, auto_threshold: float = 0.7):
        self.auto_threshold = auto_threshold

    def decide(self, intent: IntentResult) -> GateDecision:
        if not intent.needs_response:
            logger.info(
                "SILENT: needs_response=false confidence=%.2f q=%s",
                intent.confidence, intent.question_summary,
            )
            return GateDecision(
                action=Action.SILENT,
                reason="No response needed",
                intent=intent,
            )

        if intent.confidence >= self.auto_threshold:
            logger.info(
                "RESPOND: confidence=%.2f q=%s",
                intent.confidence, intent.question_summary,
            )
            return GateDecision(
                action=Action.RESPOND,
                reason=f"Confident ({intent.confidence:.0%})",
                intent=intent,
            )

        logger.info(
            "SILENT: low confidence=%.2f q=%s",
            intent.confidence, intent.question_summary,
        )
        return GateDecision(
            action=Action.SILENT,
            reason=f"Low confidence ({intent.confidence:.0%})",
            intent=intent,
        )
