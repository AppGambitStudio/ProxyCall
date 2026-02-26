"""Intent classifier — detects if the last utterance needs a response.

For 2-person calls: everything is directed at the user, so the only
question is "does this need a verbal response?"
"""

import json
import logging
import re
from dataclasses import dataclass

import ollama

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    needs_response: bool
    confidence: float  # 0.0 to 1.0
    question_summary: str
    urgency: str  # "immediate", "can_wait", "fyi_only"


class IntentClassifier:
    """Classifies whether the last utterance needs a response."""

    def __init__(
        self,
        trigger_names: list[str],
        ollama_model: str = "llama3.1:8b",
        ollama_base_url: str = "http://localhost:11434",
        intent_temperature: float = 0.1,
        skip_tier1: bool = False,
    ):
        self.user_name = trigger_names[0] if trigger_names else "the user"
        self.ollama_model = ollama_model
        self.intent_temperature = intent_temperature
        self._client = ollama.Client(host=ollama_base_url)

    def classify(self, transcript: str, meeting_context: str = "") -> IntentResult:
        """Classify whether the transcript needs a response."""
        logger.info("Running intent classification")

        system_prompt = f"""You decide if the last thing said in a 2-person call needs a verbal response from {self.user_name}.

This is a natural 1-on-1 conversation. Think about what a real person would reply to.

needs_response = true:
- Questions: "How are you?", "What's the status?", "Can you confirm?"
- Requests: "Let me know", "Please share", "Walk me through"
- Goodbyes and sign-offs: "Have a good evening", "Talk to you soon", "Take care" — always say goodbye back
- Greetings: "Good morning", "Hey, how's it going?"
- Expecting a reply: speaker paused and is waiting

needs_response = false:
- Mid-sentence or still talking (hasn't finished their thought)
- Brief mid-conversation filler: "ok", "right", "uh huh", "got it", "I see"

{f"Context: {meeting_context}" if meeting_context else ""}

Reply with ONLY valid JSON:
{{"needs_response": true/false, "confidence": 0.0-1.0, "summary": "what to respond to"}}"""

        try:
            user_msg = f"Transcript:\n{transcript}"
            logger.info("Intent input: %s", transcript[:200])

            response = self._client.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": self.intent_temperature},
                think=False,
            )

            raw = response["message"].content.strip()
            logger.info("Intent output: %s", raw[:300])

            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                logger.warning("No JSON in intent response: %s", raw)
                return IntentResult(
                    needs_response=True,
                    confidence=0.6,
                    question_summary="Could not parse intent",
                    urgency="can_wait",
                )

            data = json.loads(json_match.group())
            return IntentResult(
                needs_response=data.get("needs_response", True),
                confidence=float(data.get("confidence", 0.7)),
                question_summary=data.get("summary", ""),
                urgency="immediate" if data.get("needs_response") else "fyi_only",
            )

        except Exception as e:
            logger.error("Intent classification failed: %s", e)
            return IntentResult(
                needs_response=True,
                confidence=0.5,
                question_summary="Classification failed",
                urgency="can_wait",
            )
