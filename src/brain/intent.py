"""Intent classifier — detects if the last utterance needs a response.

For 2-person calls: everything is directed at the user, so the only
question is "does this need a verbal response?"
"""

import json
import logging
import re
import time
from dataclasses import dataclass

import ollama

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


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
        self.ollama_base_url = ollama_base_url
        self.intent_temperature = intent_temperature

    def _make_client(self) -> ollama.Client:
        """Create a fresh ollama client."""
        return ollama.Client(host=self.ollama_base_url)

    def _call_ollama(self, messages: list[dict], temperature: float) -> str:
        """Call Ollama with retry logic for transient network errors."""
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = self._make_client()
                response = client.chat(
                    model=self.ollama_model,
                    messages=messages,
                    options={"temperature": temperature},
                    think=False,
                )
                return response["message"].content.strip()
            except (ConnectionError, OSError) as e:
                last_err = e
                logger.warning(
                    "Ollama connection attempt %d/%d failed: %s",
                    attempt, MAX_RETRIES, e,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        raise last_err

    def classify(self, transcript: str, meeting_context: str = "") -> IntentResult:
        """Classify whether the transcript needs a response."""
        logger.info("Running intent classification")

        system_prompt = f"""You decide if the last thing said in a 2-person call needs a verbal response from {self.user_name}.

This is a natural 1-on-1 conversation. Think about what a real person would reply to.

IMPORTANT: The transcript comes from a small ASR model and may contain misheard words, phonetic errors, or garbled text. Use the meeting context and conversation flow to infer what was actually said. For example "Naval" or "Dahwal" likely means "{self.user_name}", "a pie" might mean "API", etc. Focus on the intent behind the words, not the exact transcription.

needs_response = true (ALWAYS respond to these):
- Any question, direct or indirect
- Greetings: "Good morning", "Hey", "How are you?"
- Goodbyes and wrap-ups: "Talk soon", "Let's catch up Monday", "Have a good weekend", "See you", "Alright bye" — ALWAYS say goodbye back
- Requests: "Let me know", "Please share", "Can you check"
- Proposals or suggestions: "Let's do X", "How about Monday?", "We could try..."
- Expecting a reply: speaker finished talking and is waiting

needs_response = false (stay silent):
- Speaker is clearly mid-sentence and still talking
- Very brief filler while the other person continues: "ok", "uh huh", "right"

{f"Context: {meeting_context}" if meeting_context else ""}

Reply with ONLY valid JSON:
{{"needs_response": true/false, "confidence": 0.0-1.0, "summary": "what to respond to"}}"""

        try:
            logger.info("Intent input: %s", transcript[:200])

            raw = self._call_ollama(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transcript:\n{transcript}"},
                ],
                temperature=self.intent_temperature,
            )
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
            logger.error("Intent classification failed: %s: %s", type(e).__name__, e)
            return IntentResult(
                needs_response=True,
                confidence=0.5,
                question_summary="Classification failed",
                urgency="can_wait",
            )
