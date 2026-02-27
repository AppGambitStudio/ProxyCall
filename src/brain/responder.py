"""Response generator — generates contextual meeting responses via Ollama."""

import logging
import time

import ollama

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1.0


class Responder:
    """Generates responses on behalf of the user using LLM."""

    def __init__(
        self,
        user_name: str = "Dhaval",
        ollama_model: str = "llama3.1:8b",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 200,
        max_sentences: int = 3,
    ):
        self.user_name = user_name
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.ollama_base_url = ollama_base_url

    def _call_ollama(self, messages: list[dict]) -> str:
        """Call Ollama with retry logic for transient network errors."""
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = ollama.Client(host=self.ollama_base_url)
                response = client.chat(
                    model=self.ollama_model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                    think=False,
                )
                return response["message"]["content"].strip()
            except (ConnectionError, OSError) as e:
                last_err = e
                logger.warning(
                    "Ollama connection attempt %d/%d failed: %s",
                    attempt, MAX_RETRIES, e,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        raise last_err

    def generate(
        self,
        question_summary: str,
        recent_transcript: str,
        meeting_context: str = "",
        communication_style: str = "",
        avoid: str = "",
    ) -> str:
        """Generate a response to a question directed at the user.

        Args:
            question_summary: What's being asked (from intent classifier).
            recent_transcript: Last ~2 minutes of conversation.
            meeting_context: Formatted meeting context.
            communication_style: How the user communicates.
            avoid: Things the user should not say.

        Returns:
            Response text ready for TTS.
        """
        system_prompt = self._build_system_prompt(
            meeting_context, communication_style, avoid
        )

        user_prompt = f"""Recent conversation:
{recent_transcript}

Question directed at {self.user_name}:
{question_summary}

Generate {self.user_name}'s response ({self.max_sentences} sentences max, spoken aloud in a meeting):"""

        start = time.monotonic()

        try:
            logger.info("Responder LLM input: %s", user_prompt[:300])

            raw = self._call_ollama(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            elapsed = time.monotonic() - start
            logger.info("Responder LLM raw output (%0.1fs): %s", elapsed, repr(raw[:300]))

            # Clean up: remove thinking tags, quotes, stage directions, etc.
            text = self._clean_response(raw)
            logger.info("Responder cleaned output: %s", text[:200])

            if not text:
                logger.warning("LLM returned empty response after cleaning (raw: %s)", raw[:200])
                return "Let me get back to you on that."
            return text

        except Exception as e:
            logger.error("Response generation failed: %s", e)
            return "Let me get back to you on that."

    def _build_system_prompt(
        self, meeting_context: str, communication_style: str, avoid: str
    ) -> str:
        parts = [
            f"You are generating a spoken response on behalf of {self.user_name} in a live meeting.",
            f"Respond as if you ARE {self.user_name} speaking aloud. Use first person.",
            "",
            "IMPORTANT: The transcript comes from a small ASR model and may contain misheard words,",
            "phonetic errors, or garbled text. Use the meeting context and conversation flow to infer",
            "what was actually said. Respond to the intended meaning, not the literal transcription.",
            "",
            "Rules:",
            f"- Maximum {self.max_sentences} sentences",
            "- Sound natural and conversational — this will be spoken aloud",
            "- Be direct — start with the answer, then brief reasoning",
            "- If unsure about specifics, say 'Let me get back to you on that' rather than guessing",
            "- Never use markdown, bullet points, or formatting — plain spoken text only",
            "- Don't start with 'Sure,' or 'Great question' — just answer directly",
        ]

        if meeting_context:
            parts.extend(["", "Meeting context:", meeting_context])
        if communication_style:
            parts.extend(["", "Communication style:", communication_style])
        if avoid:
            parts.extend(["", "Things to NEVER say:", avoid])

        return "\n".join(parts)

    def _clean_response(self, text: str) -> str:
        """Clean LLM output for TTS."""
        import re

        # Strip Qwen3 thinking tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove quotes wrapping the response
        text = text.strip('"\'')
        # Remove stage directions like *pauses* or (thinking)
        text = re.sub(r"\*[^*]+\*", "", text)
        text = re.sub(r"\([^)]+\)", "", text)
        # Remove any markdown
        text = re.sub(r"[*_#`]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
