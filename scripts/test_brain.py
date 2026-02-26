"""Phase 3 Test: Brain — intent detection + response generation.

Tests the full brain pipeline with simulated meeting transcript snippets.

Usage:
    python scripts/test_brain.py
    python scripts/test_brain.py --interactive   # type custom transcript lines
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brain.context import load_meeting_context, format_context_for_llm
from src.brain.intent import IntentClassifier
from src.brain.responder import Responder
from src.brain.gate import ConfidenceGate, Action

# Simulated meeting transcript snippets for testing
TEST_CASES = [
    {
        "name": "Direct question by name",
        "transcript": "Sarah: The auth module looks solid. Dhaval, what's your take on the rate limiting approach? Should we go IP-based or token-based?",
        "should_respond": True,
    },
    {
        "name": "Indirect reference after topic",
        "transcript": "Mike: I've been looking at the rate limiting options. What do you think, Dhaval?",
        "should_respond": True,
    },
    {
        "name": "Not directed at Dhaval",
        "transcript": "Sarah: Lisa, can you give us an update on the frontend migration? How's it going with the design system?",
        "should_respond": False,
    },
    {
        "name": "General group question",
        "transcript": "Sarah: Does anyone have concerns about the Q1 timeline? Dhaval, you've been closest to the backend work.",
        "should_respond": True,
    },
    {
        "name": "Casual mention, no question",
        "transcript": "Mike: Yeah, Dhaval mentioned that earlier in the standup. Anyway, let's move on to sprint planning.",
        "should_respond": False,
    },
    {
        "name": "Round-robin turn",
        "transcript": "Sarah: Mike, great update. And Dhaval? How's your side looking?",
        "should_respond": True,
    },
]


def test_context_loading():
    print("=" * 60)
    print("Testing: Meeting Context Loading")
    print("=" * 60)

    ctx = load_meeting_context("./meetings/example.md")
    formatted = format_context_for_llm(ctx)

    print(f"  Title: {ctx.title}")
    print(f"  Attendees: {len(ctx.attendees)}")
    print(f"  Your role: {ctx.user_role}")
    print(f"  Key context points: {len(ctx.key_context)}")
    print(f"  Positions: {len(ctx.positions)}")
    print(f"  Avoid items: {len(ctx.avoid)}")
    print(f"  Formatted length: {len(formatted)} chars")
    print()
    return ctx, formatted


def test_intent_and_response(ctx, formatted_context):
    print("=" * 60)
    print("Testing: Intent Classification + Response Generation")
    print("=" * 60)

    classifier = IntentClassifier(
        trigger_names=["Dhaval", "dhaval"],
        ollama_model="llama3.1:8b",
    )
    responder = Responder(
        user_name="Dhaval",
        ollama_model="llama3.1:8b",
    )
    gate = ConfidenceGate(auto_threshold=0.8, confirm_threshold=0.7)

    style = "\n".join(f"- {s}" for s in ctx.communication_style)
    avoid = "\n".join(f"- {a}" for a in ctx.avoid)

    results = []
    for i, tc in enumerate(TEST_CASES):
        print(f"\n--- Test {i+1}: {tc['name']} ---")
        print(f"  Transcript: {tc['transcript'][:100]}...")

        # Intent classification
        t0 = time.monotonic()
        intent = classifier.classify(tc["transcript"], formatted_context)
        t_intent = time.monotonic() - t0

        print(f"  Tier 1 match: {'yes' if classifier.tier1_check(tc['transcript']) else 'no'}")
        print(f"  Intent: directed={intent.directed_at_me}, confidence={intent.confidence:.2f}, "
              f"urgency={intent.urgency} ({t_intent:.2f}s)")
        print(f"  Question: {intent.question_summary}")

        # Gate decision
        decision = gate.decide(intent)
        print(f"  Gate: {decision.action.value} — {decision.reason}")

        # Generate response if needed
        response_text = ""
        t_response = 0.0
        if decision.action == Action.RESPOND:
            t0 = time.monotonic()
            response_text = responder.generate(
                question_summary=intent.question_summary,
                recent_transcript=tc["transcript"],
                meeting_context=formatted_context,
                communication_style=style,
                avoid=avoid,
            )
            t_response = time.monotonic() - t0
            print(f"  Response ({t_response:.2f}s): {response_text}")

        correct = (intent.directed_at_me == tc["should_respond"])
        status = "\033[92mCORRECT\033[0m" if correct else "\033[91mWRONG\033[0m"
        print(f"  Expected directed_at_me={tc['should_respond']} → {status}")

        results.append({
            "name": tc["name"],
            "correct": correct,
            "intent_time": t_intent,
            "response_time": t_response,
        })

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    correct_count = sum(1 for r in results if r["correct"])
    print(f"  Accuracy: {correct_count}/{len(results)}")
    avg_intent = sum(r["intent_time"] for r in results) / len(results)
    print(f"  Avg intent classification: {avg_intent:.2f}s")
    responded = [r for r in results if r["response_time"] > 0]
    if responded:
        avg_resp = sum(r["response_time"] for r in responded) / len(responded)
        print(f"  Avg response generation: {avg_resp:.2f}s")
    print()


def interactive_mode(ctx, formatted_context):
    print("=" * 60)
    print("Interactive Mode — type transcript lines, see classification")
    print("Type 'quit' to exit")
    print("=" * 60)

    classifier = IntentClassifier(
        trigger_names=["Dhaval", "dhaval"],
        ollama_model="llama3.1:8b",
    )
    responder = Responder(
        user_name="Dhaval",
        ollama_model="llama3.1:8b",
    )
    gate = ConfidenceGate()

    style = "\n".join(f"- {s}" for s in ctx.communication_style)
    avoid = "\n".join(f"- {a}" for a in ctx.avoid)

    while True:
        print()
        try:
            line = input("Transcript > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() == "quit":
            break

        t0 = time.monotonic()
        intent = classifier.classify(line, formatted_context)
        t_intent = time.monotonic() - t0

        print(f"  Intent: directed={intent.directed_at_me}, confidence={intent.confidence:.2f}, "
              f"urgency={intent.urgency} ({t_intent:.2f}s)")
        print(f"  Question: {intent.question_summary}")

        decision = gate.decide(intent)
        print(f"  Gate: {decision.action.value} — {decision.reason}")

        if decision.action in (Action.RESPOND, Action.CONFIRM):
            t0 = time.monotonic()
            response = responder.generate(
                question_summary=intent.question_summary,
                recent_transcript=line,
                meeting_context=formatted_context,
                communication_style=style,
                avoid=avoid,
            )
            t_resp = time.monotonic() - t0
            print(f"  Response ({t_resp:.2f}s): {response}")


def main():
    parser = argparse.ArgumentParser(description="Test brain pipeline")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode — type transcript lines")
    args = parser.parse_args()

    ctx, formatted = test_context_loading()

    if args.interactive:
        interactive_mode(ctx, formatted)
    else:
        test_intent_and_response(ctx, formatted)


if __name__ == "__main__":
    main()
