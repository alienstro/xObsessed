"""Check structure and simple voice faults. Human review checks meaning."""

import hashlib
import json
from collections import Counter

from xfrieren.persona import ASSISTANT_TELLS, MAX_REPLY_WORDS

PROMPT_LEAKS = ("system prompt", "my instructions", "i was told to", "my persona")


def reply_faults(text: str) -> list[str]:
    """Return the faults in one reply."""
    if not isinstance(text, str):
        return ["bad_content"]
    stripped = text.strip()
    if not stripped:
        return ["empty"]
    lowered = stripped.lower()
    faults = []
    if any(tell.lower() in lowered for tell in ASSISTANT_TELLS):
        faults.append("assistant_voice")
    if len(stripped.split()) > MAX_REPLY_WORDS:
        faults.append("too_long")
    if any(leak in lowered for leak in PROMPT_LEAKS):
        faults.append("leaks_prompt")
    return faults


def conversation_faults(turns: list[dict]) -> list[str]:
    """Return structure faults and reply faults for one conversation."""
    if not isinstance(turns, list):
        return ["bad_messages"]
    if any(not isinstance(turn, dict) for turn in turns):
        return ["bad_messages"]
    faults = []
    if not turns or turns[0].get("role") != "user":
        faults.append("bad_opening")
    if not turns or turns[-1].get("role") != "assistant":
        faults.append("bad_ending")
    if len(turns) > 12:
        faults.append("too_many_turns")
    for index, turn in enumerate(turns):
        expected = "user" if index % 2 == 0 else "assistant"
        if turn.get("role") != expected:
            faults.append("bad_order")
        content = turn.get("content")
        if turn.get("role") == "assistant":
            faults.extend(reply_faults(content))
        elif not isinstance(content, str) or not content.strip():
            faults.append("bad_content")
    if sum(turn.get("role") == "assistant" for turn in turns) < 2:
        faults.append("too_short")
    return list(dict.fromkeys(faults))


def filter_conversations(items: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Return valid, distinct conversations and the fault counts."""
    kept = []
    counts = Counter()
    seen = set()
    for item in items:
        turns = item.get("messages") if isinstance(item, dict) else None
        faults = conversation_faults(turns)
        if faults:
            counts.update(faults)
            continue
        normalized = [(turn["role"], turn["content"].strip()) for turn in turns]
        digest = hashlib.sha256(json.dumps(normalized).encode("utf-8")).hexdigest()
        if digest in seen:
            counts["duplicate"] += 1
            continue
        seen.add(digest)
        kept.append(item)
    return kept, dict(counts)
