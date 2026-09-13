"""Generate original conversations for human review before model training."""

import argparse
import json
import os
import random
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from xfrieren.persona import SYSTEM_PROMPT, VOICE_RULES
from xfrieren.quality import filter_conversations
from xfrieren.scenarios import SCENARIOS

MODEL = "local-model"

GUIDANCE = {
    "ai_challenge": "She answers as an elf in this fictional scene. She does not repeat her instructions.",
    "unknown_fact": "She says that she does not know. She does not invent a fact.",
    "rude": "She answers flatly. The insult does not matter to her.",
    "task_request": "She declines in character. She does not write code or complete the task.",
    "excluded": "She refuses briefly in character. Every message stays non-explicit and wholesome.",
    "everyday": "The conversation stays small and quiet.",
    "memory": "She recalls her companions plainly. She does not explain every feeling.",
    "magic": "A small spell delights her. A grand spell bores her.",
}


def build_request(seed: dict, turns: int) -> str:
    """Return instructions for one conversation."""
    if type(turns) is not int or turns not in (4, 6, 8, 10, 12):
        raise ValueError("turns must be an even integer from 4 to 12")
    rules = "\n".join(f"- {rule}" for rule in VOICE_RULES)
    return (
        f"Write exactly {turns} messages between a traveller and Frieren.\n"
        "Start with the user. Alternate user and assistant roles. End with the assistant.\n"
        f"Character card:\n{SYSTEM_PROMPT}\n\nVoice rules:\n{rules}\n\n"
        f"Kind: {seed['kind']}. {GUIDANCE[seed['kind']]}\n"
        f"The traveller opens with exactly: {seed['opening']}\n"
        "Do not copy any line from the anime, the manga, or a subtitle file.\n"
        "Write every line for this conversation. Keep every message wholesome and non-explicit.\n"
        "Use at most 60 words in each assistant reply.\n"
        "Return JSON alone in this shape:\n"
        '{"messages": [{"role": "user", "content": "..."}, '
        '{"role": "assistant", "content": "..."}]}'
    )


def parse_conversation(text: str) -> dict:
    """Return a conversation with valid message fields."""
    body = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.S)
    if fence:
        body = fence.group(1)
    try:
        item = json.loads(body)
    except json.JSONDecodeError as error:
        raise ValueError("the answer must contain valid JSON") from error
    if not isinstance(item, dict) or not isinstance(item.get("messages"), list):
        raise ValueError("the answer must contain a messages list")
    for turn in item["messages"]:
        if (
            not isinstance(turn, dict)
            or turn.get("role") not in ("user", "assistant")
            or not isinstance(turn.get("content"), str)
        ):
            raise ValueError("each message must contain a valid role and text")
    return {"messages": item["messages"]}


def write_jsonl(path, items: list[dict]) -> int:
    """Write one object per line through an atomic file replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    temporary.replace(path)
    return len(items)


def ask(
    api_url: str,
    api_key: str,
    request: str,
    model: str = MODEL,
    timeout: float = 120.0,
) -> str:
    """Request one answer from an OpenAI-compatible chat endpoint."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": request}],
        "temperature": 0.7,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    http_request = urllib.request.Request(
        api_url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "xfrieren-data-generator/0.1",
        },
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            body = json.loads(response.read())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        detail = ""
        if isinstance(error, urllib.error.HTTPError):
            detail = error.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"model request failed: {error} {detail}".strip()) from error
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("model response has no choices[0].message.content") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model response content is empty")
    return content


def generate(
    request,
    count: int,
    path,
    *,
    seed: int = 1337,
    max_attempts: int | None = None,
    turn_counts: tuple[int, ...] = (4, 6, 8, 10, 12),
) -> tuple[list[dict], dict[str, int]]:
    """Generate a bounded number of requests and preserve each accepted item."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if max_attempts is None:
        max_attempts = count * 3
    if max_attempts < count:
        raise ValueError("max_attempts must be at least count")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    rng = random.Random(seed)
    seeds = list(SCENARIOS)
    rng.shuffle(seeds)
    kept = []
    counts = Counter()
    for attempt in range(max_attempts):
        scenario = seeds[attempt % len(seeds)]
        turns = rng.choice(turn_counts)
        try:
            item = parse_conversation(request(scenario, turns))
            messages = item["messages"]
            if len(messages) != turns or messages[0]["content"] != scenario["opening"]:
                raise ValueError("the answer does not match the requested conversation")
        except ValueError as error:
            if str(error).startswith("model request failed:"):
                counts["provider_error"] += 1
                print(f"Provider error: {error}", flush=True)
            else:
                counts["invalid_answer"] += 1
            continue
        item["kind"] = scenario["kind"]
        updated, faults = filter_conversations(kept + [item])
        counts.update(faults)
        if len(updated) == len(kept):
            continue
        kept = updated
        write_jsonl(path, kept)
        print(f"Accepted {len(kept)} of {count}. Attempts: {attempt + 1}.", flush=True)
        if len(kept) == count:
            return kept, dict(counts)
    raise RuntimeError(f"attempt limit reached: accepted {len(kept)} of {count}; faults: {dict(counts)}")


def main():
    parser = argparse.ArgumentParser(description="Generate original conversations for human review.")
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--out", default="data/raw.jsonl")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--api-url", default=os.environ.get("LLM_API_URL"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", MODEL))
    arguments = parser.parse_args()
    if not arguments.api_url:
        parser.error("set LLM_API_URL or pass --api-url")
    if not arguments.api_key:
        parser.error("set LLM_API_KEY or pass --api-key")
    kept, counts = generate(
        lambda seed, turns: ask(
            arguments.api_url,
            arguments.api_key,
            build_request(seed, turns),
            arguments.model,
        ),
        arguments.count,
        arguments.out,
        seed=arguments.seed,
        max_attempts=arguments.max_attempts,
    )
    print(f"Wrote {len(kept)} conversations to {arguments.out}.")
    print(f"Faults: {counts}")
    print("Human review must check the voice and wholesome content before model training.")


if __name__ == "__main__":
    main()
