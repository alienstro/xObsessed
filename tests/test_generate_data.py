import json
from types import SimpleNamespace

import pytest

from generate_data import ask, build_request, generate, parse_conversation, write_jsonl


def item(opening="Hello."):
    return {"messages": [
        {"role": "user", "content": opening},
        {"role": "assistant", "content": "You are early."},
        {"role": "user", "content": "Should I wait?"},
        {"role": "assistant", "content": "For a decade."},
    ]}


def test_the_request_holds_the_rules_and_the_opening():
    request = build_request({"kind": "rude", "opening": "You are useless."}, 6)
    assert "You are useless." in request
    assert "flat" in request.lower()
    assert "6 messages" in request
    assert "Do not copy" in request
    assert "wholesome" in request


@pytest.mark.parametrize("turns", [0, 3, 14])
def test_the_request_rejects_an_invalid_turn_count(turns):
    with pytest.raises(ValueError):
        build_request({"kind": "everyday", "opening": "Hello."}, turns)


def test_the_parser_reads_json_and_a_code_fence():
    body = json.dumps(item())
    assert parse_conversation(body) == item()
    assert parse_conversation(f"```json\n{body}\n```") == item()


@pytest.mark.parametrize(
    "body", ["not json", "[]", "null", "{}", '{"messages": 3}', '{"messages": [{}]}']
)
def test_the_parser_rejects_a_bad_shape(body):
    with pytest.raises(ValueError):
        parse_conversation(body)


def test_the_writer_writes_one_object_on_each_line(tmp_path):
    path = tmp_path / "nested" / "out.jsonl"
    assert write_jsonl(path, [item(), item()]) == 2
    assert len(path.read_text().splitlines()) == 2


def test_the_api_call_rejects_a_truncated_answer():
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(stop_reason="max_tokens", content=[])))
    with pytest.raises(ValueError, match="end_turn"):
        ask(client, "request")


def test_generation_counts_accepted_items_and_preserves_progress(tmp_path):
    calls = []

    def request(seed, turns):
        calls.append(seed)
        if len(calls) == 1:
            return "invalid"
        return json.dumps(item(seed["opening"]))

    path = tmp_path / "out.jsonl"
    kept, counts = generate(request, 2, path, max_attempts=8, turn_counts=(4,))
    assert len(kept) == 2
    assert counts["invalid_answer"] == 1
    assert len(path.read_text().splitlines()) == 2
    assert all("kind" in entry for entry in kept)


def test_generation_stops_at_the_attempt_limit(tmp_path):
    with pytest.raises(RuntimeError, match="attempt limit"):
        generate(lambda seed, turns: "invalid", 2, tmp_path / "out.jsonl", max_attempts=3)


def test_generation_does_not_overwrite_an_existing_file(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text("keep this")
    with pytest.raises(FileExistsError):
        generate(lambda seed, turns: "", 2, path)
    assert path.read_text() == "keep this"
