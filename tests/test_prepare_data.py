import json

from prepare_data import row_to_item, write_jsonl


def row(input_text="", instruction="What do you admire about me?", output="You, always."):
    return {"instruction": instruction, "input": input_text, "output": output}


def test_the_input_field_becomes_the_user_text():
    item = row_to_item(row(input_text="Tell me a secret."))
    assert item == {"messages": [
        {"role": "user", "content": "Tell me a secret."},
        {"role": "assistant", "content": "You, always."},
    ]}


def test_the_instruction_field_becomes_the_user_text_when_input_is_empty():
    item = row_to_item(row(input_text=""))
    assert item["messages"][0]["content"] == "What do you admire about me?"


def test_whitespace_input_falls_back_to_the_instruction():
    item = row_to_item(row(input_text="   "))
    assert item["messages"][0]["content"] == "What do you admire about me?"


def test_an_empty_output_is_skipped():
    assert row_to_item(row(output="")) is None
    assert row_to_item(row(output="   ")) is None


def test_an_empty_user_text_is_skipped():
    assert row_to_item(row(input_text="", instruction="")) is None


def test_write_jsonl_writes_one_line_for_each_item(tmp_path):
    path = tmp_path / "raw.jsonl"
    items = [row_to_item(row(input_text="Hi.")), row_to_item(row(input_text="Bye."))]
    count = write_jsonl(items, path)
    assert count == 2
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines == items


def test_write_jsonl_skips_a_none_item(tmp_path):
    path = tmp_path / "raw.jsonl"
    count = write_jsonl([row_to_item(row(input_text="Hi.")), None], path)
    assert count == 1
