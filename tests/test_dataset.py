import re

import pytest
import torch

from xfrieren.dataset import IGNORE_INDEX, ChatDataset, build_example
from xfrieren.persona import SYSTEM_PROMPT


class FakeTokenizer:
    """Assign one token to each word or boundary marker."""

    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        assert kwargs.get("enable_thinking") is False
        return "".join(f"<{m['role']}> {m['content']} <end> " for m in messages)

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": list(range(1, len(matches) + 1)),
            "offset_mapping": [match.span() for match in matches],
        }


def conversation():
    return [
        {"role": "user", "content": "Do you remember him"},
        {"role": "assistant", "content": "Of course"},
        {"role": "user", "content": "That is long"},
        {"role": "assistant", "content": "Not really"},
    ]


def test_the_labels_cover_reply_content_and_end_markers_alone():
    tokenizer = FakeTokenizer()
    ids, labels = build_example(tokenizer, conversation(), max_length=256)
    assert len(ids) == len(labels)
    learned = [value for value in labels if value != IGNORE_INDEX]
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}] + conversation(),
        enable_thinking=False,
    )
    words = text.split()
    assert [words[value - 1] for value in learned] == [
        "Of", "course", "<end>", "Not", "really", "<end>"
    ]
    assert labels[0] == IGNORE_INDEX


def test_every_reply_contributes_tokens():
    _, short = build_example(FakeTokenizer(), conversation()[:2], 256)
    _, long = build_example(FakeTokenizer(), conversation(), 256)
    assert sum(value != IGNORE_INDEX for value in long) == 2 * sum(
        value != IGNORE_INDEX for value in short
    )


def test_a_long_example_stops_at_the_maximum_length():
    ids, labels = build_example(FakeTokenizer(), conversation() * 40, 200)
    assert len(ids) == len(labels) == 200


def test_truncation_cannot_remove_all_reply_tokens():
    with pytest.raises(ValueError, match="no assistant tokens"):
        build_example(FakeTokenizer(), conversation(), 4)


def test_the_dataset_pads_with_zero_and_masks_the_padding():
    dataset = ChatDataset([{"messages": conversation()}], FakeTokenizer(), 256)
    item = dataset[0]
    assert item["input_ids"].shape == (256,)
    padding = item["attention_mask"] == 0
    assert padding.any()
    assert torch.all(item["labels"][padding] == IGNORE_INDEX)
    assert torch.all(item["input_ids"][padding] == 0)


def test_the_dataset_reports_its_length():
    assert len(ChatDataset([{"messages": conversation()}] * 3, FakeTokenizer(), 256)) == 3


def test_the_dataset_rejects_a_missing_pad_and_eos_token():
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="pad"):
        ChatDataset([{"messages": conversation()}], tokenizer)


def test_a_template_cannot_silently_rewrite_previous_messages():
    class UnstableTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return str(len(messages)) + super().apply_chat_template(messages, **kwargs)

    with pytest.raises(ValueError, match="template"):
        build_example(UnstableTokenizer(), conversation(), 256)
