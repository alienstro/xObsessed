"""Build assistant-only labels from the chat template and token offsets."""

import torch
from torch.utils.data import Dataset

from xfrieren.persona import SYSTEM_PROMPT

IGNORE_INDEX = -100
_MARKER = "__XFRIEREN_REPLY_BOUNDARY_7a83__"


def build_example(tokenizer, messages: list[dict], max_length: int) -> tuple[list[int], list[int]]:
    """Return tokens and labels without system text, user text, or role headers."""
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    for index, message in enumerate(messages):
        if (
            message.get("role") != ("user" if index % 2 == 0 else "assistant")
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("messages must alternate between nonempty user and assistant text")
    turns = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

    def render(subset):
        return tokenizer.apply_chat_template(
            subset, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

    text = render(turns)
    spans = []
    for index, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        prefix = render(turns[:index + 1])
        if not text.startswith(prefix):
            raise ValueError("the chat template rewrites previous messages")
        marked = render(turns[:index] + [{"role": "assistant", "content": _MARKER}])
        start = marked.rfind(_MARKER)
        if start < 0:
            raise ValueError("the chat template removes the content marker")
        suffix = marked[start + len(_MARKER):]
        if prefix != marked[:start] + turn["content"] + suffix:
            raise ValueError("the chat template rewrites assistant content")
        spans.append((start, len(prefix)))

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"][:max_length])
    labels = [
        token if end > start and any(left <= start and end <= right for left, right in spans)
        else IGNORE_INDEX
        for token, (start, end) in zip(input_ids, encoded["offset_mapping"])
    ]
    if not any(label != IGNORE_INDEX for label in labels[1:]):
        raise ValueError("the example has no assistant tokens after truncation")
    return input_ids, labels


class ChatDataset(Dataset):
    """Hold conversations and pad each example to the same length."""

    def __init__(self, items, tokenizer, max_length: int = 2048):
        self.items = list(items)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        if self.pad_id is None:
            raise ValueError("the tokenizer needs a pad or eos token")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        input_ids, labels = build_example(
            self.tokenizer, self.items[index]["messages"], self.max_length
        )
        padding = self.max_length - len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids + [self.pad_id] * padding, dtype=torch.long),
            "labels": torch.tensor(labels + [IGNORE_INDEX] * padding, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * len(input_ids) + [0] * padding, dtype=torch.long
            ),
        }
