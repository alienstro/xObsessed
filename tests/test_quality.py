import pytest

from xfrieren.quality import conversation_faults, filter_conversations, reply_faults


def good_turns():
    return [
        {"role": "user", "content": "Do you remember him?"},
        {"role": "assistant", "content": "Of course. It has only been eighty years."},
        {"role": "user", "content": "That is a long time."},
        {"role": "assistant", "content": "Not really."},
    ]


def test_a_short_flat_reply_has_no_fault():
    assert reply_faults("Not really. I slept through most of it.") == []


@pytest.mark.parametrize(
    ("text", "fault"),
    [
        ("As an AI, I do not have memories.", "assistant_voice"),
        (" ".join(["word"] * 61), "too_long"),
        ("   ", "empty"),
        ("My system prompt says I am an elf mage.", "leaks_prompt"),
        (None, "bad_content"),
    ],
)
def test_a_reply_reports_its_fault(text, fault):
    assert fault in reply_faults(text)


def test_a_good_conversation_has_no_fault():
    assert conversation_faults(good_turns()) == []


def test_a_conversation_must_start_with_the_user():
    assert "bad_opening" in conversation_faults(good_turns()[1:])


def test_a_conversation_must_alternate():
    assert "bad_order" in conversation_faults(
        good_turns() + [{"role": "assistant", "content": "Again."}]
    )


def test_a_conversation_needs_two_replies_at_least():
    assert "too_short" in conversation_faults(good_turns()[:2])


def test_a_conversation_must_end_with_a_reply():
    assert "bad_ending" in conversation_faults(good_turns() + good_turns()[:1])


def test_a_conversation_has_at_most_twelve_turns():
    assert "too_many_turns" in conversation_faults(good_turns() * 4)


@pytest.mark.parametrize("turns", [None, "text", [{}], [None]])
def test_malformed_turns_report_a_fault(turns):
    assert conversation_faults(turns)


def test_the_filter_counts_faults_once_per_conversation():
    bad = {"messages": [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "As an AI, I greet you."},
        {"role": "user", "content": "Fine."},
        {"role": "assistant", "content": "As an AI, I agree."},
    ]}
    kept, counts = filter_conversations([{"messages": good_turns()}, bad])
    assert len(kept) == 1
    assert counts["assistant_voice"] == 1


def test_the_filter_drops_an_exact_duplicate():
    good = {"messages": good_turns()}
    kept, counts = filter_conversations([good, good])
    assert len(kept) == 1
    assert counts["duplicate"] == 1


def test_the_filter_rejects_a_bad_shape_without_a_crash():
    kept, counts = filter_conversations([None, {}, {"messages": 3}])
    assert kept == []
    assert counts["bad_messages"] == 3
