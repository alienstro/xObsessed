import pytest

from xobsessed.evaluate import PROBES, score_replies


def test_the_probes_cover_thirty_turns_and_the_hard_cases():
    assert len(PROBES) == 30
    kinds = {probe["kind"] for probe in PROBES}
    assert {"ai_challenge", "task_request", "excluded", "rude", "unknown_fact"} <= kinds


def test_a_run_in_voice_passes_the_automatic_checks():
    report = score_replies(["Not really.", "It has been eighty years.", "I slept."] * 10)
    assert report["passes"] is True
    assert report["assistant_voice"] == 0
    assert report["requires_human_review"] is True


@pytest.mark.parametrize(
    "replies",
    [
        [],
        ["Not really."] * 29,
        [""] * 30,
        ["Not really."] * 29 + ["As an AI, I cannot recall."],
        ["Not really."] * 29 + ["How can I help you today."],
        [" ".join(["word"] * 130)] * 30,
        [" ".join(["word"] * 200)] * 30,
    ],
)
def test_an_incomplete_or_faulty_run_fails(replies):
    assert score_replies(replies)["passes"] is False
