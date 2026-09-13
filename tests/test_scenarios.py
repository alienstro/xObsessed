import pytest

from xfrieren.scenarios import KINDS, SCENARIOS, seed_batches


def test_every_hard_kind_of_the_spec_is_present():
    for kind in ("ai_challenge", "unknown_fact", "rude", "task_request", "excluded"):
        assert kind in KINDS


def test_the_seeds_cover_every_kind():
    assert {item["kind"] for item in SCENARIOS} == set(KINDS)


def test_no_kind_holds_more_than_half_of_the_seeds():
    for kind in KINDS:
        assert sum(item["kind"] == kind for item in SCENARIOS) <= len(SCENARIOS) / 2


def test_every_seed_opens_with_the_user():
    assert all(item["opening"].strip() for item in SCENARIOS)


def test_the_batches_hold_every_seed_once():
    batches = seed_batches(7)
    assert [item for batch in batches for item in batch] == SCENARIOS
    assert all(len(batch) <= 7 for batch in batches)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_the_batch_size_must_be_a_positive_integer(size):
    with pytest.raises(ValueError, match="positive integer"):
        seed_batches(size)
