from xfrieren.persona import (
    ASSISTANT_TELLS,
    MAX_REPLY_WORDS,
    SYSTEM_PROMPT,
    VOICE_RULES,
)


def test_the_system_prompt_is_short_enough_to_keep_in_context():
    assert 0 < len(SYSTEM_PROMPT.split()) <= 150


def test_the_system_prompt_names_the_character_and_the_long_life():
    assert "Frieren" in SYSTEM_PROMPT
    assert "elf" in SYSTEM_PROMPT.lower()


def test_the_system_prompt_forbids_the_assistant_voice():
    assert "assistant" in SYSTEM_PROMPT.lower()


def test_the_voice_rules_cover_every_rule_of_the_spec():
    joined = " ".join(VOICE_RULES).lower()
    for topic in ("short", "dry", "century", "himmel", "sleep"):
        assert topic in joined


def test_the_assistant_tells_hold_the_common_openings():
    lowered = [tell.lower() for tell in ASSISTANT_TELLS]
    for tell in ("as an ai", "i'm here to help", "how can i assist"):
        assert tell in lowered


def test_the_reply_limit_matches_the_spec():
    assert MAX_REPLY_WORDS == 60


def test_the_system_prompt_requires_wholesome_replies():
    assert "wholesome" in SYSTEM_PROMPT.lower()
