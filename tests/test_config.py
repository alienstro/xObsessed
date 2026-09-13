from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def settings():
    return yaml.safe_load((ROOT / "configs/base.yaml").read_text())


def test_the_base_model_is_the_one_that_the_spec_names():
    assert settings()["model"]["base"] == "Qwen/Qwen3-1.7B"


def test_the_adapter_reaches_every_projection():
    assert set(settings()["lora"]["target_modules"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }


def test_the_rank_matches_the_spec():
    assert settings()["lora"]["r"] == 32
    assert settings()["lora"]["alpha"] == 64


def test_the_run_has_a_time_limit():
    assert 0 < settings()["training"]["time_limit_minutes"] <= 120


def test_the_rate_and_split_match_the_spec():
    assert settings()["training"]["learning_rate"] == 1e-4
    assert settings()["data"]["val_fraction"] == 0.1
