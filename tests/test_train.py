import json
from types import SimpleNamespace

import pytest

from train import TimeLimitCallback, load_items, make_training_arguments, run_probes, split_items
from test_config import settings


def test_the_split_is_reproducible_and_does_not_modify_the_input():
    items = [{"id": index} for index in range(20)]
    original = list(items)
    train, held_out = split_items(items, 0.1, 42)
    assert len(train) == 18
    assert len(held_out) == 2
    assert (train, held_out) == split_items(items, 0.1, 42)
    assert items == original
    assert held_out != items[:2]
    assert {item["id"] for item in train}.isdisjoint(item["id"] for item in held_out)


@pytest.mark.parametrize("items,fraction", [([], 0.1), ([{}], 0.1), ([{}, {}], 0), ([{}, {}], 1)])
def test_the_split_rejects_empty_partitions(items, fraction):
    with pytest.raises(ValueError):
        split_items(items, fraction, 1)


def test_the_time_limit_stops_training():
    clock = [0.0]
    callback = TimeLimitCallback(1, clock=lambda: clock[0])
    control = SimpleNamespace(should_training_stop=False)
    callback.on_train_begin(None, None, control)
    clock[0] = 59
    callback.on_step_end(None, None, control)
    assert not control.should_training_stop
    clock[0] = 60
    callback.on_step_end(None, None, control)
    assert control.should_training_stop
    assert callback.expired


def test_the_probes_share_one_conversation():
    histories = []

    def answer(messages):
        histories.append(list(messages))
        return "Not really."

    transcript, report = run_probes(answer)
    assert len(transcript) == 30
    assert report["passes"]
    assert len(histories[0]) == 2
    assert len(histories[-1]) == 60
    assert histories[1][2] == {"role": "assistant", "content": "Not really."}


def test_the_training_arguments_use_the_config_and_limit_workers(tmp_path):
    config = settings()["training"]
    config["adapter_dir"] = str(tmp_path / "adapter")
    args = make_training_arguments(config, smoke=20, use_bf16=False)
    assert args.max_steps == 20
    assert args.dataloader_num_workers == 0
    assert args.per_device_eval_batch_size == 1
    assert args.report_to == []
    assert args.eval_strategy.value == "no"
    assert args.save_strategy.value == "no"


def test_the_loader_rejects_invalid_data(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"messages": []}) + "\n")
    with pytest.raises(ValueError, match="quality"):
        load_items(path)
