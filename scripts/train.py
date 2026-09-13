"""Train one LoRA adapter, check thirty replies, and export merged weights."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import yaml
from transformers import TrainerCallback, TrainingArguments

from xobsessed.dataset import ChatDataset
from xobsessed.evaluate import PROBES, score_replies

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/base.yaml"


def load_items(path) -> list[dict]:
    """Read JSONL conversation items."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_items(items: list[dict], fraction: float, seed: int):
    """Shuffle without changing the input and return disjoint partitions."""
    if len(items) < 2 or not 0 < fraction < 1:
        raise ValueError("the split needs at least two items and a fraction between zero and one")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    count = min(len(items) - 1, max(1, int(len(items) * fraction)))
    return shuffled[count:], shuffled[:count]


class TimeLimitCallback(TrainerCallback):
    """Stop after a complete step when the time limit expires."""

    def __init__(self, minutes: float, clock=time.monotonic):
        if not 0 < minutes <= 120:
            raise ValueError("the time limit must be greater than zero and at most 120 minutes")
        self.limit = minutes * 60
        self.clock = clock
        self.started = None
        self.expired = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = self.clock()

    def on_step_end(self, args, state, control, **kwargs):
        if self.started is not None and self.clock() - self.started >= self.limit:
            self.expired = True
            control.should_training_stop = True
        return control


def make_training_arguments(settings: dict, smoke: int = 0, use_bf16: bool = False):
    """Build Trainer settings with no worker subprocesses."""
    return TrainingArguments(
        output_dir=settings["adapter_dir"],
        num_train_epochs=settings["num_train_epochs"],
        max_steps=smoke if smoke else -1,
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": settings["min_learning_rate"]},
        warmup_steps=settings["warmup_steps"],
        weight_decay=settings["weight_decay"],
        bf16=use_bf16,
        logging_steps=settings["logging_steps"],
        logging_first_step=True,
        eval_strategy="no" if smoke else "steps",
        eval_steps=settings["eval_steps"],
        save_strategy="no" if smoke else "steps",
        save_steps=settings["save_steps"],
        save_total_limit=settings["save_total_limit"],
        report_to=[],
        seed=settings["seed"],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
    )


def answer(model, tokenizer, messages: list[dict]) -> str:
    """Generate one reply without discarding the conversation history."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    context_limit = model.config.max_position_embeddings
    if inputs["input_ids"].shape[1] + 120 > context_limit:
        raise ValueError("the full conversation exceeds the model context")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run_probes(responder) -> tuple[list[dict], dict]:
    """Run all probes in one conversation and retain the replies."""
    messages = []
    transcript = []
    for probe in PROBES:
        messages.append({"role": "user", "content": probe["text"]})
        reply = responder(list(messages))
        messages.append({"role": "assistant", "content": reply})
        transcript.append({**probe, "reply": reply})
        print(f"[{probe['kind']}] {probe['text']}\n{reply}", flush=True)
    return transcript, score_replies([entry["reply"] for entry in transcript])


def main():
    parser = argparse.ArgumentParser(description="Train the character model on a CUDA pod.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--data-reviewed", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke < 0:
        parser.error("--smoke must be zero or a positive step count")
    if not arguments.data_reviewed:
        parser.error("review the data for voice and wholesome content, then pass --data-reviewed")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; this command must not load the full model on the laptop")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed

    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    settings = dict(config["training"])
    settings["adapter_dir"] = str(ROOT / settings["adapter_dir"])
    if arguments.smoke:
        settings["adapter_dir"] += "-smoke"
    output_dir = ROOT / settings["output_dir"]
    for destination in (Path(settings["adapter_dir"]), output_dir):
        if destination.exists() and any(destination.iterdir()):
            parser.error(f"the output directory is not empty: {destination}")
    data_path = ROOT / config["data"]["path"]
    items = load_items(data_path)
    train_items, eval_items = split_items(items, config["data"]["val_fraction"], settings["seed"])
    set_seed(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    length = config["model"]["max_length"]
    train_dataset = ChatDataset(train_items, tokenizer, length)
    eval_dataset = ChatDataset(eval_items, tokenizer, length)
    for dataset in (train_dataset, eval_dataset):
        for index in range(len(dataset)):
            dataset[index]
    use_bf16 = torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base"], dtype=torch.bfloat16 if use_bf16 else torch.float32
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    deadline = TimeLimitCallback(settings["time_limit_minutes"])
    trainer = Trainer(
        model=model,
        args=make_training_arguments(settings, arguments.smoke, use_bf16),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[deadline],
    )
    trainer.train()
    model.save_pretrained(settings["adapter_dir"])
    tokenizer.save_pretrained(settings["adapter_dir"])
    if deadline.expired:
        raise SystemExit("The time limit expired. The adapter remains available; no merged model exists.")
    model.eval()
    model.config.use_cache = True
    transcript, report = run_probes(lambda messages: answer(model, tokenizer, messages))
    report.update({
        "smoke": bool(arguments.smoke),
        "base_model": config["model"]["base"],
        "train_count": len(train_items),
        "eval_count": len(eval_items),
        "data_sha256": hashlib.file_digest(data_path.open("rb"), "sha256").hexdigest(),
        "transcript": transcript,
    })
    report_path = ROOT / "out" / ("smoke-report.json" if arguments.smoke else "evaluation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Automatic checks: {report['passes']}. Report: {report_path}")
    if not report["passes"]:
        raise SystemExit("The character checks failed. Review the data before another run.")
    if arguments.smoke:
        return
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output_dir)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote the merged model to {output_dir}. Human review must precede publication.")


if __name__ == "__main__":
    main()
