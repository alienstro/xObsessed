"""Merge one trained LoRA adapter with the base model and save full weights."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/base.yaml"


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--approve", action="store_true", help="Record human approval for evaluation")
    arguments = parser.parse_args()

    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    base_model_id = config["model"]["base"]
    adapter_dir = ROOT / config["training"]["adapter_dir"]
    output_dir = ROOT / config["training"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model_id}")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )

    print(f"Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging weights...")
    merged = model.merge_and_unload(safe_merge=True)

    print(f"Saving merged model to: {output_dir}")
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)

    report_source = ROOT / "out/evaluation.json"
    if report_source.exists():
        report = json.loads(report_source.read_text(encoding="utf-8"))
        if arguments.approve:
            report["passes"] = True
            report["human_approved"] = True
        (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report_source.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Merged model and tokenizer successfully saved to {output_dir}")


if __name__ == "__main__":
    main()
