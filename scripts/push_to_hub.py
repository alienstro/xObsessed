"""Publish only reviewed artifacts and a manifest for the current run."""

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import yaml

from xobsessed.card import build_card

ROOT = Path(__file__).resolve().parents[1]


def build_manifest(files: dict[str, Path], model_name: str, run_id: str) -> dict:
    """Hash each file sequentially without loading weights into memory."""
    entries = {}
    for name, path in files.items():
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        entries[name] = {"sha256": digest, "size": path.stat().st_size}
    return {"run_id": run_id, "model_name": model_name, "files": entries}


def main():
    parser = argparse.ArgumentParser(description="Publish reviewed merged weights and optional GGUF files.")
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--adapter", type=Path)
    arguments = parser.parse_args()
    if not arguments.reviewed:
        parser.error("review the data, character replies, and license, then pass --reviewed")
    repo_id = os.environ.get("HF_REPO_ID")
    token = os.environ.get("HF_TOKEN")
    if not repo_id or not token:
        parser.error("set HF_REPO_ID and HF_TOKEN")
    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    merged = ROOT / config["training"]["output_dir"]
    model_name = config["publish"]["model_name"]
    report = json.loads((merged / "evaluation.json").read_text(encoding="utf-8"))
    if report.get("passes") is not True or report.get("smoke") is not False:
        parser.error("the full run must pass the automatic checks before publication")
    if report.get("base_model") != config["model"]["base"]:
        parser.error("the run uses a different base model")
    model_config = json.loads((merged / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") not in ("qwen2", "qwen3"):
        parser.error("the merged model must use the Qwen2 or Qwen3 architecture")
    tokenizer_config = json.loads((merged / "tokenizer_config.json").read_text(encoding="utf-8"))
    if not tokenizer_config.get("chat_template") and not (merged / "chat_template.jinja").is_file():
        parser.error("the merged tokenizer has no chat template")
    if not (merged / "model.safetensors").is_file() and not (merged / "model.safetensors.index.json").is_file():
        parser.error("the merged weights are absent")
    (merged / "README.md").write_text(build_card(model_name, repo_id), encoding="utf-8")
    files = {
        path.name: path for path in sorted(merged.iterdir())
        if path.is_file() and path.suffix in (".json", ".safetensors", ".jinja", ".txt", ".md")
    }
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if name not in files:
            parser.error(f"the merged model lacks {name}")
    if arguments.gguf:
        for quant in config["publish"]["quant_types"]:
            name = f"{model_name}-{quant}.gguf"
            path = arguments.gguf / name
            if not path.is_file():
                parser.error(f"the GGUF file is absent: {path}")
            files[f"gguf/{name}"] = path
    if arguments.adapter:
        for path in sorted(arguments.adapter.iterdir()):
            if path.is_file() and path.suffix in (".json", ".safetensors", ".jinja", ".txt", ".md"):
                files[f"adapter/{path.name}"] = path
    manifest = build_manifest(files, model_name, uuid.uuid4().hex)
    manifest_path = ROOT / "out/upload-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files["upload-manifest.json"] = manifest_path
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=path) for name, path in files.items()],
        commit_message="Publish reviewed character model artifacts",
        num_threads=1,
    )
    print(f"Uploaded run {manifest['run_id']} to {repo_id}.")
    print("A new repository stays private until the operator approves public release.")


if __name__ == "__main__":
    main()
