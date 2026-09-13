"""Verify current file hashes before an operator can delete a pod."""

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "out/upload-manifest.json"
REQUIRED = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "README.md",
    "evaluation.json",
    "gguf/{name}-BF16.gguf",
    "gguf/{name}-F16.gguf",
    "gguf/{name}-Q8_0.gguf",
    "gguf/{name}-Q6_K.gguf",
    "gguf/{name}-Q4_K_M.gguf",
]


def check_files(present, model_name: str) -> list[str]:
    """Return absent files, with support for a weight index and shards."""
    have = set(present)
    if "model.safetensors.index.json" in have and any(
        name.startswith("model-") and name.endswith(".safetensors") for name in have
    ):
        have.add("model.safetensors")
    return [
        name.format(name=model_name) for name in REQUIRED
        if name.format(name=model_name) not in have
    ]


def verify_manifest(api, repo_id: str, manifest: dict, read_remote=None) -> list[str]:
    """Compare a local manifest with one immutable Hub revision."""
    from huggingface_hub import hf_hub_download

    revision = api.repo_info(repo_id=repo_id, repo_type="model").sha
    if read_remote is None:
        def read_remote(name, commit):
            path = hf_hub_download(
                repo_id, name, revision=commit, token=api.token, repo_type="model"
            )
            return Path(path).read_bytes()

    remote_manifest = json.loads(read_remote("upload-manifest.json", revision))
    if remote_manifest != manifest:
        return ["manifest_mismatch"]
    files = manifest["files"]
    faults = [f"missing:{name}" for name in check_files(files, manifest["model_name"])]
    if faults:
        return faults
    entries = api.get_paths_info(
        repo_id=repo_id, paths=list(files), revision=revision, repo_type="model"
    )
    remote_files = {entry.path: entry for entry in entries}
    for name, expected in files.items():
        entry = remote_files.get(name)
        if entry is None:
            faults.append(f"missing:{name}")
            continue
        if getattr(entry, "size", None) != expected["size"]:
            faults.append(f"size_mismatch:{name}")
            continue
        if getattr(entry, "lfs", None) is not None:
            digest = entry.lfs.sha256
        elif entry.size <= 10 * 1024 * 1024:
            digest = hashlib.sha256(read_remote(name, revision)).hexdigest()
        else:
            faults.append(f"unverified_large_file:{name}")
            continue
        if digest != expected["sha256"]:
            faults.append(f"hash_mismatch:{name}")
    if "model.safetensors.index.json" in files:
        index = json.loads(read_remote("model.safetensors.index.json", revision))
        for shard in set(index["weight_map"].values()):
            if shard not in files:
                faults.append(f"missing_shard:{shard}")
    return faults


def main():
    parser = argparse.ArgumentParser(description="Verify the current upload by file hash.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    arguments = parser.parse_args()
    if not os.environ.get("HF_REPO_ID") or not os.environ.get("HF_TOKEN"):
        parser.error("set HF_REPO_ID and HF_TOKEN")
    if not arguments.manifest.is_file():
        parser.error("the local manifest is absent; file names alone cannot verify this run")
    from huggingface_hub import HfApi

    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    faults = verify_manifest(
        HfApi(token=os.environ["HF_TOKEN"]), os.environ["HF_REPO_ID"], manifest
    )
    if faults:
        raise SystemExit("Upload verification failed: " + ", ".join(faults))
    print(f"Verified every required file for run {manifest['run_id']}.")
    print("Confirm the GGUF download and human review before pod termination.")


if __name__ == "__main__":
    main()
