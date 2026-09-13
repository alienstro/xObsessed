"""Download the source dataset and write one JSONL conversation for each row."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/base.yaml"


def row_to_item(row: dict):
    """Return one conversation item, or None when the row is incomplete."""
    if not isinstance(row, dict):
        return None
    user_text = str(row.get("input") or "").strip() or str(row.get("instruction") or "").strip()
    reply = str(row.get("output") or "").strip()
    if not user_text or not reply:
        return None
    return {"messages": [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]}


def write_jsonl(items, path: Path) -> int:
    """Write the items and return the written count."""
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for item in items:
            if item is None:
                continue
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Download the source dataset and write JSONL.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    source = config["data"]["source"]
    output = arguments.output or ROOT / config["data"]["path"]
    from datasets import load_dataset

    dataset = load_dataset(source["dataset"], split=source["split"], revision=source["revision"])
    items = [row_to_item(row) for row in dataset]
    output.parent.mkdir(parents=True, exist_ok=True)
    written = write_jsonl(items, output)
    digest = hashlib.file_digest(output.open("rb"), "sha256").hexdigest()
    print(f"rows read: {len(dataset)}, rows written: {written}, skipped: {len(dataset) - written}")
    print(f"sha256: {digest}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
