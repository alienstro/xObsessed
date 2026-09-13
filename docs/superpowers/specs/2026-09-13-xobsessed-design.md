# xObsessed Design Specification

**Date:** 2026-09-13
**Status:** Approved

## 1. Goal

Adapt a small language model to one obsessive, affectionate character.
The model uses the public dataset `Skorcht/yandere-her-dataset` as its data source.
The project removes the local data-generation pipeline.

## 2. Decisions

| Decision | Choice |
|---|---|
| Character | The dataset persona (obsessive and affectionate). Frieren leaves the project |
| Turn shape | One user/assistant pair for each row |
| Data access | A prepare script writes JSONL. `train.py` reads the JSONL |
| Quality filter | Removed |
| System prompt | Removed |
| Project name | `xObsessed` (correct spelling) |

## 3. The data

The dataset holds 4,345 rows with the columns `instruction`, `input`, and `output`.
The page shows no license and no dataset card.

**Caution:** The dataset has no stated license. A published model that derives from it
carries a legal risk. Confirm the license with the dataset owner before a public release.

`scripts/prepare_data.py` does one job:

1. Download the dataset with a pinned `revision` (a commit hash), so a run repeats.
2. Map each row to `{"messages": [{"role": "user", "content": <input or instruction>}, {"role": "assistant", "content": <output>}]}`.
3. Write `data/raw.jsonl`.
4. Print the row count and the sha256 of the output file.

The user text uses `input` when it is not empty. The user text uses `instruction` otherwise.

## 4. The files to remove

| File | Reason |
|---|---|
| `scripts/generate_data.py` | The HF dataset replaces it |
| `src/xfrieren/scenarios.py` | It holds the generation seeds only |
| `src/xfrieren/persona.py` | The project drops the system prompt |
| `src/xfrieren/quality.py` | The project drops the quality filter |
| `tests/test_generate_data.py` | The test covers removed code |
| `tests/test_scenarios.py` | The test covers removed code |
| `tests/test_quality.py` | The test covers removed code |
| `tests/test_persona.py` | The test covers removed code |

## 5. The rename

The package moves from `src/xfrieren/` to `src/xobsessed/`.
The distribution name moves from `xfrieren` to `xobsessed`.
Every `from xfrieren...` import changes to `from xobsessed...`.
The model name in `configs/base.yaml` changes to `xobsessed-1.7b`.

These locations also carry the old name:

| File | Old value | New value |
|---|---|---|
| `.env` | `HF_REPO_ID=Alienstro/xFrieren` | `HF_REPO_ID=Alienstro/xObsessed` |
| `.env` | `WANDB_PROJECT=xFrieren` | `WANDB_PROJECT=xObsessed` |
| `scripts/pod.sh` | `REMOTE_DIR="/workspace/xFrieren"` | `/workspace/xObsessed` |
| `scripts/pod.sh` | tmux session `xfrieren-job` | `xobsessed-job` |
| `scripts/quantize.sh` | `NAME="${3:-xfrieren-1.7b}"` | `xobsessed-1.7b` |
| `scripts/runpod.py` | `User-Agent: xfrieren/0.1` | `xobsessed/0.1` |

## 6. The edits to the kept files

| File | Change |
|---|---|
| `dataset.py` | Stop injecting `SYSTEM_PROMPT`. Keep the multi-turn label logic, because one pair is a valid case |
| `train.py` | Remove the `filter_conversations` call and the quality gate. Keep the data sha256 in the report |
| `evaluate.py` | Rewrite the probes for the new character. Remove the `quality` import |
| `card.py` | Rewrite for the new character and the new data source. Remove the "original text only" claim, because the data is third-party |
| `push_to_hub.py` | Remove the `system_prompt.txt` requirement |
| `configs/base.yaml` | Set `publish.model_name` to `xobsessed-1.7b`. Add the dataset id and the revision |
| `AGENTS.md` | Rename and update the description |
| `docs/RUNBOOK.md` | Rename and update the steps |
| `docs/DATA_REVIEW.md` | Rewrite for the new dataset |

## 7. Testing

- Delete the four removed test files.
- Add `tests/test_prepare_data.py`. It checks the row mapping and the empty-`input` fallback with a fake row.
- Update the remaining tests for the rename.
- Run `pytest`. All tests must pass.

## 8. What this project does not do

It does not copy or generate its own training text.
It does not filter the data for voice or length.
It does not ship a system prompt.

## 9. What the model will still do wrong

The dataset teaches one affectionate voice. The model can still produce content that
the operator excludes. The model card must state the data source and the missing data
license.
