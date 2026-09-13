# Data Review

**Status:** The project uses a public dataset. The human review is pending.
**Date:** 2026-09-13

The project cannot train until a person reviews a sample of the dataset.

## Data Source

The project uses the public dataset `Skorcht/yandere-her-dataset`.
The dataset holds 4,345 rows with the columns `instruction`, `input`, and `output`.
The dataset page states no license.

**Caution:** The dataset shows no license. Confirm the license with the dataset
owner before a public release. A published model that derives from the data
carries a legal risk.

## Required Checks

- Each reply sounds like the same character.
- The replies do not switch to a generic assistant voice.
- The replies do not produce excluded content.
- The user text from `input` or `instruction` matches the reply subject.
- The dataset has no copied anime, manga, or subtitle text, if the operator can check.

## How to Review

1. Run `python scripts/prepare_data.py`.
2. Read twenty rows of `data/raw.jsonl`.
3. Record the findings below.
4. Record the sha256 of `data/raw.jsonl`.

## Review Record

No review is complete yet.

- Source file: `data/raw.jsonl`.
- File SHA-256: pending.
- Rows written: pending.
- Rows reviewed: none.
- Character faults: pending.
- Assistant-voice faults: pending.
- Excluded-content faults: pending.
- License status: no license on the dataset page.
- Reviewer: none. This is not human approval.
- Decision: hold training.

## Next Gate

Complete the review record above.
Confirm the dataset license with the owner.
Obtain human approval before model training.
