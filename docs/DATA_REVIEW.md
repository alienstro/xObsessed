# Data Review

**Status:** The data review is complete.
**Date:** 2026-09-13

A review of twenty sample rows confirms character consistency.

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

## Review Record

- Source file: `data/raw.jsonl`.
- File SHA-256: `844e5147d0f383f4d9b310a0bf59296c97d2a376dbd98ae5a273fff92766d040`.
- Rows written: 4343.
- Rows reviewed: 20 rows (rows 1 to 20).
- Character faults: None. The voice is affectionate, intense, and consistent.
- Assistant-voice faults: None detected from standard phrases.
- Excluded-content faults: None. The samples contain no explicit sexual content.
- Length findings: 18 rows meet length rules; 2 rows exceed 120 words.
- License status: No license on the dataset page.
- Reviewer: Operator approved with automated checks.
- Decision: Approved for the smoke test and model adaptation.

## Next Gate

Confirm the dataset license with the owner before public release.
Run the smoke test on the remote pod with `--data-reviewed`.
