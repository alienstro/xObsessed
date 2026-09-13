# Runbook

**Date:** 2026-09-13

## Requirements

- Use a CUDA RunPod pod with enough disk for the base model and output.
- Set the keys in `.env` from `.env.example`.
- Review `docs/DATA_REVIEW.md` before training.
- Keep one active remote command.

## Generate Data

Run the sample first.

```bash
uv run python scripts/generate_data.py --count 40 --out data/sample.jsonl
```

Read twenty conversations and complete `docs/DATA_REVIEW.md`.

Generate the full set only after the review.

```bash
uv run python scripts/generate_data.py --count 1200 --out data/raw.jsonl
```

## Prepare the Pod

Warning: `scripts/watchdog.sh` stops the pod after a deadline or a silent period.

```bash
scripts/pod.sh check
scripts/pod.sh sync
scripts/pod.sh data
scripts/pod.sh secrets
scripts/pod.sh setup
scripts/pod.sh run bash scripts/watchdog.sh 150 15
```

## Smoke Test

Run the smoke test only after the data review.

```bash
scripts/pod.sh run scripts/train.py --smoke 20 --data-reviewed
scripts/pod.sh watch
```

Read `out/smoke-report.json`. Stop when the first loss exceeds 5 or the replies
break character. Do not start the full run after a failed smoke test.

## Full Run

```bash
scripts/pod.sh run scripts/train.py --data-reviewed
scripts/pod.sh watch
```

Read `out/evaluation.json`. Check all probe replies by hand.

## Publish and Quantize

Warning: Keep the pod until the upload verifier reports success.

```bash
scripts/pod.sh run scripts/push_to_hub.py --reviewed
scripts/pod.sh run bash scripts/quantize.sh out/merged out/gguf xfrieren-1.7b
scripts/pod.sh run scripts/push_to_hub.py --reviewed --gguf out/gguf
scripts/pod.sh verify
```

Download the Q8 GGUF and ask three probes with `--jinja`.

Do not run `scripts/pod.sh terminate --yes` until the downloaded files pass
manual inspection and the verifier exits with code zero.
