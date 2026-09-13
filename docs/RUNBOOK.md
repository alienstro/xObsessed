# Runbook

**Date:** 2026-09-13

## Requirements

- Use a CUDA RunPod pod with enough disk for the base model and output.
- Set the keys in `.env` from `.env.example`.
- Review `docs/DATA_REVIEW.md` before training.
- Keep one active remote command.

## Prepare Data

The project uses the public dataset `Skorcht/yandere-her-dataset`.
The prepare script downloads the dataset and writes `data/raw.jsonl`.

**Caution:** Pin the dataset revision in `configs/base.yaml` to a commit hash.
The default value `main` moves when the owner changes the dataset.

```bash
uv run python scripts/prepare_data.py
```

The script prints the row count and the sha256 of the output file.

Read twenty conversations and complete `docs/DATA_REVIEW.md`.

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
scripts/pod.sh run git clone https://github.com/ggerganov/llama.cpp.git /workspace/xObsessed/llama.cpp
scripts/pod.sh run scripts/push_to_hub.py --reviewed
scripts/pod.sh run bash scripts/quantize.sh out/merged out/gguf xobsessed-1.7b
scripts/pod.sh run scripts/push_to_hub.py --reviewed --gguf out/gguf
scripts/pod.sh verify
```

## License

The project code and the model weights use the MIT license. See `LICENSE`.
The dataset `Skorcht/yandere-her-dataset` states no license.
The MIT license does not resolve the missing data license.

**Caution:** Confirm the data license with the dataset owner before a public release.
The project cannot grant rights that it does not hold.

Download the Q8 GGUF and ask three probes with `--jinja`.

Do not run `scripts/pod.sh terminate --yes` until the downloaded files pass
manual inspection and the verifier exits with code zero.
