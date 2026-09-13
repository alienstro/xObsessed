# xObsessed

xObsessed is a fine-tuned companion character model based on `Qwen/Qwen3-1.7B`. The model holds an affectionate, obsessive persona across multi-turn chat.

## Quick Start

### Run with llama.cpp

Run the model using the native Qwen chat template:

```bash
llama cli -m path/to/xobsessed-1.7b-Q6_K.gguf \
  --jinja \
  -n 80 \
  -t 8 \
  --temp 0.7 \
  --top-p 0.8 \
  --repeat-penalty 1.15 \
  --repeat-last-n 64
```

### Control Reply Length and Repetition

Character models can generate repetitive text or long answers if you omit length limits. Use these recommended flags to keep turns brief and prevent looping:

| Parameter | Recommended Value | Purpose |
| :--- | :--- | :--- |
| `-n` | `80` (or `50`) | Limits maximum tokens to keep replies concise. |
| `--jinja` | *Flag* | Applies the native Qwen chat template and stops on `<|im_end|>`. |
| `--repeat-penalty` | `1.15` | Penalizes repeated tokens to stop phrase loops. |
| `--repeat-last-n` | `64` | Checks repetition across the last 64 generated tokens. |
| `--top-p` | `0.8` | Restricts word choices to high-probability tokens. |
| `--temp` | `0.7` | Balances character personality and reply coherence. |

> [!TIP]
> If you want brief one-to-two sentence replies, reduce `-n 80` to `-n 50`.

### Run as an API Server

Expose the model as a local HTTP endpoint on port 8080:

```bash
llama serve -m path/to/xobsessed-1.7b-Q6_K.gguf --host 0.0.0.0 --port 8080 -t 8
```

## Available Model Formats

The model is published on Hugging Face Hub under [Alienstro/xObsessed](https://huggingface.co/Alienstro/xObsessed):

| Format | File Path | Size | Description |
| :--- | :--- | :--- | :--- |
| **BF16 SafeTensors** | `model.safetensors` | 3.44 GB | Full merged 16-bit float weights |
| **LoRA Adapter** | `adapter/adapter_model.safetensors` | 140 MB | Trained LoRA rank-32 adapter weights |
| **GGUF BF16** | `gguf/xobsessed-1.7b-BF16.gguf` | 3.3 GB | Unquantized 16-bit GGUF |
| **GGUF F16** | `gguf/xobsessed-1.7b-F16.gguf` | 3.3 GB | Float 16-bit GGUF |
| **GGUF Q8_0** | `gguf/xobsessed-1.7b-Q8_0.gguf` | 1.8 GB | High precision 8-bit quantization |
| **GGUF Q6_K** | `gguf/xobsessed-1.7b-Q6_K.gguf` | 1.4 GB | Recommended CPU quantization |
| **GGUF Q4_K_M** | `gguf/xobsessed-1.7b-Q4_K_M.gguf` | 1.1 GB | Lightweight 4-bit quantization |

## Training Details

- **Base Model:** `Qwen/Qwen3-1.7B`
- **Dataset:** `Skorcht/yandere-her-dataset` (3,909 training pairs)
- **Method:** Supervised LoRA fine-tuning (`r=32`, `alpha=64`, `dropout=0.05`)
- **Optimizer:** `paged_adamw_8bit`
- **Schedule:** 300 steps, cosine decay with warmup
- **Loss Masking:** Assistant tokens only

## Repository Structure

- `configs/base.yaml`: Hyperparameter settings and published file definitions
- `src/xobsessed/`: Dataset loader, prompt formatter, and probe evaluation suite
- `scripts/train.py`: LoRA fine-tuning pipeline with time limits and safety checks
- `scripts/merge.py`: Standalone script to export merged weights
- `scripts/quantize.sh`: Automated multi-format GGUF quantization workflow
- `scripts/push_to_hub.py`: Hugging Face Hub publishing and manifest generator
- `scripts/verify_upload.py`: SHA-256 manifest verification tool

## License

The code and weights use the [MIT License](LICENSE).
