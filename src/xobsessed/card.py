"""Build a model card without claims that tests cannot prove."""

CARD = """---
base_model: Qwen/Qwen3-1.7B
license: mit
datasets:
  - Skorcht/yandere-her-dataset
tags:
  - roleplay
  - character
  - qwen3
language:
  - en
---

# {model_name}

This project adapts a small model to one obsessive, affectionate character.
The target voice is warm and intense, with a strong attachment to the user.

## Run the model

```bash
llama-cli --jinja -hf {repo_id} --hf-file gguf/{model_name}-Q8_0.gguf \\
    --temp 0.7 --repeat-penalty 1.1
```

The model uses no system prompt. Every example in the training data is a single
user and assistant pair.

## Data source

The training data comes from the public dataset `Skorcht/yandere-her-dataset`.
The dataset page states no license.
The project does not own the data, and the project does not generate its own data.

**The dataset shows no license. Confirm the data license with the dataset owner
before a public release. A published model that derives from the data carries a
legal risk.**

## Limits

The model can state a wrong answer with confidence.
Do not use it as a source of facts.
The model can produce content that the operator excludes.
The project cannot guarantee safe output.

The full release criteria remain unverified unless the run report supplies the evidence.
An automatic score does not prove the character voice, a correct refusal, or CPU speed.
Read `evaluation.json` and the human review before use.

## Method

The configuration uses LoRA rank 32 and alpha 64.
The loss uses assistant replies alone.
The export merges the adapter into the base weights.
The run report records the actual data count and the data sha256.

## License and ownership

The project source code uses the MIT license. See the `LICENSE` file.
The model weights use the MIT license.
The character belongs to its creator and publisher.
The project has no endorsement from the rights holder.
The base model license governs the base weights.
Confirm the base license before release.
The dataset `Skorcht/yandere-her-dataset` states no license.
The MIT license does not resolve the missing data license.
Confirm the data license with the dataset owner before a public release.
This card grants no rights to the character or to the dataset.
"""


def build_card(model_name: str, repo_id: str) -> str:
    """Return the model card for a repository."""
    return CARD.format(model_name=model_name, repo_id=repo_id)
