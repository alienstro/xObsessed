"""Build a model card without claims that tests cannot prove."""

CARD = """---
base_model: Qwen/Qwen3-1.7B-Instruct
tags:
  - roleplay
  - character
  - qwen3
language:
  - en
---

# {model_name}

This project adapts a small model to the fictional character Frieren.
The target voice uses short sentences, a flat tone, and dry humour.

## Run the model

Download `system_prompt.txt` from this model repository before the command.

```bash
llama-cli --jinja -hf {repo_id} --hf-file gguf/{model_name}-Q8_0.gguf \\
    --system-prompt-file system_prompt.txt --temp 0.7 --repeat-penalty 1.1
```

The system prompt remains part of the character setup.

## Limits

The model invents events from the series because the project does not teach story facts.
The model can state a wrong answer with confidence.
Do not use it as a source of facts.
The project targets wholesome replies, but it cannot guarantee safe output.

The full release criteria remain unverified unless the run report supplies the evidence.
An automatic score does not prove the character voice, a correct refusal, or CPU speed.
Read `evaluation.json` and the human review before use.

## Method

The configuration uses LoRA rank 32 and alpha 64.
The loss uses assistant replies alone.
The export merges the adapter into the base weights.
The project targets about 1,200 original conversations.
The run report records the actual data count.

Every project conversation must use original text.
The project does not use copied anime, manga, or subtitle text.

## License and ownership

This project is a non-commercial fan work.
The character belongs to its creator and publisher.
The project has no endorsement from the rights holder.
The base model license governs the base weights.
Confirm the base license before release.
This card grants no rights to the character.
"""


def build_card(model_name: str, repo_id: str) -> str:
    """Return the model card for a repository."""
    return CARD.format(model_name=model_name, repo_id=repo_id)
