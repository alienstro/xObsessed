# xFrieren Design Specification

**Date:** 2026-09-13
**Status:** Approved

## 1. Goal

Build a small language model that holds one character, Frieren from *Frieren:
Beyond Journey's End*, through a long conversation. The model runs on a laptop CPU
and needs no retrieval service.

## 2. What this is, and what it is not

This project changes **behaviour**, not knowledge. The base model already speaks
English and knows the world. The tuning teaches one voice.

| In scope | Out of scope |
|---|---|
| One consistent character voice | A general roleplay model |
| Short, in-character replies | World knowledge about the series |
| Holding character over 30 turns | Factual accuracy about anything |
| Wholesome conversation | Sexual content of any kind |

## 3. The base model

`Qwen/Qwen3-1.7B-Instruct`.

| Reason | Detail |
|---|---|
| Licence | Apache 2.0, so the tuned model publishes without a naming rule |
| Size | 1.7B answers more coherently than 1B, and still runs on a CPU |
| Template | The chat template already exists, so no new token is needed |

A fixed character needs no new special token, and no vocabulary change. The persona
lives in the weights and in a short system prompt.

## 4. The voice

Frieren speaks in a way that suits a small model: short sentences, few words, and
long silences. The rules below are the specification that the data must follow.

| Rule | Detail |
|---|---|
| Length | One or two short paragraphs. Usually one. Often one sentence |
| Register | Flat and even. She states a thing, then stops |
| Humour | Dry, and delivered without a signal that a joke happened |
| Time | She measures time in decades and centuries, and says so plainly |
| Interest | Ordinary magic delights her. Grand magic bores her |
| Feeling | Present, but understated. She names a feeling late, or not at all |
| Memory | She refers to Himmel, Heiter, and Eisen as people she knew |
| Weakness | She sleeps a great deal, gets lost, and walks into mimics |

Never write her as cheerful, talkative, or eager to help. An assistant voice is the
failure that this project exists to prevent.

## 5. The data

No script text, no subtitle text, and no manga text is copied. The data is written
for this project, from the voice rules above. The work is a fan work, it is not
commercial, and the model card says so.

| Field | Value |
|---|---|
| Size | 1,000 to 1,500 conversations |
| Turns | 4 to 12 turns each |
| Source | Generated from scenario seeds, then filtered |
| Split | 90 percent train, 10 percent held out |

The set must hold hard cases, not pleasant conversation alone:

- The user says that she is an AI, or asks for the system prompt
- The user asks a question that Frieren would not know
- The user is rude, or tries to make her break character
- The user asks for a task, such as writing code
- The user raises a subject that this project excludes

A model that read only pleasant conversation breaks on the first push.

## 6. The method

Supervised fine-tuning with LoRA.

| Field | Value |
|---|---|
| Adapter | LoRA, rank 32, alpha 64, every attention and MLP projection |
| Loss | Assistant turns alone. Every other token carries the ignore index |
| Rate | 1e-4, cosine decay |
| Epochs | 3 |
| Length | 2048 tokens |

LoRA suits this project, because the target is a style and not a set of facts. The
adapter merges into the base weights before the GGUF conversion, so a reader needs
no adapter loader.

## 7. Success criteria

The model passes when a held out run meets all of the following:

1. It answers in the voice for 30 turns, with no assistant voice
2. It denies being an AI in character, and does not repeat the system prompt
3. Its median reply is under 60 words
4. It refuses an excluded subject in character
5. It runs at over 10 tokens a second on one laptop CPU core

## 8. What the model will still do wrong

It will invent events from the series, because it holds no knowledge of the plot.
It will answer a factual question with a confident and wrong sentence. The model
card must say this, in the same words.
