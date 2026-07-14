# Router fine-tuning (experimental)

Cortex can offload chat action classification to a compact Qwen2.5-3B-Instruct model fine-tuned with LoRA/QLoRA through Unsloth, exported to GGUF, and served by Ollama.

This is an experimental optimization. The router is always active, but by default it uses the main configured provider and model.

## Purpose

Chat action classification is frequent and structured: answer directly, use RAG, search the web, fetch a URL, or ask a clarifying question. A smaller model can potentially reduce the cost and latency of choosing that branch, provided that it meets the teacher model's held-out accuracy.

The fine-tuned intent classifier is an implementation option for the ReAct-lite routing policy described in [Architecture](architecture.md#chat-routing), not a second routing stage.

## Dependencies

```bash
uv sync --extra finetune
```

The optional group includes `datasets` and `huggingface-hub`. The Unsloth training step requires a compatible GPU environment such as Kaggle or Colab.

## Runtime configuration

Optional router-specific variables override the primary LLM settings:

- `ROUTER_LLM_PROVIDER` — `ollama`, `openai`, `openrouter`, or `lmstudio`; empty uses `LLM_PROVIDER`
- `ROUTER_OLLAMA_MODEL`
- `ROUTER_OPENAI_MODEL`
- `ROUTER_OPENROUTER_MODEL`
- `ROUTER_LMSTUDIO_MODEL`
- `ROUTER_OLLAMA_BASE_URL`
- `ROUTER_TEMPERATURE` — defaults to `0.0`

An empty provider-specific model value falls back to the corresponding primary model.

## Training and activation

`scripts/finetune/pipeline.sh` orchestrates the CPU-side steps around GPU training:

```bash
# Generate teacher-labelled data and push it to Hugging Face Hub
scripts/finetune/pipeline.sh prepare

# Run scripts/finetune/train_unsloth.ipynb on a GPU environment.
# The notebook publishes the LoRA adapter and GGUF artifact.

# Download the GGUF, register it with Ollama, and evaluate it
scripts/finetune/pipeline.sh activate

# Repeat held-out scoring without activation
scripts/finetune/pipeline.sh score
```

`prepare` expands `scripts/finetune/action_seeds.py`, labels cases with a schema-validated teacher model, creates stratified train/held-out splits, and pushes the dataset to Hugging Face Hub.

`score` reports per-class accuracy, latency, and a confusion matrix against teacher labels. Do not activate a candidate based only on aggregate accuracy; inspect class-level regressions, especially tool-using actions.

Defaults such as `DATASET_REPO`, `GGUF_REPO`, `GGUF_FILENAME`, and `OLLAMA_MODEL` are configurable. `TEACHER_API`, `TEACHER_MODEL`, and `TEACHER_BASE_URL` select the labelling backend.

