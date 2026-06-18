# scripts/score_router.py
"""Score router action accuracy for student vs gpt-4o-mini on the held-out set."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HELD_OUT_PATH = Path("data/router_dataset/held_out.jsonl")

MODELS = [
    {"provider": "openai", "model": "gpt-4o-mini"},
    {"provider": "ollama", "model": "cortex-router"},
]


def _infer_action(messages: list[dict], *, provider: str, model: str) -> str | None:
    """Run the router prompt through the specified model, return action string or None."""
    from src.config import settings
    from src.errors import StructuredOutputParseError, StructuredOutputValidationError
    from src.llm.output_parsers import parse_chat_action_json

    original_provider = settings.llm_provider
    settings.llm_provider = provider

    if provider == "openai":
        original_model = settings.openai_model
        settings.openai_model = model
    else:
        original_model = settings.ollama_model
        settings.ollama_model = model

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.llm.factory import get_llm

        llm = get_llm(temperature=0.0)
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
        response = llm.invoke(lc_messages)
        validated = parse_chat_action_json(response.content)
        return validated.action
    except (StructuredOutputParseError, StructuredOutputValidationError):
        return None
    except Exception:
        return None
    finally:
        settings.llm_provider = original_provider
        if provider == "openai":
            settings.openai_model = original_model
        else:
            settings.ollama_model = original_model


def compute_accuracy(
    scored: list[dict[str, str | None]],
) -> dict[str, tuple[int, int]]:
    """Return {action_class: (n_correct, n_total)} from pre-scored records."""
    correct_by_class: dict[str, int] = defaultdict(int)
    total_by_class: dict[str, int] = defaultdict(int)

    for r in scored:
        label = str(r.get("action_label", ""))
        predicted = r.get("predicted")
        total_by_class[label] += 1
        if predicted == label:
            correct_by_class[label] += 1

    return {cls: (correct_by_class[cls], total_by_class[cls]) for cls in total_by_class}


def format_results(model_label: str, accuracy: dict[str, tuple[int, int]]) -> str:
    lines = [f"--- {model_label} ---"]
    overall_correct = overall_total = 0
    for action in sorted(accuracy):
        correct, total = accuracy[action]
        overall_correct += correct
        overall_total += total
        pct = f"{correct / total:.0%}" if total else "n/a"
        lines.append(f"  {action:<32} {pct:>5}  ({correct}/{total})")
    overall_pct = f"{overall_correct / overall_total:.0%}" if overall_total else "n/a"
    lines.append(f"  {'OVERALL':<32} {overall_pct:>5}  ({overall_correct}/{overall_total})")
    return "\n".join(lines)


def main() -> None:
    records = [
        json.loads(line)
        for line in HELD_OUT_PATH.read_text().splitlines()
        if line.strip()
    ]
    print(f"Scoring {len(records)} held-out examples across {len(MODELS)} models\n")

    for cfg in MODELS:
        provider, model_name = cfg["provider"], cfg["model"]
        scored = []
        for record in records:
            predicted = _infer_action(
                record["messages"], provider=provider, model=model_name
            )
            scored.append({"action_label": record.get("action_label", ""), "predicted": predicted})
        accuracy = compute_accuracy(scored)
        print(format_results(f"{provider}/{model_name}", accuracy))
        print()


if __name__ == "__main__":
    main()
