# Model evaluation

Cortex separates deterministic AI contracts, which run in CI, from model-backed semantic evaluation and prompt optimization, which run manually with provider credentials.

## Deterministic regression gate

The credential-free gate protects routing, citation provenance, and finance-tool planning on every pull request:

```bash
uv run python -m src.evals.regression_gate
```

See [Testing and quality](testing-and-quality.md#ai-regression-gate) for its CI policy.

## Model comparison with DeepEval

Install evaluation dependencies:

```bash
uv sync --extra evals
```

`src/evals/model_comparison.py`:

- Loads cases from `src/evals/golden_set.json`
- Runs `summarize_node` for each configured provider/model pair
- Scores faithfulness and answer relevancy with DeepEval
- Writes results to `src/evals/results.csv`

Edit `MODEL_CONFIGS` in the script, then run:

```bash
uv run python src/evals/model_comparison.py
```

## Prompt optimization with DSPy

Cortex uses DSPy MIPROv2 to search for prompt instructions and examples against the golden set. The existing Jinja2 prompt pipeline remains the production default; optimization is an opt-in toolchain.

Relevant implementation:

- `SummarizeSignature` and `ReportSignature` define typed inputs and outputs.
- `SummarizeModule` and `ReportModule` wrap structured chain-of-thought generation.
- `DspyPromptOptimizer` builds training data, runs optimization, compares scores, and persists the optimized program.

Run optimization:

```python
from src.evals.model_comparison import load_golden_set
from src.prompts.dspy_optimizer import DspyPromptOptimizer, SummarizeModule

optimizer = DspyPromptOptimizer()
result = optimizer.optimize(SummarizeModule(), load_golden_set(), "summarize")
print(result.before_score, result.after_score, result.improvement)
optimizer.save(result, "optimized_summarize")
```

Compare an optimized program with the original:

```python
from src.evals.model_comparison import load_golden_set
from src.prompts.dspy_optimizer import DspyPromptOptimizer, SummarizeModule

optimizer = DspyPromptOptimizer()
results = optimizer.compare(
    SummarizeModule(),
    "optimized_prompts/optimized_summarize.json",
    load_golden_set(),
    "summarize",
)
for result in results:
    print(result["query"], result["original_score"], result["optimized_score"])
```

Load the persisted program:

```python
from src.prompts.dspy_optimizer import DspyPromptOptimizer, SummarizeModule

optimizer = DspyPromptOptimizer()
module = optimizer.load(
    SummarizeModule(),
    "optimized_prompts/optimized_summarize.json",
)
prediction = module(query="your question", source_blocks="...", domain="")
print(prediction.summaries)
```

Tests:

```bash
uv run pytest tests/test_dspy_optimizer.py -v
```

Expand `src/evals/golden_set.json` with representative cases before treating an optimization score as evidence of general improvement.

