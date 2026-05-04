# LoRA / QLoRA Experiment

This folder adds the graduate-level parameter-efficient fine-tuning path for
the RAG PDF chatbot.

The main app still uses RAG for uploaded PDFs because users can change the
document corpus at any time. LoRA is used here as a research comparison: it
fine-tunes `google/flan-t5-small` on document-grounded QA examples generated
from the current chunks.

## Workflow

1. Build the normal RAG index in the Streamlit app.
2. Open the **LoRA / QLoRA** tab.
3. Generate QA training data from `indexes/chunks.parquet`.
4. Train the LoRA adapter.
5. Evaluate the adapter on the generated eval split.

Equivalent command-line workflow:

```bash
python experiments/lora_qlora/prepare_qa_dataset.py
python experiments/lora_qlora/train_lora.py
python experiments/lora_qlora/evaluate_adapter.py
```

Outputs:

- `data/train.jsonl`
- `data/eval.jsonl`
- `adapters/flan-t5-small-lora/`
- `results/adapter_eval.jsonl`
- `results/adapter_eval_summary.json`

## QLoRA

`train_qlora.py` is included as an optional readiness check because QLoRA
typically requires CUDA and `bitsandbytes`. On most Mac laptops, regular LoRA
with `google/flan-t5-small` is the practical path.

```bash
python experiments/lora_qlora/train_qlora.py --check-only
```

## Honest Report Framing

Use this wording in the report:

> The production chatbot remains RAG-based because user-uploaded PDFs change
> dynamically. LoRA/QLoRA is integrated as a research-level experiment that
> fine-tunes a small local model on document-grounded QA examples, allowing a
> comparison between retrieval-only prompting and parameter-efficient model
> adaptation.
