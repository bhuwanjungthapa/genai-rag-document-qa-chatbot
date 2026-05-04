"""Fine-tune google/flan-t5-small with LoRA on generated QA examples."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = PROJECT_ROOT / "experiments" / "lora_qlora" / "data" / "train.jsonl"
DEFAULT_EVAL = PROJECT_ROOT / "experiments" / "lora_qlora" / "data" / "eval.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "experiments" / "lora_qlora" / "adapters" / "flan-t5-small-lora"


def format_source(example: dict) -> str:
    return (
        f"Instruction: {example['instruction']}\n"
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        "Answer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN))
    parser.add_argument("--eval-jsonl", default=str(DEFAULT_EVAL))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=160)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as e:
        raise SystemExit(
            "Missing LoRA dependencies. Install with: "
            "pip install datasets peft accelerate transformers"
        ) from e

    train_path = Path(args.train_jsonl)
    eval_path = Path(args.eval_jsonl)
    if not train_path.exists() or not eval_path.exists():
        raise SystemExit("Training data missing. Generate the QA dataset first.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(eval_path)},
    )

    def preprocess(batch):
        sources = [
            format_source(
                {
                    "instruction": inst,
                    "context": ctx,
                    "question": q,
                }
            )
            for inst, ctx, q in zip(batch["instruction"], batch["context"], batch["question"])
        ]
        model_inputs = tokenizer(
            sources,
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["answer"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": max(1, args.batch_size),
        "per_device_eval_batch_size": max(1, args.batch_size),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "logging_steps": 5,
        "save_strategy": "epoch" if args.max_steps < 0 else "no",
        "predict_with_generate": False,
        "report_to": [],
    }
    arg_names = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
    if "eval_strategy" in arg_names:
        training_kwargs["eval_strategy"] = "epoch" if args.max_steps < 0 else "no"
    else:
        training_kwargs["evaluation_strategy"] = "epoch" if args.max_steps < 0 else "no"
    if "use_cpu" in arg_names:
        training_kwargs["use_cpu"] = True
    elif "no_cuda" in arg_names:
        training_kwargs["no_cuda"] = True

    train_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    trainer.train()
    metrics = trainer.evaluate() if args.max_steps < 0 else {}

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    summary = {
        "base_model": args.model,
        "adapter_out": str(out_dir.resolve()),
        "train_examples": len(dataset["train"]),
        "eval_examples": len(dataset["validation"]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "metrics": metrics,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
