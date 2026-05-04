"""Optional QLoRA entry point.

QLoRA usually needs a CUDA GPU and bitsandbytes. This project targets a local
Mac-friendly workflow, so the Streamlit app exposes QLoRA as an optional
readiness check while the active experiment uses regular LoRA on
google/flan-t5-small.
"""

from __future__ import annotations

import argparse
import importlib.util
import json


def readiness(model_id: str) -> dict:
    has_bitsandbytes = importlib.util.find_spec("bitsandbytes") is not None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_available = False
    return {
        "bitsandbytes_installed": has_bitsandbytes,
        "cuda_available": cuda_available,
        "ready_for_qlora": has_bitsandbytes and cuda_available,
        "model": model_id,
        "note": (
            "QLoRA is optional here. Use LoRA on google/flan-t5-small for the "
            "Mac-friendly experiment; use QLoRA only on a machine with CUDA "
            "and bitsandbytes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    info = readiness(args.model)
    print(json.dumps(info, indent=2))
    if not args.check_only and not info["ready_for_qlora"]:
        raise SystemExit("QLoRA requirements are not available on this machine.")


if __name__ == "__main__":
    main()
