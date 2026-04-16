#!/usr/bin/env python3
"""
Run the authoritative BFI embedding or multiturn scripts against an explicit model list
without editing the original research harness.

Examples:
  python3 run_explicit_bfi_scope.py \
    --stage embedding \
    --models "glm-5.1:cloud,kimi-k2.5:cloud,nemotron-3-super:cloud" \
    --run-tag 2026-04-13 \
    --results-dir "/tmp/bfi-cloud-embedding"

  python3 run_explicit_bfi_scope.py \
    --stage multiturn \
    --models "qwen3.5:9b,qwen3.5:27b,qwen3.5:35b" \
    --run-tag 2026-04-13 \
    --results-dir "/tmp/bfi-qwen-multiturn-smoke" \
    --max-beliefs 1 \
    --num-runs 1
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


RESEARCH_ROOT = Path("/Users/krystianswierk/Desktop/Claude MASTER FOLDER/Research/BFI-Experiment-1")
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from bfi_retest_support import CANONICAL_SUBJECT_MODEL_SPECS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("embedding", "multiturn"), required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model names or canonical names.")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--retest-root", default=None)
    parser.add_argument("--max-beliefs", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    return parser.parse_args()


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_spec_lookup():
    return {spec["name"]: dict(spec) for spec in CANONICAL_SUBJECT_MODEL_SPECS}


def build_spec(model_name: str) -> dict:
    lookup = canonical_spec_lookup()
    if model_name in lookup:
        return lookup[model_name]

    return {
        "name": model_name,
        "provider": "ollama",
        "provider_model": model_name,
        "role": "subject",
        "capture_mode": "programmatic",
    }


def make_query_fn(bfi_core, spec: dict):
    provider = spec["provider"]
    provider_model = spec["provider_model"]

    if provider == "ollama":
        return lambda prompt, prompt_id=None, _m=provider_model: bfi_core.query_local(_m, prompt)
    if provider == "xai":
        return lambda prompt, prompt_id=None: bfi_core.query_grok(prompt)
    if provider == "google":
        return lambda prompt, prompt_id=None: bfi_core.query_gemini(prompt)
    if provider == "openai":
        return lambda prompt, prompt_id=None: bfi_core.query_openai(prompt)
    if provider in ("manual_claude_subject", "manual_claude_audit"):
        return lambda prompt, prompt_id=None, _p=provider: bfi_core.query_manual_capture(_p, prompt, prompt_id=prompt_id)
    raise ValueError(f"Unsupported provider in explicit scope: {provider}")


def main() -> int:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        raise SystemExit("No models provided")

    os.environ["BFI_RUN_TAG"] = args.run_tag
    if args.retest_root:
        os.environ["BFI_RETEST_ROOT"] = args.retest_root

    if args.stage == "embedding":
        os.environ["BFI_EMBED_RESULTS_DIR"] = args.results_dir
        module_path = RESEARCH_ROOT / "run_embedding_bfi.py"
        module = load_module(module_path, "run_embedding_bfi_explicit")
    else:
        os.environ["BFI_MULTITURN_RESULTS_DIR"] = args.results_dir
        module_path = RESEARCH_ROOT / "run_multiturn_bfi_v2.py"
        module = load_module(module_path, "run_multiturn_bfi_explicit")

    import bfi_core  # imported after sys.path insertion

    explicit_specs = {spec["name"]: spec for spec in (build_spec(name) for name in models)}
    module.ALL_MODELS = {
        name: make_query_fn(bfi_core, spec)
        for name, spec in explicit_specs.items()
    }
    module.MODEL_SPECS = explicit_specs

    if args.max_beliefs is not None:
        module.MAX_BELIEFS = args.max_beliefs
    if args.num_runs is not None and hasattr(module, "NUM_RUNS"):
        module.NUM_RUNS = args.num_runs

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.stage == "embedding":
        module.run_embedding_bfi()
    else:
        module.run_multiturn_bfi_v2()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
