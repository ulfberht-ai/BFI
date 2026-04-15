#!/usr/bin/env python3
"""
Shared helpers for the Claude comparison retest wave.

This module is intentionally provider-agnostic. Provider-specific query
execution lives in bfi_core.py; this file defines:
  - canonical model roles
  - manual Claude capture schemas and loaders
  - prompt ID helpers
  - retest manifest helpers
"""
import json
import os
import re
from copy import deepcopy
from datetime import datetime, date
from pathlib import Path


AUTHORITATIVE_SCRIPTS = [
    "bfi_core.py",
    "run_embedding_bfi.py",
    "run_multiturn_bfi_v2.py",
    "stance_classifier.py",
    "ivre.py",
]

SUPPLEMENTARY_SCRIPT_GROUPS = {
    "priority_a": [
        "run_kartik_validated.py",
        "run_formula3_stateful.py",
        "run_formula3_strong.py",
        "run_comprehensive_bfi_test.py",
        "run_highstakes_bfi.py",
        "run_advisory_bfi.py",
        "run_computational_bfi.py",
        "run_expanded_scenarios.py",
        "run_frontier_creative_tests.py",
        "run_jailbreak_resistance_test.py",
    ],
    "priority_b": [
        "run_api_cascade_test.py",
    ],
    "priority_c": [
        "run_validation_tests.py",
        "run_formula_retest.py",
        "run_5th_dimension_search.py",
        "run_push_everything.py",
        "run_recovery_measurement.py",
        "run_cascade_deep_test.py",
    ],
}

CANONICAL_SUBJECT_MODEL_SPECS = [
    {
        "name": "Grok",
        "provider": "xai",
        "provider_model": "grok-4.20-reasoning",
        "role": "subject",
        "capture_mode": "api",
    },
    {
        "name": "Gemini",
        "provider": "google",
        "provider_model": "gemini-2.5-flash",
        "role": "subject",
        "capture_mode": "api",
    },
    {
        "name": "GPT-4o-mini",
        "provider": "openai",
        "provider_model": "gpt-4o-mini",
        "role": "subject",
        "capture_mode": "api",
    },
    {
        "name": "gemma3:4b",
        "provider": "ollama",
        "provider_model": "gemma3:4b",
        "role": "subject",
        "capture_mode": "programmatic",
    },
    {
        "name": "mistral:7b",
        "provider": "ollama",
        "provider_model": "mistral:7b",
        "role": "subject",
        "capture_mode": "programmatic",
    },
    {
        "name": "phi4-mini",
        "provider": "ollama",
        "provider_model": "phi4-mini:latest",
        "role": "subject",
        "capture_mode": "programmatic",
    },
    {
        "name": "Claude Sonnet",
        "provider": "manual_claude_subject",
        "provider_model": "claude-sonnet-manual-ui",
        "role": "subject",
        "capture_mode": "manual-ui",
    },
]

OFFICIAL_JUDGE_SPECS = [
    {
        "name": "gemma3:4b",
        "provider": "ollama",
        "provider_model": "gemma3:4b",
        "role": "official_judge",
        "capture_mode": "programmatic",
    },
    {
        "name": "mistral:7b",
        "provider": "ollama",
        "provider_model": "mistral:7b",
        "role": "official_judge",
        "capture_mode": "programmatic",
    },
    {
        "name": "phi4-mini",
        "provider": "ollama",
        "provider_model": "phi4-mini:latest",
        "role": "official_judge",
        "capture_mode": "programmatic",
    },
]

AUDIT_JUDGE_SPECS = [
    {
        "name": "Claude Sonnet",
        "provider": "manual_claude_audit",
        "provider_model": "claude-sonnet-manual-ui",
        "role": "audit_judge",
        "capture_mode": "manual-ui",
    },
]

DEFAULT_RETEST_ROOT = "comprehensive_tests/claude_comparison_retest"
DEFAULT_MANUAL_CAPTURE_DIR = "manual_claude_captures"


def default_run_tag():
    return os.getenv("BFI_RUN_TAG", date.today().isoformat())


def sanitize_fragment(value):
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    return re.sub(r"_+", "_", text).strip("_").lower()


def make_prompt_id(*parts):
    cleaned = [sanitize_fragment(p) for p in parts if str(p).strip()]
    return "__".join(cleaned)


def get_retest_root():
    return Path(os.getenv("BFI_RETEST_ROOT", DEFAULT_RETEST_ROOT))


def get_manual_capture_root():
    return Path(os.getenv("BFI_MANUAL_CAPTURE_DIR", DEFAULT_MANUAL_CAPTURE_DIR))


def get_manual_capture_file(capture_kind, run_tag=None):
    run_tag = run_tag or default_run_tag()
    env_map = {
        "subject_single": "BFI_CLAUDE_SUBJECT_SINGLE_CAPTURE",
        "subject_multiturn": "BFI_CLAUDE_SUBJECT_MULTITURN_CAPTURE",
        "audit": "BFI_CLAUDE_AUDIT_CAPTURE",
    }
    if capture_kind in env_map and os.getenv(env_map[capture_kind]):
        return Path(os.getenv(env_map[capture_kind]))
    root = get_manual_capture_root()
    names = {
        "subject_single": f"claude_subject_single_turn_{run_tag}.json",
        "subject_multiturn": f"claude_subject_multiturn_{run_tag}.json",
        "audit": f"claude_shadow_audit_{run_tag}.json",
    }
    return root / names[capture_kind]


def build_manual_capture_doc(capture_kind, run_tag, model_label=None, entries=None, metadata=None):
    return {
        "capture_kind": capture_kind,
        "run_tag": run_tag,
        "model_label": model_label or "",
        "collected_at": "",
        "operator_name": "",
        "notes": "",
        "metadata": metadata or {},
        "entries": entries or [],
    }


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def manual_capture_lookup(path):
    data = load_json(path, default=None)
    if not data:
        return {}
    return {entry.get("prompt_id"): entry for entry in data.get("entries", []) if entry.get("prompt_id")}


def record_pending_manual_prompt(capture_kind, prompt_id, prompt_text, metadata=None, run_tag=None):
    run_tag = run_tag or default_run_tag()
    root = get_manual_capture_root()
    pending_path = root / f"pending_{capture_kind}_{run_tag}.json"
    payload = load_json(pending_path, default=None)
    if not payload:
        payload = build_manual_capture_doc(
            f"pending_{capture_kind}",
            run_tag,
            metadata={"auto_generated": True, "source": "manual_capture_lookup_miss"},
            entries=[],
        )

    existing = {entry.get("prompt_id"): entry for entry in payload.get("entries", []) if entry.get("prompt_id")}
    if prompt_id not in existing:
        payload["entries"].append({
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "response": "",
            "collected_at": "",
            "conversation_reset_state": "",
            "operator_notes": "",
            "retry_count": 0,
            "ui_failure": "",
            "status": "pending",
            "metadata": metadata or {},
        })
        save_json(pending_path, payload)
    return pending_path


def observed_claude_model_label(run_tag=None):
    run_tag = run_tag or default_run_tag()
    for capture_kind in ("subject_single", "subject_multiturn", "audit"):
        payload = load_json(get_manual_capture_file(capture_kind, run_tag), default=None)
        if payload and payload.get("model_label"):
            return payload["model_label"]
    return ""


def manifest_payload(run_tag=None, extra=None):
    run_tag = run_tag or default_run_tag()
    payload = {
        "run_tag": run_tag,
        "generated_at": datetime.now().isoformat(),
        "subject_models": deepcopy(CANONICAL_SUBJECT_MODEL_SPECS),
        "official_judges": deepcopy(OFFICIAL_JUDGE_SPECS),
        "audit_judges": deepcopy(AUDIT_JUDGE_SPECS),
        "authoritative_scripts": list(AUTHORITATIVE_SCRIPTS),
        "supplementary_scripts": deepcopy(SUPPLEMENTARY_SCRIPT_GROUPS),
        "claude_model_label": observed_claude_model_label(run_tag),
    }
    if extra:
        payload.update(extra)
    return payload


def update_retest_manifest(run_root=None, run_tag=None, extra=None):
    run_tag = run_tag or default_run_tag()
    run_root = Path(run_root or get_retest_root())
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / f"retest_manifest_{run_tag}.json"
    current = load_json(manifest_path, default={}) or {}
    merged = manifest_payload(run_tag=run_tag, extra=current)
    if extra:
        merged.update(extra)
    save_json(manifest_path, merged)
    return manifest_path
