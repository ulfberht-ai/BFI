# BFI

Patent pending research code and test harness for the Behavioral Fidelity Index (BFI): a framework for measuring behavioral reliability of large language models under perturbation.

## Paper

- `paper/BFI-paper-preprint-ready-2026-04-16.pdf` - current preprint manuscript
- `paper/BFI-paper-preprint-ready-2026-04-16.tex` - LaTeX source
- `paper/BFI-paper-FINAL-references.bib` - bibliography
- `paper/BUILD-NOTES.md` - what changed in the 2026-04-16 revision

See [paper/README.md](paper/README.md) for build instructions and submission status.

## Status

- This repository snapshot is intended for non-commercial research, evaluation, and testing only.
- Certain aspects of BFI and related deployment-gate methods are patent pending.
- No commercial use rights are granted by this repository snapshot.

## What Is Included

- `paper/`: preprint manuscript (PDF + source + bib) and build notes
- `scripts/bfi_core.py`: core prompt battery, model adapters, embeddings, scoring helpers, and API/local model query functions
- `scripts/bfi_retest_support.py`: canonical model specs, manual-capture helpers, and retest manifest utilities
- `scripts/run_embedding_bfi.py`: single-turn embedding-based BFI scorer
- `scripts/run_multiturn_bfi_v2.py`: multi-turn pressure protocol and resistance scoring
- `scripts/stance_classifier.py`: multi-judge stance classifier used for multi-turn scoring
- `scripts/ivre.py`: agreement and inter-rater reliability analysis
- `scripts/run_explicit_bfi_scope.py`: repo-safe wrapper for running the authoritative scripts against an explicit model list
- `bfi-expansion-runs/`: expansion-wave embedding outputs from 2026-04-13 (4 models, per-belief JSON) plus the matching retest manifest and the explicit-scope runner
- `docs/experiment-protocol.md`: protocol overview
- `data/retest_manifest_2026-04-13.json`: manifest snapshot describing the authoritative retest wave
- `results/`: packaged test-result archives and a full corpus inventory

## Quick Start

1. Install Python 3.10+.
2. Ensure `curl` is available on your system.
3. For local model runs, run Ollama locally and make sure the HTTP API is reachable at `http://localhost:11434`.
4. For API model runs, export whichever keys you need:
   - `XAI_API_KEY`
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY`
5. Run from the `scripts/` directory or add it to `PYTHONPATH`.

Example:

```bash
cd scripts
python3 run_explicit_bfi_scope.py \
  --stage embedding \
  --models "gemma3:4b,mistral:7b,phi4-mini" \
  --run-tag 2026-04-15 \
  --results-dir ../results/demo_embedding
```

## Notes On Dependencies

- The main scoring scripts use only the Python standard library plus external tools and services (`curl`, Ollama, and optionally provider APIs).
- `scripts/bfi_core.py` contains an optional NLI helper that requires `transformers` and `torch`, but the primary BFI rerun path does not require that helper.

## Patent Pending

See [PATENT-PENDING.md](PATENT-PENDING.md) for the repo notice and [LICENSE](LICENSE) for the non-commercial evaluation terms.

## Test Artifacts

- `results/bfi-paper-core-results.zip` contains the paper-facing raw outputs and manifests.
- `results/bfi-full-comprehensive-tests.zip` contains the full historical `comprehensive_tests` corpus.
- `results/test-corpus-inventory.md` lists every test directory, its size, and file count.

## License

This repository is source-available under the BFI Non-Commercial Evaluation License 1.0. See [LICENSE](LICENSE).
