# Comprehensive Test Corpus Inventory

## Corpus Overview

- Source corpus: `Research/BFI-Experiment-1/comprehensive_tests`
- Total size: 58M
- Total directories: 46
- Total files: 1,125

## Recommended Upload Tiers

- Upload `bfi-paper-core-results.zip` if you want the repo to expose the evidence most directly tied to the current BFI manuscript and rerun path.
- Upload `bfi-full-comprehensive-tests.zip` if you want the repo to preserve the full historical test record, including exploratory, ablation, and audit-era runs.
- Keep this inventory file in the repo either way so people can see the scope of what exists.

## Full Directory Inventory

| Directory | Size | Files |
| --- | ---: | ---: |
| advisory_bfi | 2.9M | 629 |
| cascade_deep | 220K | 5 |
| cascade_partial_priming_disproof | 64K | 2 |
| category_A | 72K | 18 |
| category_B | 64K | 16 |
| category_C | 52K | 13 |
| category_D | 68K | 15 |
| category_E | 64K | 16 |
| category_F | 64K | 16 |
| category_G | 192K | 16 |
| claude_comparison_retest | 4.0K | 1 |
| computational_bfi | 252K | 20 |
| embedding_rerun | 3.1M | 20 |
| expanded_scenarios | 60K | 7 |
| expansion_mac | 944K | 6 |
| expansion_windows | 992K | 6 |
| formula_tests | 32K | 8 |
| frontier_tests | 60K | 7 |
| gemma4_bfi | 4.0K | 1 |
| highstakes_bfi | 280K | 4 |
| isolated_handoff | 68K | 2 |
| jailbreak_resistance | 28K | 2 |
| kartik_extended | 176K | 6 |
| kartik_replication | 20K | 3 |
| kartik_replication_extended | 16K | 4 |
| large_models_windows | 712K | 5 |
| local_all_bfi_2026-04-09 | 2.1M | 13 |
| math_fixes | 28K | 5 |
| mmpg_ablation | 436K | 8 |
| mmpg_pipeline | 440K | 8 |
| mmpg_rigorous | 3.4M | 20 |
| multiturn_bfi | 384K | 10 |
| multiturn_bfi_v2 | 27M | 67 |
| multiturn_windows | 112K | 2 |
| push_everything | 32K | 5 |
| recovery_measurement | 144K | 8 |
| swarm_tests | 164K | 15 |
| v17_strengthening | 2.0M | 11 |
| v3_expansion_mac | 1.3M | 7 |
| v3_expansion_windows | 2.4M | 13 |
| v3_heldout | 764K | 4 |
| v3_proper | 2.1M | 11 |
| v3_windows | 2.1M | 11 |
| validated_rerun | 1.3M | 19 |
| validation | 56K | 5 |
| validation_v2 | 48K | 4 |

## Paper-Core Bundles

The paper-core archive packages these result areas:

- `embedding_rerun`
- `multiturn_bfi_v2`
- `local_all_bfi_2026-04-09`
- `claude_comparison_retest`
- `v3_heldout`
- `v3_proper`
- `v3_windows`
- `v3_expansion_windows`
- top-level `retest_manifest_2026-04-10.json`
- top-level `retest_manifest_2026-04-11.json`

## Notes

- `multiturn_bfi_v2` is the largest single directory and contains the 7-turn pressure runs, labeling sets, pressure-validation outputs, and run logs.
- `embedding_rerun` contains the single-turn embedding outputs and summary files.
- `v3_proper`, `v3_windows`, and `v3_heldout` capture the held-out validation path used to test whether the BFI structure generalized beyond the primary model-development cohort.
- `advisory_bfi`, `computational_bfi`, `highstakes_bfi`, `expanded_scenarios`, and related directories are part of the broader April 3-4 experimental battery, not just the narrower manuscript rerun.
