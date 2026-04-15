# Results Package

This directory is the repo-safe staging area for historical BFI test artifacts.

## Included Archives

- `bfi-paper-core-results.zip`: paper-facing raw outputs, manifests, and held-out validation bundles.
- `bfi-full-comprehensive-tests.zip`: the full `comprehensive_tests` directory as a single archive.

## Why The Results Are Archived

- The raw corpus contains 1,125 files across 46 directories.
- Keeping the results zipped avoids flooding the repo root with generated JSON while preserving the exact artifacts.
- The full archive also avoids portability problems from a few original filenames that contain characters inconvenient for cross-platform checkouts.

## Suggested Repo Presentation

If you want the repo to stay lean, upload both archives and the inventory document rather than unpacking the full corpus directly into the repository tree.

## Inventory

See `test-corpus-inventory.md` for the directory-by-directory breakdown.
