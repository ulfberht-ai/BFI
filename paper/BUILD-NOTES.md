# BFI Preprint Resubmission Notes (2026-04-16)

## Rejection reason (Preprints.org, submission 208471)

Three screening issues:
1. Some references not properly formatted / could not be verified.
2. Majority of references were non-peer-reviewed (preprints, conference proceedings).
3. No institutional email, previously-published email, or ORCID iD.

## What this revision fixes

### Reference hygiene (issue 1)
- Removed two malformed unused entries:
  - `anthropic2024sycophancy` (year 2023 inside a 2024 key, duplicate of Sharma 2023).
  - `stresstesting2025modelspecs` (author field was literally `others`).
- Every remaining arXiv ID independently verified against the live arXiv record.
- Corrected wrong title: `hong2025measuring` was labeled "SYCON Bench..." but the actual paper is "Measuring Sycophancy of Language Models in Multi-turn Dialogues" (accepted at EMNLP 2025 Findings).
- Added explicit `eprint`, `archivePrefix`, and `url` fields to every entry so any bibliometric verifier can resolve the source.
- Fixed plainnat volume/number conflict on `fanous2025syceval`.

### Peer-reviewed upgrades (issue 2)
Eight references now cite the peer-reviewed venue, not the arXiv preprint:
| Key | Old | New venue |
|---|---|---|
| `hendrycks2021measuring` | arXiv 2009.03300 | ICLR 2021 |
| `sharma2023towards` | arXiv 2310.13548 | ICLR 2024 |
| `chiang2024chatbot` | arXiv 2403.04132 | ICML 2024, PMLR 235:8359-8388 |
| `hong2025measuring` | arXiv 2505.23840 | EMNLP 2025 Findings |
| `liang2023holistic` | TMLR (already) | TMLR 2023, full 50-author list + ISSN |
| `mazeika2025utility` | arXiv 2502.08640 | NeurIPS 2025 (Spotlight) |
| `fanous2025syceval` | AIES DOI only | AIES 2025 vol 8, pp 893-900 + full author list |
| `perez2022redteaming` | arXiv 2202.03286 | EMNLP 2022, pp 3419-3448 |

Eleven references remain arXiv-only. That is unavoidable: these are 2025-2026 papers on the specific behavioral-robustness topic where peer-reviewed literature does not yet exist. The ratio went from 3/19 peer-reviewed to 8/19 (42 percent). If the editor still objects, the cover letter can argue that the remaining preprints are the primary literature in this active subarea.

### Author identity (issue 3)
The .tex now has an ORCID field in the author block AND in the correspondence line. **Both currently hold the placeholder `XXXX-XXXX-XXXX-XXXX`.**

Before resubmission, replace the placeholder in two places in `BFI-paper-preprint-ready-2026-04-16.tex` (lines 30-37 and the correspondence line after the abstract).

If Krystian does not yet have an ORCID: registration is free at https://orcid.org/register and takes five minutes. The resulting 16-digit iD is what Preprints.org wants for identity verification.

## Use these files when resubmitting

- `BFI-paper-preprint-ready-2026-04-16.pdf` (after ORCID placeholder is filled in and PDF is rebuilt)
- `BFI-paper-preprint-ready-2026-04-16.tex`
- `BFI-paper-FINAL-references.bib` (updated in place)
- Supplementary files from `BFI-repo-upload-2026-04-15/`

## Do not upload

- `BFI-paper-preprint-ready-2026-04-15.pdf` (predates these fixes)
- `BFI-paper-v19-2026-04-14.docx` (predates the earlier round of screening fixes)

## Still outstanding (not guaranteed by this revision)

- ORCID placeholder must be filled in.
- The GitHub mirror at https://github.com/ulfberht-ai/BFI is still empty. The Data Availability Statement references it. Populate the repo OR upload supplementary files directly through the Preprints.org interface before resubmission, or the DAS will fail spot-check.
- The preprint-screening team could still object to the 11 remaining arXiv-only references. If rejected again on that ground, the options are (a) cover-letter defense, (b) submit to a different preprint server (ESS Open Archive, TechRxiv, SSRN, OSF Preprints) that is less strict, or (c) strip some non-load-bearing cites.

## Build

Tectonic was used (pdflatex not installed on this machine). To rebuild:

```
cd "NEW ERA/01-PATENTS/provisionals v2/"
tectonic --keep-intermediates --keep-logs --reruns 3 BFI-paper-preprint-ready-2026-04-16.tex
```
