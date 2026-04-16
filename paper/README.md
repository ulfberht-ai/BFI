# BFI Paper

Preprint manuscript for the Behavioral Fidelity Index (BFI) framework.

## Files

- `BFI-paper-preprint-ready-2026-04-16.pdf` - compiled manuscript (189 KB)
- `BFI-paper-preprint-ready-2026-04-16.tex` - LaTeX source
- `BFI-paper-FINAL-references.bib` - bibliography (19 entries, mix of peer-reviewed venues and arXiv preprints)
- `BUILD-NOTES.md` - notes on the 2026-04-16 revision (what was changed, what still needs to happen before submission to a preprint server)

## Citing

```
Swierk, K. A. (2026). BFI: Measuring What Capability Benchmarks Miss - Behavioral Reliability of LLMs Under Perturbation. Preprint.
```

## Rebuilding the PDF

This manuscript was built with Tectonic 0.15. Any standard LaTeX engine that supports `natbib` will work.

```
tectonic --keep-intermediates --reruns 3 BFI-paper-preprint-ready-2026-04-16.tex
```

Or with a TeX Live install:

```
pdflatex BFI-paper-preprint-ready-2026-04-16.tex
bibtex   BFI-paper-preprint-ready-2026-04-16
pdflatex BFI-paper-preprint-ready-2026-04-16.tex
pdflatex BFI-paper-preprint-ready-2026-04-16.tex
```

## Submission Status

The 2026-04-15 version was screened out at Preprints.org. See `BUILD-NOTES.md` for the specific screening feedback and what was changed in this revision.

## Note on Author Identity Field

The ORCID iD in the author block currently holds a placeholder (`XXXX-XXXX-XXXX-XXXX`). It will be replaced with the corresponding author's real ORCID before the next submission attempt. The PDF in this repo is the post-reference-cleanup pre-ORCID working draft.
