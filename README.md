# paper-shelf

A personal shelf of machine-learning research papers, organized by venue and topic.
Each paper gets its own folder holding the source PDF and a small metadata file.

Currently tracks a curated selection from **ICML 2026** (Seoul, July 6–11, 2026).

## Repository structure

```
.
├── README.md            ← you are here (repo structure & conventions)
├── .gitignore           ← excludes PDFs and .DS_Store
└── IMCL2026/
    ├── README.md        ← curated content index (papers + abstracts)
    ├── report.md        ← long-form notes / write-up
    ├── <Category>/
    │   └── <Paper_Title>/
    │       ├── <Paper_Title>__arXiv-<id>.pdf   (not tracked — see below)
    │       └── source.txt                       (title, venue, arXiv + OpenReview links)
    └── Spotlights/
        └── <Category>/
            └── <Paper_Title>/ ...
```

### Conventions

- **One folder per paper**, named after the paper title (spaces → `_`).
- Each paper folder contains:
  - `<title>__arXiv-<id>.pdf` — the author preprint (when available).
  - `source.txt` — title, venue/track, and links (arXiv, OpenReview).
  - `NO_PDF_AVAILABLE.txt` — placeholder when no preprint exists yet, in place of the PDF.
- **Orals / top categories** live directly under `IMCL2026/<Category>/`.
- **Spotlights** are grouped under `IMCL2026/Spotlights/<Category>/`.

## ICML 2026 contents

| Track | Category | Papers |
|-------|----------|-------:|
| Orals | LLM_Reasoning | 9 |
| Orals | Multimodal_Vision | 5 |
| Orals | Agents_Benchmarks | 4 |
| Orals | Robotics | 3 |
| Orals | Causality | 1 |
| Orals | Interpretability_Memory | 1 |
| Orals | Reinforcement_Learning | 1 |
| Orals | Spatial_3D_Vision | 1 |
| Orals | Theory_World_Modeling | 1 |
| Orals | Trustworthiness_Hallucination | 1 |
| Spotlights | LLM_Reasoning | 127 |
| Spotlights | Multimodal_Vision | 41 |
| Spotlights | Robotics | 20 |

~215 papers total · 102 with PDFs · 113 awaiting preprints.

See [`IMCL2026/README.md`](IMCL2026/README.md) for the full annotated index with abstracts.

## A note on PDFs

PDFs (~700 MB across 102 files) are **gitignored** and not stored in this repo. They are
author arXiv preprints, re-downloadable from the links in each `source.txt`. Only the
text content (notes, metadata, indexes) is version-controlled.

This may migrate to [Git LFS](https://git-lfs.com/) later. To do so: install `git lfs`,
run `git lfs track "*.pdf"`, remove the `*.pdf` line from `.gitignore`, and commit.
