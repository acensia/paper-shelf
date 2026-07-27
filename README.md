# paper-shelf

A personal shelf of machine-learning research papers, organized by venue and topic.
Each paper gets its own folder holding the source PDF and a small metadata file.

Two kinds of shelf:

- **Venue collections** — a curated selection from **ICML 2026** (Seoul, July 6–11, 2026).
- **Theme collections** — papers grouped by topic rather than venue: **VLA** (vision-language-action
  foundations), **NVIDIA** (papers led by NVIDIA), and **long-horizon tasks** (LLM / VLM / VLA),
  the last kept as a git submodule at [`LongHorizon/`](https://github.com/acensia/long-horizon-papers).

## Repository structure

```
.
├── README.md            ← you are here (repo structure & conventions)
├── .gitignore           ← excludes PDFs and .DS_Store
├── VLA/                 ← theme: VLA foundations (RT-1/RT-2, OpenVLA, Octo, π₀, …)
│   ├── README.md        ← annotated index, sorted newest-first
│   └── <Paper_Title>/   (per-paper layout as below)
├── NVIDIA/              ← theme: NVIDIA-led papers (GR00T, Cosmos, Eureka, …)
│   ├── README.md
│   └── <Paper_Title>/
├── LongHorizon/         ← theme (git submodule): long-horizon task papers
│   ├── README.md
│   ├── LLM/ · VLM/ · VLA/
│   │   └── <Paper_Title>/
├── VLAStudy/            ← git submodule: focused VLA model studies
│   └── <Paper_Title>/   ← source metadata and local review materials
└── ICML2026/            ← venue
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
- **Orals / top categories** live directly under `ICML2026/<Category>/`.
- **Spotlights** are grouped under `ICML2026/Spotlights/<Category>/`.
- **Theme collections** are flat — paper folders sit directly under the theme (`VLA/<Paper_Title>/`).
  A theme only subdivides into categories once it's big enough to need them, as `LongHorizon/` does.
- Each theme's `README.md` is its annotated index, sorted newest-first.
- A paper is filed in **one** theme, not duplicated; cross-references go in the theme READMEs.

## Theme contents

| Theme | Scope | Papers |
|-------|-------|-------:|
| [`VLA/`](VLA) | VLA foundations — RT-1/RT-2, Open X-Embodiment, OpenVLA, Octo, π₀, Gemini Robotics | 19 |
| [`NVIDIA/`](NVIDIA) | NVIDIA-led work — GR00T, Cosmos, Eureka, MimicGen, Isaac Lab, Megatron-LM | 18 |
| [`LongHorizon/`](https://github.com/acensia/long-horizon-papers) | Long-horizon tasks across LLM / VLM / VLA (submodule) | 36 |

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

See [`ICML2026/README.md`](ICML2026/README.md) for the full annotated index with abstracts.

## A note on PDFs

PDFs are **gitignored** and not stored in this repo — roughly 1.2 GB if you download everything
(~700 MB for the ICML 2026 set, 521 MB across the 37 files in `VLA/` and `NVIDIA/`). They are
author arXiv preprints, re-downloadable from the links in each `source.txt`. Only the
text content (notes, metadata, indexes) is version-controlled.

This may migrate to [Git LFS](https://git-lfs.com/) later. To do so: install `git lfs`,
run `git lfs track "*.pdf"`, remove the `*.pdf` line from `.gitignore`, and commit.
