
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20479271.svg)](https://doi.org/10.5281/zenodo.20479271)

# DCR-Attention — single-paper assembly (`paper-v1.0`)

**Title.** *DCR-Attention: Exact Q-Axis Sparse Decode at Production Scale —
A Measured Scaling Law and Retention of Long-Range Dependence, under Matched-Control
Discipline.*

**Authors.** Evgeny Vyaltsev, Daniil Vyaltsev. ORCID 0009-0004-3712-6798.

**Release route.** GitHub tag `paper-v1.0` → Zenodo DOI (priority record). Not arXiv.
Same material may later be submitted to TMLR (rolling) or a 2027 conference cycle.

## What this is
One paper assembled from the existing project corpus (Candidate α of the synthesis).
It unifies four research threads around a single object — softmax attention
concentration — and validates every quantitative claim under an explicit
matched-control discipline.

Contributions: exact Q-axis ranking (Theorem 1); a measured, descriptively-stated
scaling law ΔPPL = O(N^{-(1-α)}), α≈0.31 (Theorem 3); retention of long-range
dependence to N=128K via true selection, with a leakage dissociation; a per-query
correctness certificate; a multi-batch memory-wall systems result with a fused ABKV
kernel (measured 1.21× over SDPA); and the matched-control methodology, with an
explicit retraction ledger.

## Layout
```
paper/main.tex     LaTeX source (compiles with pdflatex, 3 passes)
paper/main.pdf     compiled preprint (13 pp.)
PLAN.md            assembly plan (section→source map, deferred items, ledger)
REPRO.md           registered-run provenance and environment
docs/              synthesis note (rationale for Candidate α)
LICENSE            CC-BY-4.0
```

## Build
```
cd paper && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

## How to cite
```
@misc{dcr_attention_2026,
  author = {Vyaltsev, Evgeny and Vyaltsev, Daniil},
  title  = {DCR-Attention: Exact Q-Axis Sparse Decode at Production Scale},
  year   = {2026}, note = {Zenodo, paper-v1.0}, doi = {<MINTED-ON-RELEASE>}
}
```

## Scope honesty
Five items are explicitly **deferred** (stated as in-progress / future, not claimed):
multi-seed retention boundary; causal boundary mechanism (starvation vs mis-selection,
inject running); TTFT decomposition; causal upgrade of Theorem 3 (Gap-1); cross-model α.
Eight retracted hypotheses are listed in the paper's retraction ledger (§9).
