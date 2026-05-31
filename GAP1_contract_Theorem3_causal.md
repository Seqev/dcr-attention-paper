# Gap-1 — Pre-registered matched-control causal test of Theorem 3

**Type:** registered experiment contract (architect → Claude Code)
**Date:** 2026-05-31
**Decision authority:** Seqev
**Target:** upgrade Theorem 3 from *descriptive* to *causally grounded* — or close it as
descriptive under a matched-control null. Either outcome ships in `paper-v1.1`.
**Discipline:** INS-27 (diagnostic-first, contract tests before fixes), INS-28 (fp32
softmax everywhere), INS-29 (grep-verify all HF internal APIs). No bundled fixes, no
speculative changes. **Pre-registration is binding: both outcomes are fixed below
before any run.**

---

## 0. Question and pre-registration

Theorem 3 claims $\Delta\mathrm{PPL}(N)=O(N^{-(1-\alpha)})$ *because* decode attention is
entropy-concentrated. That premise (concentration is load-bearing) is currently
**descriptive**. Gap-1 tests it causally with a magnitude-matched intervention, exactly
as the matched-control note tested (and falsified) the spectral-mode hypothesis.

**Primary metric:** paired next-token NLL differential
$D_{\mathrm{NLL}} = (\mathrm{NLL}_{\text{treat}}-\mathrm{NLL}_{\text{ctrl}})$ per
measurement (operationally meaningful; KL is secondary — see the NLL/KL dissociation in
the matched-control note).

**Pre-registered outcomes (decided now):**
- **H0 / DESCRIPTIVE:** 95% bootstrap CI of $D_{\mathrm{NLL}}$ brackets 0 **and**
  Wilcoxon signed-rank $p>0.05$ → entropy concentration is *not* causally load-bearing
  beyond matched magnitude. Theorem 3 is written as **explicitly descriptive**, citing
  this matched-control null. (This is a legitimate, citable result — see nit A.)
- **H1 / CAUSAL-GROUNDED:** CI of $D_{\mathrm{NLL}}$ strictly $>0$ **and** $p<0.05$ →
  concentration is causally load-bearing. Add a causal-validation paragraph to §3.2 and
  strengthen the abstract.

No reinterpretation of a null as "weak positive" is permitted. The decision is read off
the CI + Wilcoxon, fixed above.

---

## 1. Matched-magnitude construction (in logit space)

For a query at (layer $\ell$, head $h$, decode position $t$), let
$\boldsymbol{\ell}\in\mathbb{R}^N$ be the pre-softmax logits
$\boldsymbol{\ell}=QK^\top/\sqrt{d}$, $p=\mathrm{softmax}(\boldsymbol{\ell})$,
$H(p)=-\sum_i p_i\log p_i$.

Entropy gradient (derive in code, assert against finite-difference check before use):
$$g := \nabla_{\boldsymbol{\ell}} H = -\,p\odot(\log p + H).$$

- **Treatment $\delta_T$ (destroy concentration):** step along $+g/\|g\|$ sized so the
  realized $\Delta H \approx \Delta_{\text{target}}$ nats (raise entropy toward uniform).
  Use a per-query line search to hit $\Delta_{\text{target}}$; record realized
  $\Delta H_T$ and $\|\delta_T\|_2$.
- **Matched control $\delta_C$ (preserve concentration, equal magnitude):** draw random
  $r\sim\mathcal{N}(0,I_N)$, project onto $\{v:\langle v,\hat g\rangle=0\}$:
  $r_\perp = r-(r\cdot\hat g)\hat g$; set $\delta_C=\|\delta_T\|_2\cdot \hat r_\perp$. By
  construction $\langle\delta_C,g\rangle=0\Rightarrow \Delta H_C\approx0$ to first order.

**Critical invariant (the nit-A lesson):** match $\|\delta\boldsymbol{\ell}\|$ — the
*perturbation* — not $\|\boldsymbol{\ell}\|$. **Contract test:** per cell, assert
$\|\delta_T\|/\|\delta_C\| = 1.000 \pm 10^{-3}$ and measured $|\Delta H_C| \le 0.1\,\Delta_{\text{target}}$.
If $|\Delta H_C|$ is not $\ll\Delta_{\text{target}}$, the linearization is inadequate →
**stop and report** (do not proceed with a biased control); re-derive with a
second-order correction.

Apply each perturbation to $\boldsymbol{\ell}$, recompute $p'$ in **fp32** (INS-28),
recompute attention output and next-token NLL vs the unablated baseline. Three forward
passes per measurement: baseline / treatment / matched-control.

---

## 2. Secondary, Theorem-3-specific readout (same forwards, no extra cost)

For each of the three passes, also compute the **top-$k$ truncation gap**
$\Delta\mathrm{PPL}_{\text{top-}k\text{ vs dense}}$ at $k_{\mathrm{eff}}=\lfloor c N\rfloor$,
$c=0.5$. Pre-registered secondary differential
$D_{\text{gap}}=\Delta\mathrm{PPL}^{\text{treat}}_{\text{gap}}-\Delta\mathrm{PPL}^{\text{ctrl}}_{\text{gap}}$.
This tests Theorem 3's mechanism directly: does concentration causally control the
*truncation* error (not just dense prediction)? Report alongside primary; it does not
change the H0/H1 gate but sharpens interpretation.

---

## 3. Random-K baseline (checklist component 2; contextual, non-gating)

Re-run the Figure-1 entropy fit $H_N=\alpha\log N+\beta$ on a model whose K-projection
weights are re-initialized (untrained), keeping everything else fixed. Report
$\alpha_{\text{randomK}}$ vs $\alpha_{\text{trained}}\approx0.31$:
- $\alpha_{\text{randomK}}\approx1$ → $\alpha<1$ is a fact about *training* (strengthens
  the premise).
- $\alpha_{\text{randomK}}<1$ → partly algebraic/architectural (note in Limitations).

Contextualizes the result; does not gate H0/H1.

---

## 4. Harness (reuse, don't rebuild)

Start from the existing matched-control ablation harness (the one behind
`matched_controls_note`, 300-measurement design). **Swap only** the intervention:
key-space dominant-mode ablation → logit-space matched-magnitude perturbation of §1.

- Grep-verify (INS-29) the attention forward in `models/llama/attention.py` to locate
  where $\boldsymbol{\ell}=QK^\top/\sqrt d$ is materialized before softmax, and confirm
  the fp32-softmax path (INS-28). Do **not** assume the hook point — grep it.
- Cells: 12 (layer, head) pairs spanning early/mid/late layers; 25 WikiText-2 prompts;
  context $N=256$ (minimal, harness-native). **Optional extension** if budget allows:
  sweep $N\in\{256,1\text{K},8\text{K}\}$ to show the differential's $N$-trend (more
  faithful to the scaling claim). Start with $N=256$; extend only after the $N=256$
  cell passes the §1 contract tests.
- $\Delta_{\text{target}}$: pilot at 3 values (e.g. $\{0.25,0.5,1.0\}$ nats) on one cell;
  pick the smallest that produces a measurable baseline effect, then lock it for the run.
- seed=0 per-config `torch.Generator`; log route counters per cell (no silent SDPA
  fallback); B=1, greedy.

---

## 5. Statistics (mirror the matched-control note exactly)

- Paired bootstrap (10k resamples) 95% CI on $D_{\mathrm{NLL}}$ and $D_{\text{gap}}$.
- Wilcoxon signed-rank $p$ on the paired differential.
- Report $\Delta_{\text{treat}}$, $\Delta_{\text{ctrl}}$, differential, CI, $p$ — same
  table shape as matched-control Table 1.
- Multi-seed: **required** before the claim is final (P0b precedent: single-seed hero
  claims failed multi-seed). Minimal run = single seed for the go/no-go; if it lands near
  the decision boundary, escalate to 3 seeds before writing.

---

## 6. Acceptance gates / decision tree

1. **Contract gates first** (else invalid, do not interpret): $\|\delta\|$ ratio
   $=1.000\pm10^{-3}$; $|\Delta H_C|\le0.1\,\Delta_{\text{target}}$; finite-difference
   check on $g$ passes; route counters clean.
2. **Then read primary:**
   - CI brackets 0 **and** $p>0.05$ → **DESCRIPTIVE** branch.
   - CI $>0$ **and** $p<0.05$ → **CAUSAL-GROUNDED** branch.
   - Near boundary (CI touches 0 / $0.01<p<0.1$) → run 3 seeds, then re-decide; do not
     write either branch yet.
3. Report secondary $D_{\text{gap}}$ and random-K $\alpha$ as interpretation, not gates.

---

## 7. Paper integration (both branches pre-written)

- **DESCRIPTIVE:** §3.2 keeps "stated as a description"; add one sentence + a table
  reporting the matched-control null (CI, $p$); abstract unchanged. Cite as the honest
  causal status. Retraction ledger unaffected.
- **CAUSAL-GROUNDED:** §3.2 gains a "Causal validation" paragraph + the table (now
  significant); abstract upgrades "(Theorem 3, descriptive)" → "(Theorem 3, causally
  validated under matched control)"; §10 Gap-1 bullet moves from future to done.
- Either way: ship as `paper-v1.1` (versioned Zenodo DOI; concept-DOI unchanged). The
  **pre-registration itself** can be stated in `paper-v1.0` §10 as a registered protocol,
  which stakes priority on the test design now.

---

## 8. Deliverables and effort

- `scripts/gap1_matched_logit.py` (harness, derived from existing matched-control script)
- `gap1_results.json` (per-cell: $\Delta_{\text{treat}},\Delta_{\text{ctrl}}$, $D_{\mathrm{NLL}}$, $D_{\text{gap}}$, $\|\delta\|$ ratio, $\Delta H_C$, route counters)
- `gap1_report.md` (Table-1-shaped summary + decision-tree verdict + random-K $\alpha$)
- Contract tests in `tests/` for §1 invariants (run before the experiment, per INS-27)

**Effort:** variation of an existing harness → order of hours of GPU on RTX 4060 Ti for
$N=256$ single-seed; +hours for the $N$-sweep and 3-seed escalation if triggered.

---

## 9. What this does and does not close

- **Closes:** the causal *status* of Theorem 3's premise (concentration load-bearing or
  not) — definitively, in whichever direction the data points. Stakes priority on the
  pre-registered causal test design immediately (in v1.0 §10) and on the result in v1.1.
- **Does not close:** the exponent value $\alpha$ itself (still measured), cross-model
  generality (separate Направление 2), or the retention boundary (separate gates M1–M4).

**End of Gap-1 contract. Awaiting Seqev "go" before dispatch to Claude Code.**
