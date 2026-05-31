# План статьи (Candidate α) — DCR-Attention, единая сборка

**Версия плана:** 1.0 — 2026-05-31
**Тип:** assembly plan для одной статьи из существующего корпуса
**Релиз:** GitHub tag `paper-v1.0` → Zenodo DOI (приоритет), не arXiv
**Дисциплина:** descriptive ≠ causal; pre-registration; matched-control; scope-honesty;
явный retraction ledger. Каждое число — из зарегистрированного прогона в корпусе.

---

## Заголовок

*DCR-Attention: Exact Q-Axis Sparse Decode at Production Scale — a Measured Scaling
Law and Retention of Long-Range Dependence, under Matched-Control Discipline.*

## Тезис (one-liner)

Один объект — концентрация softmax — даёт точную алгебру выбора (Theorem 1),
измеренный закон масштабирования качества (Theorem 3), удержание дальней зависимости
при sparse-decode (retention) и per-query сертификат корректности; всё валидировано
под matched-control дисциплиной, рождённой в *закрытой* спектральной программе.

## Что несёт статья (контрибуции, в порядке новизны)

1. **C1** Точная Q-ось (Theorem 1) — training-free, нефальсифицируемое тождество; 45/45 brute-force, Jaccard ≥ 0.998.
2. **C2** Закон масштабирования (Theorem 3) — $\Delta\mathrm{PPL}=O(N^{-(1-\alpha)})$, $\alpha\approx0.31$; два независимых эмпирических подтверждения; descriptive.
3. **C3** Retention (ядро) — sparse-decode удерживает дальнюю зависимость через истинную selection до N=128K при c≥~0.10; leakage dissociation при экстремально низком c.
4. **C4** Per-query сертификат $\varepsilon\le 2W e^{H(\alpha)}/k$ + точная неасимптотическая геометрия ошибки (V-Gram cancellation) — checkable accuracy guarantee.
5. **C5** Memory-wall pathology + ABKV-kernel (mode A, измерено 1.21× SDPA) — kernel rewrite как структурный prerequisite.
6. **C6** Структурный негатив (Prop 5.13) + multi-axis предиктор $r^\star$ (Pearson 0.9925) — предсказание режима провала.
7. **C7** Matched-control методология (3-компонентный чеклист) + production-артефакт (sort+explainable dispatch).

## Структура (10 разделов + 4 приложения)

| § | Заголовок | Источник | Готовность |
|---|---|---|---|
| 1 | Introduction (3 open questions) | qaxis §1 | LIGHT |
| 2 | Background & related work | Geometric_Memory §1 + qaxis §6 | READY |
| 3 | Theory: entropy concentration | qaxis §3 (Thm1/Thm3) + Φ v2.2 §3.25,§4 | READY |
| 4 | Empirical scaling (hero) | qaxis §4 (Tab 1–3) | READY |
| 5 | Retention of long-range dependence | RETENTION_STUDY §1–4 (core CONFIRMED) | READY ядро / future граница |
| 6 | Systems: kernel & multi-batch wall | qaxis §5 + phase_8_1a (ABKV) | READY |
| 7 | Methodology: matched-control discipline | matched_controls_note + RESEARCH_PROGRAM_SUMMARY | READY |
| 8 | Distinction from concurrent work | qaxis §6 + Geometric_Memory | READY |
| 9 | Limitations & retraction ledger | синтез §3.2 (R1–R8) | READY |
| 10 | Conclusion & future work | синтез §6 | LIGHT |
| App A | Theorem 3 proof | Document 1 / qaxis App A | READY |
| App B | Kernel engineering plan M0–M7 + ABKV | Document 3 + phase_8_1a | READY |
| App C | Production artifact: sort + explainable dispatch | DCR_stress + EXPLAINABLE_RUNTIME | READY |
| App D | Matched-control raw + Detector v2 calibration | matched_controls_note + Φ v2.2 §6 | READY |

## Ключевые числа (canonical, со ссылкой на источник)

- $\alpha\approx0.31$; $\alpha_h\in[0.22,0.39]$; $K_{\mathrm{eff}}\approx87$@20K; $K_{\mathrm{eff}}/N\approx0.4\%$ — qaxis §3.
- Hero rank-window: $+0.308\%$@(20K, c=0.5); top-K Pareto sweet spot $+0.118\%$@(8K, c=0.5) — qaxis Tab 1, 3.
- top-K vs rank-window 3.06–3.32× iso-coverage — qaxis Tab 2.
- PCA-коллапс 149×→438× (N=500→2K); latency 5×→36× — qaxis §4.3.
- Theorem 3 проверка: предсказано $4^{0.69}=2.66\times$, измерено $2.42\times$ (N=2K→8K) — qaxis §4.6.
- Memory wall: B=8 → 16 GB, latency 43→2000+ ms; SDPA per-elem 0.29× — qaxis §5.5.
- ABKV mode A: per-layer 0.478 ms; e2e 166.08 ms; **1.21× SDPA = 95.4% теор. потолка 1.269×** — phase_8_1a §1.5.
- Pass-2 dominance: 74.4% M4 e2e на L4 (поправка R8) — phase_8_1a §1.2.
- Matched-control нуль: NLL diff $+2.72\times10^{-4}$, CI $[-1.7,7.0]\times10^{-4}$, Wilcoxon $p=0.26$ — matched_controls Tab 1.
- Per-query сертификат: 0 false-negatives на 45 калибровочных ячейках — Φ v2.2 Lemma 6.1.
- Retention: gate v3 ROBUST, 0 FAIL на 10 informative cells @c=0.15; leakage out_corr/correct до 0.70 — RETENTION_STUDY §3–4.

## Явно отложено (НЕ заявлять решённым; §5 in-progress + §10 future)

- Граница retention как multi-seeded факт (M1).
- Каузальный механизм границы — starvation vs mis-selection (M2/M3, inject running).
- TTFT-декомпозиция длинного контекста (M4).
- Каузальный апгрейд Theorem 3 (Gap-1, matched-magnitude).
- Cross-model $\alpha$ как закон (Направление 2).

## Retraction ledger (§9, явно)

R1 спектральный механизм; R2 U-профиль; R3 3-метрическая связка; R4 $\psi\to\tau_k^2$;
R5 K_crit; R6 mis-selection (инвертировано данными); R7 super-linear overhead;
R8 Pass-3 bottleneck. Все — с указанием, *как* пали.

## Пакет релиза

```
dcr-attention/  (tag paper-v1.0)
├── paper/
│   ├── main.tex
│   └── refs (thebibliography, inline)
├── figures/        (из qaxis Fig 1–6, при наличии)
├── README.md       (что это, как цитировать, DOI placeholder)
├── REPRO.md        (registered runs, single-command)
└── LICENSE
→ GitHub release → Zenodo DOI → ORCID 0009-0004-3712-6798
```

## Порядок написания глав (по «несущести»)

§7 (методология) → §3 (теория) → §4 (hero) → §5 (retention) → §6 (systems) →
§2/§8 (background/distinction) → §1/§10 (intro/conclusion) → §9 (limitations/ledger) →
приложения. Сборка в один main.tex.

*Конец плана.*
