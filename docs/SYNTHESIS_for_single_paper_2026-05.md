# Сквозное обобщение проекта → сборка одной статьи

**Документ:** Synthesis / assembly note
**Автор:** Claude (architect), в соавторстве с Seqev
**Дата:** 2026-05-31
**Назначение:** свести весь корпус (10 PDF + 9 markdown) в одну карту, отделить
публикабельное от отозванного, и предложить скелет **одной** статьи с честными
альтернативами. Дисциплина корпуса (descriptive ≠ causal; pre-registration;
matched-control; scope-honesty; falsification over narrative) перенесена в сам
этот документ.

---

## 0. Резюме в одном абзаце

Весь корпус изучает **один физический объект — концентрацию softmax-распределения
внимания** — под четырьмя именами и в четырёх эпистемических статусах. Это даёт
готовую ось для единой статьи. Рекомендуемая сборка: **«DCR-Attention: точная
Q-ось, измеренный закон масштабирования и удержание дальней зависимости при
sparse-decode, валидированные под matched-control дисциплиной»** — где живой
контент (нити B + retention) несёт результат, а методология matched-control
(рождённая в *закрытой* спектральной нити A) делает каждое утверждение
переживающим рецензию. Φ-Transport (нить C) даёт per-query сертификат и
негативные результаты как теоретический хребет; DCR Sort (нить D) — production-
артефакт в приложении. Альтернатива — отдельная **methods-статья** про matched-
control детектор «механизм-vs-декорация»; она тоже сильна, но раскалывает корпус
на две публикации.

---

## 1. Инвентаризация: что есть и в каком статусе

| Файл | Содержание | Статус | Текст/растр |
|---|---|---|---|
| `RESEARCH_PROGRAM_SUMMARY.md` | Нить A — Spectral Gap, 5 стадий, каузальный нуль | **закрыта (честно)** | text |
| `matched_controls_note.pdf` | Каузальный чеклист из 3 компонент (нить A) | **публикабельный актив** | text (6 c.) |
| `Cross_Project_Analysis_2026-05.md` | Сквозная карта 4 нитей + внешняя литература | мета-карта (опора) | text |
| `Geometric_Memory_Deep_Research.md` | Карта SOTA 2024–2026 + 4 уникальных места проекта | мета-карта (опора) | text |
| `qaxis_arxiv.pdf` (=Document 2 = Paper 2) | Нить B — Thm 1, Thm 3, hero, M1, PCA-коллапс, memory wall | **живая, основной контент** | text (18 c.) |
| `Document_1_Paper1_Appendix_D.pdf` | Нить B — доказательство Theorem 1 (Paper 1 App. D) | финализирован | растр → digest |
| `Document_3_Engineering_Plan.pdf` | Нить B — kernel-план M0–M7 (= qaxis §5.2 / App B) | фикс | растр → digest |
| `Document_4_Track3_MultiBatch.pdf` | Нить B — multi-batch (= qaxis §5.5 memory wall) | измерен v2 | растр → digest |
| `ArXiv_Submission_Combined.pdf` | Сборка Documents 1–4 | пакет | растр |
| `RETENTION_STUDY_DRAFT.md` | Нить B′ — удержание дальней зависимости при top-K decode | **DRAFT, near-submit** | text |
| `CHANGELOG.md` | §5.5 переписан под измеренный Track 3 | служебный | text |
| `preview_v2_2_positioning.pdf` | Нить C — Φ-Transport, актуальная версия | **в работе** | text (69 c.) |
| `preview_v1_0_paper_complete.pdf` | Нить C — Φ-Transport, полная v1.0 | вытеснена v2.2 | растр |
| `preview_v0_9_section3_4_5.pdf` | Нить C — §3–5 (Brenier/CDF/PIC теоремы) | вытеснена v2.2 | text (29 c.) |
| `preview_section_3_1.pdf` | Нить C — ранний §3.1 (TODO-плейсхолдеры) | вытеснена v2.2 | text (5 c.) |
| `DCR_stress_REPORT.md` | Нить D — AVX-512 stress v4.1.0-rc1 (2800 кейсов) | продакшн | text |
| `EXPLAINABLE_RUNTIME_v420_rfc_rev2.md` | Нить D — explainable dispatch RFC | ratified | text |
| `M6_TRITON_TOPK_FALLBACK.md` | Нить B — Triton top-K fallback (dormant) | dead-code, fallback | text |
| `phase_8_1a_design_decision.md` | Нить B — ABKV Day-2 kernel architecture | CLOSED | text |

**Растровые PDF без текстового слоя** (Documents 1/3/4, ArXiv_Combined, preview_v1.0)
не извлекаются `pdftotext`, но их содержание **полностью** покрыто текстовыми
эквивалентами: Theorem 1 — в абстракте `qaxis` и §2.1 `Geometric_Memory`;
Engineering Plan — в `qaxis` §5.2 + Appendix B; Track 3 — в `qaxis` §5.5. Отдельная
растеризация не нужна для синтеза.

---

## 2. Единая ось: один объект, четыре имени

Объект, на который нанизан весь корпус:
$$p_{N,i} = \mathrm{softmax}\!\big(\langle Q, K_i\rangle/\sqrt{d}\big)_i,
\qquad H_N = -\sum_i p_{N,i}\log p_{N,i},\qquad K_{\mathrm{eff}}(N)=e^{H_N}.$$

| Нить | Имя объекта | Формальный носитель |
|---|---|---|
| **A** Spectral | spike-dominated Hessian | $H=-\beta\,\mathrm{Cov}_p(k)$ — ковариация под концентрированной мерой |
| **B** Q-axis | entropy concentration / бюджет | $H_N=\alpha\log N+\beta$, $K_{\mathrm{eff}}\sim N^\alpha$, $\Delta\mathrm{PPL}=O(N^{-(1-\alpha)})$ |
| **C** Φ-Transport | routing entropy | $\varepsilon\le 2W\,e^{H(\alpha(q))}/k_{\mathrm{eff}}$ (Thm 3.25) |
| **D** DCR Sort | bucket occupancy entropy | $H(p)\le\log B$ (Thm 3.13), `inversion_density ρ` как сигнал концентрации |

**Сквозной урок (воспроизводимый, не философский).** Нити A и B изучают
**буквально одно явление** на **одной модели** (Llama-3.2-1B) на **одном датасете**
(WikiText-2). Различие — только в эпистемическом статусе:
- **A заявила механизм** → matched-magnitude ablation доминантной моды → **нуль**
  (differential NLL $+2.7\times10^{-4}$, 95% CI скобки на нуле, Wilcoxon $p=0.26$).
- **B заявила алгебру + описание** → Theorem 1 нельзя фальсифицировать ablation'ом
  (это тождество ранжирования); Theorem 3 честно помечена «measured scaling exponent
  per-model» → пережила, дала hero $+0.118\%\,\Delta$PPL при $N=8$K.

Вывод корпуса (`Cross_Project_Analysis` §2.1): *в текущем состоянии науки об
attention механистические заявления о геометрии не проходят matched-control;
описательные и алгебраические — проходят.* Это и есть единственный реально новый
эпистемический результат проекта, и он должен быть методологическим хребтом любой
статьи.

---

## 3. Реестр активов: что публикабельно vs что отозвано

### 3.1. Переживающее рецензию (использовать)

| # | Результат | Источник | Статус строгости |
|---|---|---|---|
| S1 | **Theorem 1** — точная Q-ось $\langle u_Q,K_i\rangle{>}\langle u_Q,K_j\rangle \Leftrightarrow \langle Q,K_i\rangle{>}\langle Q,K_j\rangle$, $u_Q{=}Q/\|Q\|$ | Paper 1 App. D; qaxis | **алгебраическое тождество** (нефальсифицируемо ablation'ом); подтверждено 45/45 brute-force, Jaccard ≥ 0.998 |
| S2 | **Theorem 3** — $\Delta\mathrm{PPL}(N){=}O(N^{-(1-\alpha)})$, $\alpha{\approx}0.31$; hero $+0.308\%$@20K, top-K $+0.118\%$@8K (50% saving) | qaxis §3–4.6 | **descriptive** (фит по одной модели); 2 независимых подтверждения; **каузально не проверена** (см. §6 Gap-1) |
| S3 | **Per-query сертификат** $\varepsilon\le 2W e^{H(\alpha(q))}/k_{\mathrm{eff}}$ + Detector v2, exp-decay no-false-negative (Lemma 6.1), 0 FN на 45 калибр. ячейках | Φ v2.2 Thm 3.25/Cor 3.27 | **сильнейший теор. вклад нити C** |
| S4 | **Точная неасимптотическая геометрия ошибки** $\varepsilon^2{=}\sum_{ij}\gamma_i\gamma_j K_{ij}$ + V-Gram cancellation $\sigma^2_{\mathrm{eff}}{=}K_{ii}{-}\bar c$; валид. на Llama-3.2-1B до 0.5% | Φ v2.2 §4, Thm 4.7 | **без прямого предшественника** в sparse-attn литературе |
| S5 | **Структурный негатив** Prop 5.13 — sparsity в Q-спектре ⇏ rank-local recoverability под single-axis PCA; реализован как BERT-MNLI −17pt | Φ v2.2 Prop 5.13 | **контролируемый негатив** — переживает по уроку нити A |
| S6 | **Multi-Axis предиктор** $r^\star{=}\lceil 1.142\,d_{\mathrm{eff}}(Q){-}1.048\rceil$, Pearson 0.9925 (40 trials, 32–86% error reduction) | Φ v2.2 §5.7 | предсказывает *режим провала* — сильнее, чем предсказание успеха |
| S7 | **Matched-control чеклист** (matched-magnitude + random-spectrum baseline + pre-registration) | matched_controls_note | **главный недоиспользуемый актив**; окно для methods-публикации открыто |
| S8 | **Retention boundary** — Q-axis top-K удерживает дальнюю зависимость без потери селективности (wrong-needle diff CI_upper ≤ 0.05) до N=128K при c ≥ ~0.10; через *истинную* selection (needle-in-top-k ≈ correct) | RETENTION_STUDY | **CONFIRMED descriptive**; near-submit (см. §6 gates) |
| S9 | **Leakage dissociation** — при c=0.02/128K большинство верных ответов **без** needle в top-k (out_corr/correct до 0.70) → факт делокализован на prefill; accuracy перестаёт быть валидным индикатором механизма | RETENTION_STUDY F2 | **чистый новый механистический результат** + compression corollary |
| S10 | **Memory wall** — Python masked-dense DCR падает между B=4 и B=8 (16 GB, allocator fragmentation), kernel rewrite — структурный prerequisite, не оптимизация | qaxis §5.5; Track 3 | измерен; сильнее исходной гипотезы |
| S11 | **PCA-коллапс при масштабе** — Q-axis advantage 149×→438× (N=500→2K), latency gap 5×→36× | qaxis §4.3 | измерен |
| S12 | **Production sort + explainable dispatch** — v4.1.0-rc1, 2800 correctness ops, det 50/50 bit-identical; RFC «runtime decisions become falsifiable» | DCR_stress, EXPLAINABLE_RUNTIME | продакшн-артефакт |

### 3.2. Отозванное / фальсифицированное (НЕ заявлять)

| # | Утверждение | Как пало | Дисциплина |
|---|---|---|---|
| R1 | Спектральная геометрия = ядро вычисления (доминантная мода load-bearing) | matched-magnitude нуль (Stage 5) | закрыть, не реанимировать (`Cross_Project` R1) |
| R2 | Универсальный U-профиль глубины спектральной концентрации | не воспроизвёлся на 5 архитектурах (Stage 3) | model-specific |
| R3 | 3-метрическая связка $\chi,S_\lambda,R$ как нетривиальная структура | воспроизведена случайным спектром (Stage 4) | алгебраическое тождество |
| R4 | $\psi(\alpha,k)\to\tau_k^2$ tail-equivalence; $\alpha^\star{=}r{-}1/2$ closed form | не предсказывает при операц. $k$ | контролируемый негатив (нить C) |
| R5 | «K_crit ≈ 2600» / абсолютный порог $k_{\mathrm{eff}}$ для retention | $k_{\mathrm{eff}}{=}19666$ FAIL, $k_{\mathrm{eff}}{=}1602$ holds | граница — это N×density interaction |
| R6 | Failure mode retention = mis-selection (unresolvable dot-product competition) | **инвертировано данными**: значима только starvation (CI [+0.02,+0.17]), mis-selection не детектируется | inject-проба running (§6 M2) |
| R7 | Super-linear Python overhead как механизм memory wall | механизм — allocator fragmentation | пересмотр сильнее исходного |
| R8 | Pass-3 как bottleneck kernel | Pass-2 (`torch.argsort` stable) = 74.4% M4 на L4 | поправка v3.1 (phase_8_1a §7.3) |

** Retraction ledger дисциплина:** все R-пункты документируются явно, не «тихо
дропаются». Это часть нарратива честности, который сам по себе является
конкурентным преимуществом.

---

## 4. Кандидаты на «одну статью»

### Кандидат α (РЕКОМЕНДУЕТСЯ) — Консолидированная DCR-Attention под matched-control дисциплиной

**Заголовок (рабочий):** *Q-Axis Sparse Attention at Production Scale: Exact
Ranking, a Measured Scaling Law, and Retention of Long-Range Dependence — under
Matched-Control Measurement Discipline.*

**Тезис:** объединить живой контент (S1, S2, S8, S9, S10, S11) с теоретическим
хребтом нити C (S3, S4, S5) и методологическим хребтом нити A (S7). Эпистемология
становится **методом** статьи, а не отдельной публикацией.

**Почему это сильнейшая сборка:**
- Несёт реальный результат (hero $+0.118\%$@8K, retention до 128K) — не только методологию.
- Matched-control дисциплина (S7) превращает каждое числовое заявление в
  переживающее: Theorem 3 либо каузально апгрейдится (Gap-1), либо честно
  остаётся descriptive — обе ветки публикабельны.
- Per-query сертификат (S3) даёт то, чего нет ни у DSA, ни у NSA, ни у Quest:
  checkable per-input accuracy guarantee.
- Negative results (S5, R4, R6) + retraction ledger = нарратив честности, который
  рецензент 2026 года вознаграждает (поле перенасыщено «красивыми унификациями»).

**Чего НЕ делать в этой сборке:** не вести абстракт унификацией Φ-Transport (R2 в
`Cross_Project`); унификация четырёх примитивов — §substrate, не контрибуция №1.

### Кандидат β (АЛЬТЕРНАТИВА) — Methods-статья про matched-control детектор

**Заголовок:** *Matched-Magnitude Controls for Causal Testing of Spectral
Mechanisms* (+ аудит чужих indexer'ов DSA/NSA).

**Тезис:** поднять S7 до самостоятельной methods-публикации и **применить к чужим
заявлениям**: lightning indexer DSA ($I_{t,s}{=}\sum w\cdot\mathrm{ReLU}(q{\cdot}k)$),
обученная маска NSA — механизмы или декорации? Прогнать через matched-magnitude.

**Плюсы:** высокоцитируемый methods-вклад; бьёт в слабое место («не догоняем по
числам — проверяем, чем оно является»); окно открыто (AIPsy-Affect, Méloux et al.
только подходят к этой дисциплине).
**Минус:** раскалывает корпус на 2 статьи; оставляет живой контент B/B′ без дома.

### Кандидат γ (НЕ рекомендуется как «одна статья») — Φ-Transport как unification paper

Унификация attention↔OT — занятое поле (Sinkformers, AMS-perspective Geshkovski–
Rigollet; «Transformers as OT» **desk-rejected** на ICLR 2026). Вести этим — антипаттерн.
Φ-Transport переживёт **своими негативными результатами и per-query сертификатом**,
не унификацией — поэтому его сильные части (S3–S6) идут *внутрь* Кандидата α.

**Решение — за Seqev.** Если цель «одна статья сейчас» → **α**. Если есть бюджет на
methods-трек и хочется максимума цитируемости → **α** как Paper 2-final + **β**
параллельно (но это уже две статьи).

---

## 5. Скелет рекомендуемой статьи (Кандидат α)

> Каждый раздел помечен источником-файлом для сборки.

**Abstract.** Вести с: (1) точная Q-ось training-free (S1), (2) измеренный закон
$O(N^{-(1-\alpha)})$ + hero (S2), (3) retention до 128K через истинную selection
(S8) + leakage dissociation (S9), (4) per-query сертификат (S3), (5) memory wall →
kernel как prerequisite (S10). Все числа scoped: «Llama-3.2-1B, WikiText-2, k_window=64».

**§1 Introduction.** Gap: quality-валидация sparse-attn капается на N≤1024; зазор
до production N∈[8K,128K]. Three open questions (Q1 scaling? Q2 PCA failure? Q3
deployable latency point?) — из `qaxis` §1.

**§2 Background.** Q-axis recap; long-context economics; почему N=500 недостаточно;
SOTA-таблица (selection-based / trained / low-rank) — из `Geometric_Memory` §1 +
`qaxis` §6. Чётко: training-free + exact ranking = ниша.

**§3 Theory.**
- §3.1 Theorem 1 (S1) — алгебраическое тождество, нефальсифицируемо.
- §3.2 Entropy concentration + Theorem 3 (S2) — $H_N=\alpha\log N+\beta$, α≈0.31.
  Привязать к внешнему якорю: критическое масштабирование $\beta_n\asymp\log n$
  (arXiv:2510.05554) — α как эмпирическая реализация фазового перехода.
- §3.3 Per-query сертификат (S3) + V-Gram error geometry (S4) — из Φ v2.2.

**§4 Empirical scaling (hero).** Таблицы 1–3 из `qaxis` §4: hero $+0.308\%$@20K;
top-K 3× over rank-window; Pareto sweet spot $+0.118\%$@8K; PCA-коллапс 149×→438× (S11).
Статистика: per-step ΔNLL z=5.32, p≪10⁻⁶, но в пределах bf16 noise floor (честно).

**§5 Retention of long-range dependence (новый, из RETENTION_STUDY).**
- Лестница gate v1→v2→v3 (single-needle → exact-id → soft disambiguation).
- c-sweep: граница в **N, не в c** (S8); все N≤32K держат до c=0.02.
- Leakage dissociation (S9) + compression corollary.
- **Честно:** mechanism at boundary = starvation, НЕ mis-selection (R6); K_crit
  framing не выживает (R5). Pending inject (M2).

**§6 Systems.** Kernel architecture (FlashDecoding + top-K, M0–M7); triple-tier
validation; memory wall (S10); ABKV mode A (phase_8_1a: predicted 1.21× SDPA на hero
L4, 95.4% теор. потолка); Pass-2 dominance поправка (R8).

**§7 Methodology — matched-control discipline (хребет).** Чеклист из 3 компонент
(S7); worked example asymmetry 10–30× ∥δK∥_F (matched_controls_note). Применить как
**стандарт валидации всех числовых заявлений статьи**, и явно — к Theorem 3 (Gap-1).

**§8 Related work / Distinction.** DSA/NSA/Quest/RetrievalAttention/Loki; Theorem 1
как special case MIPS↔NNS (VLDB 2025); приоритет vs arXiv:2512.03494 (low-entropy↔top-k),
2512.07647 (TV-distance bound). Честно: «only training-free path with exact ranking +
closed-form scaling law».

**§9 Limitations + Retraction ledger.** Single model/dataset; α model-specific;
Triton-числа projected; retention multihop floors на 1B. Явный retraction ledger
(R1–R8) — нарратив честности.

**§10 Conclusion + Future work.** Low-rank Q-axis (DCR-Loki, ~16× data movement),
larger models (α↓ ожидается), domain shift, cross-model retention.

**Appendices.** A: Theorem 3 proof (Document 1 / qaxis App A). B: Engineering Plan
M0–M7 (Document 3). C: production sort + explainable dispatch (DCR_stress + RFC) как
S12. D: matched-control raw (matched_controls_note Table 1).

---

## 6. Что домерить до сабмита (gates)

Из `RETENTION_STUDY` §5 + `Cross_Project` Направление 1:

- **Gap-1 (HIGHEST) — каузальный апгрейд Theorem 3.** Matched-magnitude по чеклисту
  S7: treatment = инъекция шума в логиты, поднимающая $H_N$ на Δ нат при мин. ∥δ∥;
  matched control = возмущение той же ∥δ logits∥_F, сохраняющее $H_N$ (перестановка
  масс); pre-register оба исхода; random-spectrum baseline (необученные K → тот же
  α?). Выход: либо первый каузально валидированный scaling-механизм, либо честный
  перевод в descriptive. ~часы GPU на 4060 Ti. **Это закрывает R3-риск нити A на нити B.**
- **M1 (retention) — multi-seed границы.** c* стоит на n=60 single-seed @128K;
  Paper-2 прецедент: single-seed hero claims падали под multi-seed (P0b mean
  +1.078% ±0.149%). Обязательно до публикации границы.
- **M2 (retention) — каузальный механизм (inject running).** oracle-inject needle на
  FAIL-промптах: recover? → решает starvation vs mis-selection (R6). НЕ писать
  механизм до возврата inject.
- **M3 (retention) — валидация needle_in_topk (anti-inject).** Вся leak-логика (S9)
  стоит на *реконструированном* nit; mirror-test (force needle OUT) обязателен.
- **M4 (retention) — latency с TTFT decomposition.** Любое «fast long context»
  обязано отделять TTFT (128K prefill, dense, НЕ ускорен) от per-token decode.
- **Cross-model α (Направление 2).** α на 5–7 моделях с разными softmax-режимами
  (vanilla+RoPE, SSMax, YaRN/Qwen, α-entmax). Закрывает R4 (model-specific) законом,
  не оговоркой. 1 GPU-день, векторизованный замер $H_N$ + lstsq-фит.

---

## 7. Риски и приоритет (risk register)

| # | Риск | Митигейшн |
|---|---|---|
| P1 | Каждый месяц задержки = −1 уникальное место (arXiv:2512.03494 уже качественно заявил low-entropy↔top-k) | сабмитить с тем, что есть; не идеализировать |
| P2 | Theorem 1 — special case MIPS↔NNS (VLDB 2025) | позиционировать как attention-specific форму с $u_Q{=}Q/\|Q\|$, в related work честно |
| P3 | Theorem 3 model-specific | Gap-1 + cross-model α (Направление 2) — закрыть законом |
| P4 | DSA/NSA уходят дальше по числам | ниша = training-free off-the-shelf + per-query сертификат, которого у них нет |
| P5 | Реанимация спектральной нити A | НЕ делать; каузальная стадия — где гипотеза должна была выжить |
| P6 | Riemannian формализм (Fisher metric, геодезические) | НЕ делать; overclaim, класс «декоративной геометрии», который A фальсифицировала |

---

## 8. Сухой итог для сборки

1. **Ось одна** — softmax-концентрация; четыре имени; используй контраст
   «механизм фальсифицирован / описание+алгебра выжили» как методологический хребет.
2. **Одна статья = Кандидат α**: DCR-Attention (S1,S2,S8–S11) + per-query сертификат
   и негативы из Φ (S3–S6) + matched-control дисциплина из A (S7) + production-артефакт
   из D (S12) в приложении.
3. **Веди абстракт результатом** (hero, retention), **не унификацией** (R2).
4. **Закрой Gap-1** (matched-control Theorem 3) — это превращает Theorem 3 из «фита по
   одной модели» в каузально проверенное (или честно descriptive) утверждение.
5. **Domer retention gates M1–M4** до сабмита границы; **никогда** не заявляй
   K_crit/mis-selection как решённое (R5/R6).
6. **Retraction ledger (R1–R8) — в §Limitations явно.** Это конкурентное
   преимущество, а не слабость.

*Конец документа.*
