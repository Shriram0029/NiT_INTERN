# Research Review, Improvement & Execution/Testing Plan
## Augmentation Selection for Banking Complaint Data using Genetic+Grey-Wolf Optimization and Low-Resource NLP

This document is the working checklist for turning `architecture_corrected.md` into a defensible research artifact (paper + reproducible codebase). It is organized as: **(1) Critical Review Checklist, (2) Improvement Plan, (3) Execution Roadmap, (4) Testing & Evaluation Protocol, (5) Risk Register.**

---

## 1. Critical Review Checklist

Use this before every draft/experiment cycle. Each item: ☐ open, and note the resolution once closed.

### 1.1 Framing & Terminology
- ☐ Confirm whether the project targets **cyber/fraud-specific** complaints or **general banking complaints**. The current CFPB sample is dominated by credit-report disputes, not cyber fraud. Decide and update title, abstract, and dataset description consistently.
- ☐ If keeping "cyber banking," define the filtering rule (which `issue`/`sub_issue` values count as cyber/fraud) and report the resulting subset size — a fraud-only CFPB subset may be too small for RoBERTa fine-tuning, which is itself a motivation for augmentation (make this argument explicit in the paper).
- ☐ Justify the "low-resource" claim precisely (§0 item 5 of architecture_corrected.md) — low-resource *per class*, not low-resource language.
- ☐ Rename "Fraud Graph," "Fraud Density," "Fraud-Aware Complexity Index" per the corrected architecture, or justify keeping them if scope is fraud-only.

### 1.2 Data Quality
- ☐ Quantify the fraction of empty `complaint_what_happened` rows in the actual working dataset (sample showed ~40%).
- ☐ Decide the admissibility filter threshold (min token count) and log how many complaints are excluded.
- ☐ Check `XXXX` redaction density per complaint — heavily redacted complaints may be near-uninformative even when non-empty; consider adding this as a CCI feature.
- ☐ Verify class balance of the final label taxonomy (product/issue-derived). Report imbalance ratio; this motivates augmentation directly.
- ☐ De-duplicate near-identical narratives (CFPB data has many template complaints).

### 1.3 Method Soundness
- ☐ Formalize each CCI dimension (§2 of architecture_corrected.md) with an equation, not a description.
- ☐ Specify GA hyperparameters: population size, selection method, crossover rate, mutation rate, elitism %, generations per chunk.
- ☐ Specify GWO hyperparameters: pack size, iterations, convergence parameter `a` schedule.
- ☐ Clarify optimization granularity (per-complaint vs per-chunk policy) — resolved in corrected architecture as per-complaint with warm-started shared population; confirm implementation matches.
- ☐ Specify the embedding model + numeric threshold for semantic similarity validation (Phase 7).
- ☐ Specify label-consistency check mechanism (classifier re-scoring vs rule-based entity/label match).
- ☐ Specify replay-buffer ratio and forgetting-detection protocol (fixed held-out set, evaluated every N chunks).
- ☐ Justify why GA *and* GWO are both needed (ablation must show GWO-only and GA-only underperform the hybrid — otherwise the hybrid is unmotivated complexity).

### 1.4 Novelty & Literature Grounding
- ☐ Cite prior work combining metaheuristics (GA, PSO, GWO, etc.) with NLP data augmentation policy search — position this work's specific contribution (selection-before-generation using a complaint-complexity feature vector) against it.
- ☐ Cite prior work on augmentation *selection* vs. *generation* (e.g., learned/RL-based augmentation policy search such as AutoAugment-style methods adapted to NLP) to show this is not the first selection-based approach, and clarify the delta.
- ☐ Cite CFPB-complaint-classification baselines already published, to contextualize expected F1/accuracy ranges.

### 1.5 Reproducibility
- ☐ Fixed random seeds, and multi-seed reporting (GA/GWO are stochastic — single-run numbers are not publishable without variance).
- ☐ Train/validation/test split protocol defined *before* any chunk streaming begins (held-out test set must never appear in Policy Memory or replay buffer).
- ☐ Environment/version pinning (transformers, sentence-transformers, backtranslation model versions).
- ☐ Policy Memory persisted and released (or its schema documented) for reproducibility.

### 1.6 Ethics / Compliance
- ☐ Confirm dataset license/terms for CFPB data (public domain, but confirm redistribution terms if releasing derived/augmented data).
- ☐ Confirm augmented synthetic complaints do not reconstruct sensitive info that CFPB redaction (`XXXX`) intended to hide (e.g., BERT/backtranslation regenerating plausible names/numbers in place of `XXXX` spans) — add an explicit check in Phase 7 that `XXXX` spans are preserved, not "filled in."

---

## 2. Improvement Plan (prioritized)

| Priority | Item | Rationale |
|---|---|---|
| P0 | Add Phase 0 admissibility filter | Prevents wasted optimization on empty narratives (~40% of raw data) |
| P0 | Formalize CCI feature equations | Currently unfalsifiable/unreproducible |
| P0 | Define baselines (No-Aug, Random, Fixed-BT, Fixed-BERT, Rule-based-selection) | Required to claim the optimizer adds value |
| P0 | Resolve fraud-framing vs. general-complaint framing | Cascades into title, related work, dataset description |
| P1 | Specify all GA/GWO hyperparameters and search budget per chunk | Needed for compute-cost claims and reproducibility |
| P1 | Add forgetting-detection protocol with fixed held-out set | Incremental learning claim currently unverified |
| P1 | Add ablation: GA-only vs GWO-only vs Hybrid vs No-optimization (rule-based CCI thresholds) | Establishes necessity of the hybrid metaheuristic |
| P2 | Add `XXXX`-preservation check to semantic validation | Data-ethics safeguard |
| P1 | Multi-seed significance testing with multiple-comparison correction + effect size (§4.4) | Raw p<0.05 across 5 comparisons against F inflates false-positive risk; effect size stops "significant but trivial" being reported as a win |
| P1 | Human evaluation of generated augmentations with inter-rater agreement (§4.4b) | Automated similarity/consistency checks don't establish that text reads as natural banking-complaint language |
| P2 | Cost/latency logging per chunk (wall-clock, GPU-hours) + cost-per-performance-point metric | Supports the "minimizes unnecessary augmentation" cost claim with real, comparable numbers instead of raw counts alone |
| P2 | Populate §4.8 results templates (Table 1, Table 2, significance report, forgetting curve, cost table, ethics log) as concrete deliverables | These were referenced by Stage 2/6/7 exit gates but had no defined schema — closing this gap makes the exit gates checkable |

---

## 3. Execution Roadmap (gated by deliverables, not calendar time)

**No stage in this roadmap carries a week/day estimate, and none should be added.** Each stage has only an **entry condition**, **exit gate**, and **deliverable**. A stage does not start until the previous stage's exit gate is met, and a failed exit gate sends work back a stage rather than sliding a deadline — progress is measured by evidence produced, not time elapsed. If a stage is blocked, the correct status update is "blocked on X exit criterion," never "behind schedule."

### Stage 0 — Scoping & Data Audit
- **Entry**: project kickoff.
- **Work**: resolve fraud-vs-general framing decision; run full data audit (size, empty-narrative %, class balance, redaction density, duplicate rate); freeze train/val/test split (test set locked away, never touched until final evaluation).
- **Exit gate**: framing decision documented + `data_audit_report.md` produced with all statistics above + splits frozen and hashed (so later stages can verify the test set was never altered).
- **Deliverable**: `data_audit_report.md`.

### Stage 1 — CCI Module
- **Entry**: Stage 0 exit gate met.
- **Work**: implement each CCI feature with unit tests against 5–10 hand-labeled example complaints.
- **Exit gate**: all CCI unit tests pass; feature distributions plotted and manually sanity-checked (no constant/degenerate features).
- **Deliverable**: `cci.py` + `test_cci.py` + feature-distribution plots.

### Stage 2 — Baselines
- **Entry**: Stage 1 exit gate met.
- **Work**: implement and train No-Augmentation, Random-Augmentation, Fixed-BT, Fixed-BERT, Hybrid-fixed, Rule-based-selection-on-CCI-thresholds.
- **Exit gate**: all baselines trained and evaluated on the locked test set, results logged in the Table 1 schema (§4.8) with mean±std and 95% CI over ≥5 seeds — a single-seed number does not satisfy this gate.
- **Deliverable**: baseline results table (Table 1, §4.8).

### Stage 3 — GA Module
- **Entry**: Stage 1 exit gate met (can run in parallel with Stage 2).
- **Work**: implement chromosome encoding, GA operators, elite preservation; unit-test on a synthetic toy fitness function before wiring to the real classification loop.
- **Exit gate**: GA converges on the toy function within a defined tolerance across ≥5 seeds.
- **Deliverable**: `ga.py` + convergence plots on toy function.

### Stage 4 — GWO Module
- **Entry**: Stage 3 exit gate met.
- **Work**: implement wolf-position update from GA elites; unit-test on the same toy fitness function used for GA.
- **Exit gate**: GA+GWO converges faster and/or to a better optimum than GA-only on the toy function.
- **Deliverable**: `gwo.py` + GA-only vs GA+GWO convergence comparison.

### Stage 5 — Full Pipeline Integration (Pilot)
- **Entry**: Stages 2 and 4 exit gates met.
- **Work**: wire Phases 0–9 end-to-end on a small pilot subset (e.g., 200 complaints); verify Policy Memory persistence and warm-start behavior across chunks.
- **Exit gate**: pilot run completes with no crashes, Policy Memory correctly warm-starts across ≥3 consecutive chunks, and per-chunk wall-clock is within the budget set in Stage 0.
- **Deliverable**: pilot run report with per-chunk timing.

### Stage 6 — Full-Scale Runs + Ablations
- **Entry**: Stage 5 exit gate met.
- **Work**: run full pipeline vs. all baselines (Stage 2) vs. ablations (GA-only, GWO-only, rule-based selection, no-optimization) — ≥5 seeds each, per §4.4.
- **Exit gate**: full ablation matrix (§4.3) populated per the Table 2 schema (§4.8) — mean±std, 95% CI, and corrected p-value + effect size for every F-vs-other comparison. A cell with a raw-only p-value (no correction, no effect size) does not satisfy this gate.
- **Deliverable**: results table (Table 2, §4.8) + significance report (§4.8 schema).

### Stage 7 — Forgetting & Robustness Checks
- **Entry**: Stage 6 exit gate met.
- **Work**: evaluate the incrementally-trained model on the fixed held-out set after every chunk; plot F1-over-time; stress-test on the empty/near-empty-narrative subset excluded in Stage 0.
- **Exit gate**: forgetting curve produced and interpreted (declining, flat, or improving) + zero crashes on the excluded/edge-case subset.
- **Deliverable**: forgetting-curve plot + robustness test log.

### Stage 8 — Write-up
- **Entry**: Stages 6 and 7 exit gates met.
- **Work**: draft paper sections mapped 1:1 to the corrected architecture phases; populate every claim in the introduction with a specific number from the audit/ablation/cost/ethics results.
- **Exit gate**: every quantitative claim in the draft has a traceable source table/figure in this document.
- **Deliverable**: paper draft + results appendix.

---

## 4. Testing & Evaluation Protocol

### 4.1 Unit-Level Tests (per module)
- **Phase 0 filter**: empty string → rejected; 1-token string → rejected (if below threshold); normal complaint → passes.
- **CCI**: known synthetic inputs → expected feature ranges (e.g., a complaint with 10 dollar amounts should score high Entity Density).
- **GA operators**: crossover/mutation preserve valid policy schema (no out-of-range budgets, no invalid strategy labels).
- **GWO update**: wolf positions remain within policy bounds after update (no divergence).
- **Semantic validation**: a generated sample with a swapped label → rejected; a sample with an altered `XXXX` span → rejected; a valid paraphrase → accepted.

### 4.2 Integration Tests
- End-to-end pipeline run on a fixed 20-complaint fixture returns deterministic phase outputs given a fixed seed.
- Policy Memory correctly retrieves nearest historical policy for a repeated/near-identical CCI vector.

### 4.3 Ablation Matrix (mandatory for the research claim)

| Configuration | GA | GWO | Selection | Purpose |
|---|---|---|---|---|
| A — No Augmentation | – | – | – | Lower bound |
| B — Fixed augmentation (BT only / BERT only / Hybrid) | – | – | – | Shows naive augmentation baseline |
| C — Rule-based selection on raw CCI thresholds | – | – | ✔ (rule-based) | Isolates value of *any* selection vs. none |
| D — GA-only policy search | ✔ | – | ✔ (learned) | Isolates GA contribution |
| E — GWO-only (random init, no GA elites) | – | ✔ | ✔ (learned) | Isolates GWO contribution |
| F — Full hybrid GA→GWO (proposed) | ✔ | ✔ | ✔ (learned) | Full system |

Report Accuracy, Macro-P/R/F1, augmentation-samples-generated (cost proxy), and wall-clock per chunk for all six rows.

### 4.4 Statistical Testing
- ≥5 random seeds per stochastic configuration (B–F); report the exact seed list for reproducibility, not just the count.
- Report mean ± std and a 95% bootstrap confidence interval (≥1,000 resamples) for every metric in the ablation matrix, not mean±std alone — std is easy to eyeball as "close enough" when the CI would show non-overlap or overlap clearly.
- Primary significance test: paired bootstrap or Wilcoxon signed-rank test between F (proposed) and each of the five other configurations (A–E) on the same test folds.
- **Multiple-comparison correction is mandatory**: F is compared against 5 other configurations, so raw p<0.05 per comparison inflates the family-wise error rate. Apply Holm-Bonferroni (or Benjamini-Hochberg if treating this as exploratory) across the 5 comparisons and report both raw and corrected p-values.
- **Report effect size alongside significance**, not instead of it: rank-biserial correlation or Cliff's delta for the Wilcoxon test. A statistically significant but trivial effect size (e.g., <0.1) should be stated as such in prose, not silently presented as "F wins."
- Flag any comparison where the *corrected* p-value is not significant, or where significance holds but effect size is negligible — do not claim superiority there. The paper's abstract/claims should only assert wins that survive both checks.
- Where class imbalance is material (per §1.2), also report per-class (not only macro) F1 for the minority class(es), since macro-F1 can mask a system that improves easy classes while leaving the minority class flat or worse.

### 4.4b Qualitative / Human Evaluation of Generated Augmentations
- Automated semantic-similarity and label-consistency checks (§4.1) do not by themselves establish that augmented text reads as natural, on-topic banking-complaint language.
- Draw a fixed random sample of generated augmentations (stratified across augmentation strategy and CCI complexity bucket; sample size large enough for a meaningful agreement statistic, e.g. ≥100 items per strategy) and have ≥2 independent raters score naturalness, semantic fidelity to the source complaint, and `XXXX`-span preservation on a short rubric.
- Report inter-rater agreement (Cohen's κ or Krippendorff's α) alongside the mean scores — a naturalness claim without an agreement number is not verifiable by a reviewer.
- This is a one-time calibration check on the accepted-sample pool, not a per-chunk gate; it validates that the automated Phase 7 filters are doing their job, not a substitute for them.

### 4.5 Forgetting / Continual-Learning Test
- Track Macro-F1 (and minority-class F1) on the **fixed held-out set** after every chunk across the full streaming run; a declining trend indicates catastrophic forgetting despite the replay buffer — report this curve regardless of outcome.
- Compute a single scalar "forgetting score" (e.g., max F1 achieved minus final F1) alongside the curve so the result is comparable across ablations, not just visually inspected.
- Repeat the forgetting run for at least 2 seeds — a single streaming run cannot distinguish a real forgetting trend from run-to-run noise.

### 4.6 Cost/Efficiency Test
- Log wall-clock and (if GPU-based) peak memory per phase per chunk; aggregate into total GPU-hours for the full run.
- Report total augmentation samples generated by the proposed system vs. the Fixed-Hybrid baseline, and express it as a ratio (e.g., "38% fewer samples generated for equivalent Macro-F1") — the core "minimizes unnecessary augmentation" claim needs this comparative number, not just the raw count.
- Report cost per unit of performance gain (e.g., GPU-hours per +1 Macro-F1 point over No-Augmentation) so the efficiency claim is defensible even if the hybrid is slower in absolute terms than a baseline.

### 4.7 Ethics/Privacy Test
- Automated check: for every generated sample, confirm all `XXXX` spans present in the source complaint are still present (unfilled) in the generated version. Log any violation as a validation failure, not just a semantic-similarity failure.
- Report the check as a pass rate over the full accepted-sample pool (target: 100%, per §6 Definition of Done) with the denominator stated explicitly (e.g., "10,482 / 10,482 accepted samples passed") — a percentage without the denominator is not auditable by a reviewer.

### 4.8 Results Reporting Templates

All tables below use the same column schema so Stage 2 and Stage 6 outputs are directly comparable and paper-ready without reformatting.

**Table 1 — Baseline Results (Stage 2 deliverable)**

| Configuration | Accuracy | Macro-P | Macro-R | Macro-F1 | Minority-class F1 | # Aug. samples | GPU-hours |
|---|---|---|---|---|---|---|---|
| A — No Augmentation | mean±std | mean±std | mean±std | mean±std [95% CI] | mean±std | 0 | — |
| B — Fixed-BT | … | … | … | … | … | … | … |
| B — Fixed-BERT | … | … | … | … | … | … | … |
| B — Fixed-Hybrid | … | … | … | … | … | … | … |
| C — Rule-based selection | … | … | … | … | … | … | … |

**Table 2 — Full Ablation Matrix (Stage 6 deliverable)**

| Configuration | Accuracy | Macro-P | Macro-R | Macro-F1 [95% CI] | Minority-class F1 | # Aug. samples | GPU-hours | vs. F: raw p | vs. F: corrected p | vs. F: effect size |
|---|---|---|---|---|---|---|---|---|---|---|
| A — No Augmentation | | | | | | | | | | |
| B — Fixed-Hybrid | | | | | | | | | | |
| C — Rule-based | | | | | | | | | | |
| D — GA-only | | | | | | | | | | |
| E — GWO-only | | | | | | | | | | |
| F — Full hybrid (proposed) | | | | | | — (reference row) | | — | — | — |

**Significance Report (accompanies Table 2)**
- One row per pairwise comparison (F vs. A, F vs. B, F vs. C, F vs. D, F vs. E): test used, raw p, Holm-Bonferroni-corrected p, effect size, and a plain-language verdict ("significant and non-trivial" / "significant but negligible effect" / "not significant").

**Forgetting Curve (Stage 7 deliverable)**
- Line plot: x-axis = chunk index, y-axis = Macro-F1 and minority-class F1 on the fixed held-out set, one line per seed (or mean±band across seeds); scalar forgetting score reported in the caption per §4.5.

**Cost Table (Stage 7/8 deliverable)**
- Per configuration: total GPU-hours, total augmentation samples generated, samples-generated ratio vs. Fixed-Hybrid, and GPU-hours per +1 Macro-F1 point over No-Augmentation.

**Ethics Check Log (Stage 8 deliverable)**
- Single summary line per run: `accepted_samples`, `xxxx_preserved`, `pass_rate`. Any run with pass_rate < 100% blocks submission per §6, not just a footnote.

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Fraud-only CFPB subset too small for meaningful fine-tuning | Medium | High | Decide framing early (Stage 0); if too small, broaden to general banking complaints and reframe paper accordingly |
| GA+GWO per-complaint search too slow for practical claims | Medium | Medium | Log wall-clock from Stage 5 pilot; cap search budget; report cost honestly even if it weakens the "efficient" claim |
| Catastrophic forgetting undermines incremental learning claim | Medium | High | Stage 7 test with fixed held-out set; have LoRA/frozen-backbone fallback ready as an ablation, not an afterthought |
| Reviewers challenge novelty vs. existing augmentation-policy-search literature | Medium | High | Complete §1.4 literature grounding before submission, not after first draft |
| Backtranslation/BERT-generated samples leak plausible values into `XXXX` redaction spans | Low-Medium | High (ethics) | §4.7 automated check; treat as a hard reject condition in Phase 7, not a soft penalty |

---

## 6. Definition of Done (for the research artifact)

- [ ] All items in §1 Critical Review Checklist closed or explicitly deferred with justification.
- [ ] Ablation matrix (Table 2, §4.8) fully populated with mean±std, 95% CI, corrected p-values, and effect sizes over ≥5 seeds — no cell left as a single-run number.
- [ ] Significance report (§4.8) complete for all 5 pairwise comparisons against configuration F, with a plain-language verdict for each.
- [ ] Human-evaluation naturalness/fidelity scores (§4.4b) reported with inter-rater agreement.
- [ ] Forgetting curve (§4.5) reported regardless of result, across ≥2 seeds, with a scalar forgetting score.
- [ ] Cost table (§4.6/§4.8) reported, including GPU-hours per +1 Macro-F1 point over No-Augmentation.
- [ ] Ethics check log (§4.7/§4.8) shows 100% `XXXX`-preservation pass rate with explicit numerator/denominator.
- [ ] Data audit numbers (Stage 0) match what's stated in the paper's dataset section.
- [ ] No remaining calendar-time language (weeks, deadlines, "by day N") anywhere in the roadmap or this checklist — every remaining item is gated by an exit criterion, not a date.
