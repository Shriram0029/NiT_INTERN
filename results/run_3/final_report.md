# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-19 13:55 UTC

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| Total Samples | 915 |
| Classes | Credit reporting or other personal consumer reports, Credit reporting, credit repair services, or other personal consumer reports, Debt collection, Credit card or prepaid card, Mortgage |
| Training Set | 732 |
| Validation Set | 80 |
| Test Set | 183 |

## 2. Label Distribution

| Class | Count |
|---|---|
| Credit reporting or other personal consumer reports | 292 |
| Credit reporting, credit repair services, or other personal consumer reports | 249 |
| Debt collection | 107 |
| Credit card or prepaid card | 47 |
| Mortgage | 37 |

## 3. FACI Statistics

| Statistic | Value |
|---|---|
| Average FACI Scalar | 0.2049 |
| Average Complexity | 0.5000 |
| Average Semantic Entropy | 0.8000 |

## 4. Policy Statistics

| Statistic | Value |
|---|---|
| Total Policies Generated | 25 |

### Strategy Breakdown

| Strategy | Count |
|---|---|
| Hybrid | 25 |

## 5. Optimizer Summary

| Metric | Value |
|---|---|
| Average Expected Utility | 0.8216 |
| Final Fitness Score | 0.8290 |
| Average Runtime (s/chunk) | 5.29 |
| Average Peak Memory (MB) | 0.9 |

## 6. Training Curves & Visualizations

See `visualizations/` for:
- `training_loss.png` — Cross-entropy loss per chunk
- `macro_f1.png` — Macro F1 trajectory per chunk
- `optimizer_convergence.png` — GA-GWO fitness convergence
- `faci_distribution.png` — FACI score histogram
- `policy_distribution.png` — Augmentation strategy distribution
- `replay_distribution.png` — Replay buffer class balance
- `confusion_matrix.png` — Final classification confusion matrix
- `baseline_comparison.png` — Macro F1 across all baselines
- `ablation_comparison.png` — Ablation study impact

## 7. Evaluation Metrics — Final Test Set (Hybrid GA+GWO)

| Metric | Value |
|---|---|
| Loss | 0.0000 |
| Accuracy | 0.6667 |
| Macro Precision | 0.6473 |
| Macro Recall | 0.7043 |
| **Macro F1** | **0.6648** |
| Weighted F1 | 0.6622 |
| ROC AUC | 0.8821 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting or other personal consumer reports | 0.7326 | 0.7778 | 0.7545 |
| Credit reporting, credit repair services, or other personal consumer reports | 0.5625 | 0.4909 | 0.5243 |
| Debt collection | 0.7083 | 0.5862 | 0.6415 |
| Credit card or prepaid card | 0.5333 | 0.8889 | 0.6667 |
| Mortgage | 0.7000 | 0.7778 | 0.7368 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting or other personal consumer reports | Credit reporting, credit repair services, or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting or other personal consumer reports | 63 | 14 | 2 | 2 | 0 |
| Credit reporting, credit repair services, or other personal consumer reports | 20 | 27 | 3 | 3 | 2 |
| Debt collection | 3 | 6 | 17 | 2 | 1 |
| Credit card or prepaid card | 0 | 0 | 1 | 8 | 0 |
| Mortgage | 0 | 1 | 1 | 0 | 7 |

## 10. Discussion

The Hybrid GA-GWO framework adaptively allocated augmentation budgets according to FACI-derived semantic complexity, improving Macro F1 over all static baselines.

## 11. Limitations

Dataset size (150 samples) limits statistical power. CPU-only inference restricts throughput.

## 12. Conclusion

Extension to larger corpora; integration of Butterfly Optimisation Algorithm (BOA); dynamic class-imbalance weighting.

---
*Report generated automatically by the pipeline. For statistical analysis across
multiple seeds, see `results/multi_seed/multi_seed_summary.csv`.*
