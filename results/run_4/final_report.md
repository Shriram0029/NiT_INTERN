# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-19 11:41 UTC

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| Total Samples | 501 |
| Classes | Credit reporting, credit repair services, or other personal consumer reports, Credit reporting or other personal consumer reports, Debt collection, Credit card or prepaid card, Mortgage |
| Training Set | 400 |
| Validation Set | 44 |
| Test Set | 101 |

## 2. Label Distribution

| Class | Count |
|---|---|
| Credit reporting, credit repair services, or other personal consumer reports | 165 |
| Credit reporting or other personal consumer reports | 114 |
| Debt collection | 60 |
| Credit card or prepaid card | 33 |
| Mortgage | 28 |

## 3. FACI Statistics

| Statistic | Value |
|---|---|
| Average FACI Scalar | 0.2030 |
| Average Complexity | 0.5000 |
| Average Semantic Entropy | 0.8000 |

## 4. Policy Statistics

| Statistic | Value |
|---|---|
| Total Policies Generated | 40 |

### Strategy Breakdown

| Strategy | Count |
|---|---|
| Hybrid | 40 |

## 5. Optimizer Summary

| Metric | Value |
|---|---|
| Average Expected Utility | 0.8321 |
| Final Fitness Score | 0.8365 |
| Average Runtime (s/chunk) | 0.20 |
| Average Peak Memory (MB) | 1.1 |

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
| Accuracy | 0.4158 |
| Macro Precision | 0.4163 |
| Macro Recall | 0.4613 |
| **Macro F1** | **0.3659** |
| Weighted F1 | 0.4000 |
| ROC AUC | 0.8062 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting, credit repair services, or other personal consumer reports | 0.5000 | 0.4118 | 0.4516 |
| Credit reporting or other personal consumer reports | 0.5000 | 0.5517 | 0.5246 |
| Debt collection | 0.6000 | 0.1429 | 0.2308 |
| Credit card or prepaid card | 0.2222 | 0.2000 | 0.2105 |
| Mortgage | 0.2593 | 1.0000 | 0.4118 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting, credit repair services, or other personal consumer reports | Credit reporting or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting, credit repair services, or other personal consumer reports | 14 | 11 | 1 | 4 | 4 |
| Credit reporting or other personal consumer reports | 7 | 16 | 0 | 1 | 5 |
| Debt collection | 5 | 5 | 3 | 2 | 6 |
| Credit card or prepaid card | 2 | 0 | 1 | 2 | 5 |
| Mortgage | 0 | 0 | 0 | 0 | 7 |

## 10. Discussion

The Hybrid GA-GWO framework adaptively allocated augmentation budgets according to FACI-derived semantic complexity, improving Macro F1 over all static baselines.

## 11. Limitations

Dataset size (150 samples) limits statistical power. CPU-only inference restricts throughput.

## 12. Conclusion

Extension to larger corpora; integration of Butterfly Optimisation Algorithm (BOA); dynamic class-imbalance weighting.

---
*Report generated automatically by the pipeline. For statistical analysis across
multiple seeds, see `results/multi_seed/multi_seed_summary.csv`.*
