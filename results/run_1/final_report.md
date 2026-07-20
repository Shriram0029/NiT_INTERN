# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-19 13:48 UTC

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
| Average FACI Scalar | 0.2042 |
| Average Complexity | 0.5000 |
| Average Semantic Entropy | 0.8000 |

## 4. Policy Statistics

| Statistic | Value |
|---|---|
| Total Policies Generated | 20 |

### Strategy Breakdown

| Strategy | Count |
|---|---|
| Hybrid | 20 |

## 5. Optimizer Summary

| Metric | Value |
|---|---|
| Average Expected Utility | 0.8323 |
| Final Fitness Score | 0.8323 |
| Average Runtime (s/chunk) | 7.56 |
| Average Peak Memory (MB) | 63.6 |

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
| Accuracy | 0.5847 |
| Macro Precision | 0.5429 |
| Macro Recall | 0.5930 |
| **Macro F1** | **0.5288** |
| Weighted F1 | 0.5630 |
| ROC AUC | 0.8495 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting or other personal consumer reports | 0.6633 | 0.8025 | 0.7263 |
| Credit reporting, credit repair services, or other personal consumer reports | 0.5366 | 0.4000 | 0.4583 |
| Debt collection | 0.5000 | 0.2069 | 0.2927 |
| Credit card or prepaid card | 0.3478 | 0.8889 | 0.5000 |
| Mortgage | 0.6667 | 0.6667 | 0.6667 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting or other personal consumer reports | Credit reporting, credit repair services, or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting or other personal consumer reports | 65 | 12 | 2 | 2 | 0 |
| Credit reporting, credit repair services, or other personal consumer reports | 25 | 22 | 2 | 4 | 2 |
| Debt collection | 8 | 7 | 6 | 7 | 1 |
| Credit card or prepaid card | 0 | 0 | 1 | 8 | 0 |
| Mortgage | 0 | 0 | 1 | 2 | 6 |

## 10. Discussion

The Hybrid GA-GWO framework adaptively allocated augmentation budgets according to FACI-derived semantic complexity, improving Macro F1 over all static baselines.

## 11. Limitations

Dataset size (150 samples) limits statistical power. CPU-only inference restricts throughput.

## 12. Conclusion

Extension to larger corpora; integration of Butterfly Optimisation Algorithm (BOA); dynamic class-imbalance weighting.

---
*Report generated automatically by the pipeline. For statistical analysis across
multiple seeds, see `results/multi_seed/multi_seed_summary.csv`.*
