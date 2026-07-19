# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-19 12:18 UTC

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
| Total Policies Generated | 20 |

### Strategy Breakdown

| Strategy | Count |
|---|---|
| Hybrid | 20 |

## 5. Optimizer Summary

| Metric | Value |
|---|---|
| Average Expected Utility | 0.8194 |
| Final Fitness Score | 0.8223 |
| Average Runtime (s/chunk) | 0.35 |
| Average Peak Memory (MB) | 1.5 |

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
| Accuracy | 0.4257 |
| Macro Precision | 0.5767 |
| Macro Recall | 0.5037 |
| **Macro F1** | **0.4282** |
| Weighted F1 | 0.4378 |
| ROC AUC | 0.8225 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting, credit repair services, or other personal consumer reports | 0.5556 | 0.2941 | 0.3846 |
| Credit reporting or other personal consumer reports | 0.5152 | 0.5862 | 0.5484 |
| Debt collection | 0.8333 | 0.2381 | 0.3704 |
| Credit card or prepaid card | 0.8000 | 0.4000 | 0.5333 |
| Mortgage | 0.1795 | 1.0000 | 0.3043 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting, credit repair services, or other personal consumer reports | Credit reporting or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting, credit repair services, or other personal consumer reports | 10 | 12 | 0 | 1 | 11 |
| Credit reporting or other personal consumer reports | 5 | 17 | 1 | 0 | 6 |
| Debt collection | 2 | 4 | 5 | 0 | 10 |
| Credit card or prepaid card | 1 | 0 | 0 | 4 | 5 |
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
