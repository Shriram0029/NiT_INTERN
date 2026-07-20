# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-19 13:58 UTC

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
| Average Expected Utility | 0.8265 |
| Final Fitness Score | 0.8318 |
| Average Runtime (s/chunk) | 5.23 |
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
| Accuracy | 0.5792 |
| Macro Precision | 0.5895 |
| Macro Recall | 0.6689 |
| **Macro F1** | **0.6154** |
| Weighted F1 | 0.5796 |
| ROC AUC | 0.8665 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting or other personal consumer reports | 0.7083 | 0.6296 | 0.6667 |
| Credit reporting, credit repair services, or other personal consumer reports | 0.4167 | 0.4545 | 0.4348 |
| Debt collection | 0.5600 | 0.4828 | 0.5185 |
| Credit card or prepaid card | 0.5625 | 1.0000 | 0.7200 |
| Mortgage | 0.7000 | 0.7778 | 0.7368 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting or other personal consumer reports | Credit reporting, credit repair services, or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting or other personal consumer reports | 51 | 26 | 3 | 1 | 0 |
| Credit reporting, credit repair services, or other personal consumer reports | 18 | 25 | 8 | 2 | 2 |
| Debt collection | 2 | 9 | 14 | 3 | 1 |
| Credit card or prepaid card | 0 | 0 | 0 | 9 | 0 |
| Mortgage | 1 | 0 | 0 | 1 | 7 |

## 10. Discussion

The Hybrid GA-GWO framework adaptively allocated augmentation budgets according to FACI-derived semantic complexity, improving Macro F1 over all static baselines.

## 11. Limitations

Dataset size (150 samples) limits statistical power. CPU-only inference restricts throughput.

## 12. Conclusion

Extension to larger corpora; integration of Butterfly Optimisation Algorithm (BOA); dynamic class-imbalance weighting.

---
*Report generated automatically by the pipeline. For statistical analysis across
multiple seeds, see `results/multi_seed/multi_seed_summary.csv`.*
