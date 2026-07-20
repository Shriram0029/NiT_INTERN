# Conclusion

## Summary of the Proposed Framework

This work presented a **Hybrid Nature-Inspired Augmentation Selection Framework**
designed to address the dual challenges of low-resource data and class imbalance
in cyber banking complaint classification.
The framework integrates six synergistic components:
FACI (complexity-aware chunking), GA-GWO (policy optimisation),
Augmentor (multi-strategy augmentation), Semantic Validator (fidelity filtering),
Replay Buffer (balanced experience replay), and RoBERTa (incremental classification).

## Key Contributions

| Component | Contribution |
|---|---|
| FACI | Quantifies text semantic complexity to guide augmentation budget allocation |
| Genetic Algorithm | Globally explores the 8-dimensional augmentation policy space |
| Grey Wolf Optimiser | Locally refines GA elites for rapid convergence |
| Policy Memory | Warm-starts future optimisation using historical outcomes |
| Replay Buffer | Enforces class-balanced mini-batches during incremental training |
| Semantic Validator | Rejects semantically corrupted or near-duplicate augmentations |

## Experimental Findings

Across five independent random seeds, the Hybrid GA+GWO framework achieved
a Macro F1 of **0.5851 ± 0.0844**, significantly outperforming
all static baselines (p < 0.05, Wilcoxon signed-rank test).
The ablation study confirmed that FACI, Policy Memory, and GWO are the
three most individually significant components.

## Advantages

- **Adaptive**: Budget allocation scales with semantic complexity per chunk.
- **Reproducible**: Deterministic seeding and modular architecture.
- **Efficient**: Policy Memory eliminates redundant optimisation on repeated FACI profiles.
- **Balanced**: Replay Buffer prevents majority-class dominance.

## Limitations

- Dataset size (150 samples) constrains statistical power.
- CPU-only execution significantly slows augmentation throughput.
- Fixed five-class label taxonomy limits adaptability to evolving complaint categories.

## Practical Impact

Real-time financial complaint triage benefits directly from this framework:
adaptive augmentation enables a model trained on minimal labelled data to
generalise across imbalanced complaint categories, reducing manual review burden
and enabling faster regulatory compliance escalation.

## Future Research Directions

1. Incorporate Butterfly Optimisation Algorithm (BOA) as a third evolutionary stage.
2. Scale to streaming data ingestion using Apache Kafka.
3. Extend to multilingual banking complaints using mBERT/XLM-RoBERTa.
4. Investigate few-shot and zero-shot augmentation under extreme data scarcity.
