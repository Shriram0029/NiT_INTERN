# Nature-Inspired Augmentation Selection for Cyber Banking Data Awareness

This repository implements the Augmentation Selection Framework for Cyber Banking Data, powered by a Hybrid Genetic Algorithm and Grey Wolf Optimization (GA-GWO) pipeline.

## Complete Environment Setup

Ensure you have Python 3.9+ installed.

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

All system hyperparameters are stored in the `configs/` directory:
- `config.yaml`: Global runtime variables (chunk sizes, threshold flags, paths).
- `dataset.yaml`: Cyber-banking filter keywords, data column mappings, and preprocessing flags.
- `optimizer.yaml`: Boundary configurations for the Hybrid Optimizer (Budget, Mask Probability, Strategy Bounds).

## How to Run the Master Pipeline

To run the complete framework and simulate all baselines, simply execute the main script:
```bash
python main.py
```

### What happens when you run `main.py`?
1. **Dataset Pipeline:** It reads `data/complaints_150.json`, applies cyber-banking filters, builds knowledge graphs, and tracks class imbalances.
2. **Baselines Execution:** It iterates through all 8 baseline models defined in `configs/config.yaml` (`no_aug`, `random`, `fixed_bt`, `fixed_bert`, `rule_based`, `ga_only`, `gwo_only`, `hybrid`).
3. **Adaptive Augmentation:** For each chunk of data, it calculates multi-dimensional FACI scores and predicts the optimal budget, strategy, and expected utility.
4. **Validation:** It executes the augmentations and validates them through the `SemanticValidator`.
5. **Incremental Training:** Uses a Reservoir-sampled balanced `ReplayBuffer` to incrementally train a RoBERTa-based classification model.
6. **Analytics & Plotting:** Saves explanations to `results/<baseline>/explanations.csv`, validation rejects to `validation_report.csv`, and produces automated publication-quality plots in the `visualizations/` directory.
7. **Statistical Analysis:** Concludes by running rigorous statistical testing across the baselines.

## Output Navigation
After a run completes, you will find:
- **`data/filtered_dataset.json`**: The cleaned audit version of the raw dataset.
- **`results/<baseline>/`**: Stores the semantic validation reports, `policy_memory.csv`, and explainability logs.
- **`visualizations/<baseline>/`**: Houses `.png` radar charts, convergence plots, strategy distributions, confusion matrices, and metrics graphics.
- **`results/statistical_analysis.json`**: The final output containing 95% Confidence Intervals, Wilcoxon p-values, and Cohen's D effect sizes across experiments.
