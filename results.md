# Augmentation Selection Results

This file tracks the latest test results across our different runtime configurations, in accordance with the Execution Plan.

## Current Evaluation Metrics

*Note: Models are evaluated on a synthetic 50-sample hold-out test set.*

| Configuration | Baseline Mode | Final Test Macro F1 | Unique Predicted Labels | Notes |
|---|---|---|---|---|
| **Proposed Hybrid (GA-GWO)** | `hybrid` | 0.1029 | {1, 3} | Highest F1 in initial testing |
| **Rule-Based Selection** | `rule_based` | 0.0774 | {2} | Degraded to majority-class/single-class prediction |

### Detailed Logs

#### 1. Proposed Hybrid (GA-GWO)

* **Mode**: `hybrid`
* **Test Set Class Distribution**: `{0: 9, 3: 10, 1: 10, 2: 8, 4: 13}`
* **Final Test Macro F1**: `0.1029`
* **Predictions**: `[1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]`

#### 2. Rule-Based Selection Baseline

* **Mode**: `rule_based`
* **Test Set Class Distribution**: `{0: 11, 3: 11, 4: 3, 1: 13, 2: 12}`
* **Final Test Macro F1**: `0.0774`
* **Predictions**: `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]`

*(Additional baselines like `no_aug`, `fixed_bt`, `fixed_bert`, and `random` will be populated here once their runs complete.)*

---

## Output Directories & Use Cases

The pipeline automatically generates several artifacts, statistics, and graphs to aid in tracking model performance and algorithm convergence.

### 1. `visualizations/`

Contains dynamically generated plots tracking both the machine learning models and the optimization algorithms.

* **`reports/`**: Contains final evaluation graphics, notably `confusion_matrix.png` which shows the true vs. predicted classes on the hold-out test set.
* **`faci/`**: Visualizes the Fraud-Aware Complexity Index for each chunk, including radar charts (multi-dimensional complexity) and index distributions over time.
* **`optimizer/`**: Tracks the GA and GWO algorithmic convergence curves (e.g., `convergence_chunk_X.png`).
* **`training/`**: Plots the loss history of the Incremental RoBERTa classifier over the streaming chunks.
* **`graph/`**: *(Reserved for rendering specific complaint Fraud Graphs for qualitative inspection).*

### 2. `results/`

Contains raw numerical outputs, history CSVs, and statistical tests.

* **`optimizer_history/history.csv`**: Extremely detailed per-chunk log containing the exact configuration chosen, the FACI scalar, and the optimizer's fitness evaluation. Useful for post-hoc analysis of *why* an augmentation budget was selected.
* **`explanations.csv`**: A plain-text explanation of the augmentation decision for each chunk (Explainability Engine outputs).
* **`policy_memory.csv`**: The database dump of the Policy Memory. It stores historical successful policies which the model retrieves using K-Nearest-Neighbors (KNN) to warm-start future chunks.
* **`confusion_matrices/`**, **`tables/`**, **`figures/`**, **`statistical_tests/`**: Directories reserved for the ablation studies required in Stage 6 (Ablation Matrix) and Stage 7 (Forgetting Tests) of the execution plan.
