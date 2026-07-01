# Nature-Inspired Augmentation Selection for Cyber Banking Data Awareness using Genetic Optimization and Low-Resource NLP

This repository contains the codebase for a research-grade streaming NLP framework that processes Consumer Financial Protection Bureau (CFPB) cyber-banking complaints. It applies a **Hybrid Genetic Algorithm (GA) & Grey Wolf Optimization (GWO)** approach to dynamically discover the optimal text augmentation policies at runtime, minimizing semantic drift while maximizing classification utility.

## Core Architecture

The system operates as a chunk-wise streaming pipeline (e.g., processing 5-10 complaints at a time). It extracts named entities (e.g., `OTP`, `CARD`), constructs a **Fraud Graph**, and calculates a **Fraud-Aware Complexity Index (FACI)**. 

To determine the best augmentation policy, the system employs a **Hybrid Nature-Inspired Optimization Framework**:
1. **Genetic Algorithm (GA)**: Globally explores an 8-dimensional continuous policy space (BackTranslation ratio, BERT ratio, budget, mask probabilities, semantic thresholds, chunk priority, learning rate, and entity protection weight).
2. **Grey Wolf Optimization (GWO)**: Locally refines the elite policies discovered by the GA using Alpha, Beta, and Delta wolves.

Historical policies are saved to a **Policy Memory**, ensuring continuous adaptive learning. Augmented samples are stored in a **Replay Buffer** to train a RoBERTa-based classifier incrementally without catastrophic forgetting.

## File Structure

- **`main.py`**: The master orchestrator simulating a streaming environment and triggering the hybrid optimizer.
- **`configs/`**: Contains YAML files for the environment (`config.yaml`), optimizer constraints (`optimizer.yaml`), and data labels (`dataset.yaml`).
- **`src/`**: Modular implementations of all pipeline components (GA, GWO, FACI, Replay Buffer, Evaluator, Visualizer, etc.).
- **`outputs/` & `results/`**: Output directories tracking comprehensive metrics, CSV optimizer logs, and inference checkpoints.
- **`visualizations/`**: Auto-generated publication-ready plots for training loss, convergence tracking, confusion matrices, and FACI distributions.

## Environment & Requirements

Install dependencies from `requirement.txt`:
```bash
pip install -r requirement.txt
```

## How to Run

Execute the main simulation pipeline via:
```bash
python main.py
```

Check the `visualizations/` and `results/` directories after execution to see the output plots and tracking CSVs!
# NiT
