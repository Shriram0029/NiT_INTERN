# Project Status & Future Implementation Roadmap

## 1. Current Status (v1: Hybrid GA + GWO)

The project currently implements a highly modular, streaming NLP framework designed for processing low-resource Consumer Financial Protection Bureau (CFPB) cyber-banking complaints. It leverages a **Hybrid Genetic Algorithm (GA) & Grey Wolf Optimization (GWO)** approach to dynamically discover optimal text augmentation policies.

### Completed Components & Features:
- **Chunk-wise Streaming Pipeline:** Successfully processes real CFPB complaints in robust, randomized batches from JSON data, mitigating initial dummy-data positional artifacts.
- **Data Analytics & Graphing:** Extracts entities (e.g., OTP, CARD), constructs Fraud Graphs, and computes the dynamic Fraud-Aware Complexity Index (FACI), proven to vary based on true input complexity.
- **Hybrid Optimization Framework:**
  - **GA:** Conducts global exploration across an 8-dimensional policy space (BackTranslation ratio, BERT ratio, budget, mask probabilities, etc.).
  - **GWO:** Refines elite GA policies using Alpha, Beta, and Delta wolves for local exploitation.
- **Robust Augmentation Handlers:** Integrates native HuggingFace `transformers` translation pipelines (MarianMT) to eliminate legacy dependency issues with `nlpaug` and `fairseq`.
- **Memory & Incremental Learning:** Utilizes a Policy Memory for historical adaptive learning and a Replay Buffer to train a RoBERTa-based classifier incrementally, mitigating catastrophic forgetting.
- **Visualizations & Logging:** Generates comprehensive execution artifacts, CSV logs, and publication-ready plots covering chunk-wise training loss, logit distributions, convergence, and metrics.
- **Version Control:** Repository is initialized, fully tracked, and synchronized with the remote GitHub origin.

---

## 2. Future Implementation (v2: Cost-Aware Adaptive Augmentation Selection with BOA)

The future roadmap transitions the core optimization strategy to the **Butterfly Optimization Algorithm (BOA)**, focusing heavily on predictive, cost-aware augmentation selection before generation.

### Planned Upgrades & Architecture Shifts:
- **Butterfly Optimization Algorithm (BOA):** Replacing GA + GWO with a BOA swarm model utilizing Stevens' Power Law for non-linear perceived fragrance intensity to better differentiate high vs. mediocre augmentation methods.
- **Meta-Learning Warm-up:** Pre-training the utility predictor on public NLP datasets (e.g., SST-2, TREC-6) as a shared prior to overcome the Epoch 1 cold-start problem.
- **Expanded Feature Set (Dual-Pathway Predictor):** Upgrading the sentence analyzer from hand-crafted features to an 11-feature architecture combining structural/semantic data with a parallel 128-dim BERT [CLS] learned representation pathway.
- **Cost-Aware Fitness (AUS):** Replacing the standard fitness function with the **Augmentation Utility Score (AUS)** ($\Delta F1 / \text{Cost}$) to natively penalize generation time, evaluation time, and semantic risk.
- **Epoch-Scheduled Exploration ($p$):** Moving from static exploration probabilities to a dynamic schedule (0.5 during warm-up to 0.95 at convergence) to prevent premature stagnation.
- **Semantic Validation Gate:** Pre-training checks for label consistency (NLI scoring), entity preservation for NER tasks, and cosine distance diversity filters to reject hallucinations before they hit the buffer.
- **Rigorous Baselines:** Implementing a contextual Thompson Sampling (multi-armed bandit) baseline to explicitly validate and justify the algorithmic complexity introduced by BOA.

### Next Steps for Development:
1. Initialize the offline Meta-Learning warm-up pipeline.
2. Develop and integrate the Dual-Pathway Utility Predictor.
3. Scaffold the BOA logic (Stevens' Power Law fragrance, scheduled $p$ switching).
4. Build the Semantic Validator module.
5. Setup ablation study runners and Thompson Sampling baseline comparison scripts.
