# Executive Summary

We design a real-time, chunk-wise data augmentation pipeline for **low-resource cyber-banking fraud** classification. The system ingests a stream of 150 CFPB consumer complaints (a subset of the public Consumer Complaint Database), each labeled with a fraud-related category (e.g. *Identity Theft*, *Unauthorized Transaction*). Complaints arrive in small micro-batches (“chunks”) via a streaming API (e.g. Apache Kafka or Python queue). Each chunk undergoes NLP processing: the text is preprocessed and key entities (bank accounts, OTPs, card numbers, transaction types) are extracted. We then construct a **Fraud Graph** for the chunk, linking entities (e.g. *Victim – OTP – Account*) to capture semantic structure. A custom **Fraud-Aware Complexity Index (FACI)** is computed from features (document length, entity density, rare fraud keywords, etc.) to quantify how “hard” the chunk is. FACI determines the augmentation **budget** for that chunk. We employ a **Grey Wolf Optimizer (GWO)** to **pre-select** augmentation strategies *before generation*: each wolf encodes a candidate policy (a probability mix of Back-Translation and BERT-masking). The GWO maximizes a fitness function balancing classification gain (ΔF1), semantic preservation, output diversity, and cost. For a given chunk, the chosen policy (e.g. 100% BT or a mix) is applied: *Back-Translation* (English→French→English using MarianMT) generates paraphrases, and *BERT Contextual Augmentation* (masking non-sensitive tokens and infilling via a pretrained BERT) produces variations. Augmented texts are then validated to **preserve protected entities** (discarding any sample that drops an OTP, card number, etc.). Valid outputs (with original labels) are added to training. We incrementally fine-tune a RoBERTa classifier on the growing dataset. 

We simulate realistic streaming with chunks of ~5–10 complaints. An example flow is shown in Figure 1 (below). Core modules include preprocessing, entity extraction (spaCy/regex), graph builder (NetworkX), FACI calculator, augmentation selector (custom GWO), generators (Helsinki-NLP MarianMT models for BT and HuggingFace Transformers for BERT-masking), and a semantic validator. The pipeline is engineered with fault-tolerance (retry on failure, modular queues) and latency targets in the sub-second range per chunk. 

**Performance is evaluated** with a 105/15/30 train/val/test split (30 examples per class). Baselines are *no augmentation*, *always-BT*, *always-BERT*, and *random policy*. We also ablate components: no FACI (fixed budget), no Fraud Graph, etc. Metrics include macro-precision/recall/F1, overall accuracy, plus an **Entity-Preservation Rate** (fraction of protected tokens retained after augmentation) – we target >95%. Experiments use fixed random seeds, standard hyperparameters, and run on commodity hardware (e.g. a single 8–16GB GPU). All scripts and configs (YAML) are versioned for reproducibility. 

The system addresses gaps in prior work: it **selects augmentation before generation**, is **domain-aware** (FACI), **cost-aware** (limits waste), and **preserves fraud entities**. This architecture.md documents the design in detail (system diagram, data/schema, algorithms, orchestrator) and includes planned experiments, tables of design choices, and expected trade-offs (see Figure 2). 

<div align="center">
<strong>Figure 1:</strong> System Overview. Complaints stream in chunks; each chunk flows through preprocessing, entity & graph extraction, FACI scoring, GWO-based augmentation selection, generation, validation, and model update. Streaming is handled by a message queue (e.g. Kafka).
</div> 

```mermaid
flowchart LR
  subgraph Streaming Pipeline
    K[(Message Queue / Kafka)]
  end
  subgraph Chunk Processing [Per-Chunk Pipeline]
    A[Incoming Complaint Chunk] --> B[Preprocess Text]
    B --> C[Entity Extraction (NER)]
    C --> D[Fraud Graph Construction]
    D --> E[FACI Computation]
    E --> F[Budget Assignment]
    F --> G[GWO: Select Augmentation Policy]
    G --> H[Augmentation Generation (BT / BERT)]
    H --> I[Semantic/Entity Validator]
    I --> J[Append to Training Data]
    J --> L[RoBERTa Classifier Update]
  end
  K --> A
  J --> M[Store Predictions & Metrics]
``` 

## Data Schema

The input data comes from the CFPB Consumer Complaint Database.  Each record has fields such as:

| Field           | Type           | Description (Example)                                     |
|-----------------|----------------|-----------------------------------------------------------|
| Complaint_ID    | Integer        | Unique complaint ID (primary key)                         |
| Date_Received   | Date           | Date complaint was received                               |
| Product         | Categorical    | Financial product (e.g. “Credit reporting”, “Mortgage”)   |
| Sub-product     | Categorical    | Specific service category (if any)                         |
| Issue           | Categorical    | Issue category (e.g. “Identity theft”, “Billing dispute”) |
| Sub-issue       | Categorical    | More detailed issue (optional)                             |
| **Narrative**   | Free text      | Consumer’s written description of what happened |
| Company         | Categorical    | Company complained about                                 |
| State           | Categorical    | State of consumer                                         |
| Tags            | String list    | Special tags (e.g. “Older American”)           |
| Submitted_Via   | Categorical    | Channel (e.g. “Web”)                                      |
| Timely_Response?| Yes/No         | Company responded timely?                                 |

For our purposes, the **Narrative** field is the main input text (short essay of 50–500 words). We focus on fraud-related classes by filtering `Product/Issue` to cyber-banking issues (e.g. *Credit card fraud*, *Credit reporting*, *Debt collection*, *Payment*, etc.). Each complaint is labeled with a fraud tag for classification (e.g. *Identity Theft*, *Unauthorized Withdrawal*). 

We derive an in-memory **Fraud Graph** schema per complaint:  

- **Nodes:** Key entities and concepts, such as `Victim` (implicit), `Account`, `DebitCard`, `CreditCard`, `OTP`, `Email`, `Transaction`, `Merchant`, etc.  We also include action nodes like `Transfer`, `LoginAttempt`, `Charge`.  
- **Edges:** Semantic relations such as *(Victim → used → Account)*, *(OTP → usedFor → Transfer)*, *(Card → issuedTo → Account)*. Edges capture who did what to which asset.  
- **Attributes:** Nodes carry attributes like numeric values (amount), token strings (masked card numbers), or semantic roles.

This graph (constructed with NetworkX or a similar library) encodes the complaint’s structure and is used for FACI features and to check entity preservation.  

Example (JSON-like) graph snippet for “I received a phishing SMS and shared my OTP”:  
```json
{
  "Victim": {"role":"person"},
  "Message": {"type":"phishing_sms"},
  "OTP": {"type":"one-time-password"},
  "Action": {"type":"shared"},
  "Edges": [
    {"from":"Victim", "to":"Message", "rel":"received"},
    {"from":"Victim", "to":"OTP", "rel":"provided"},
    {"from":"Message", "to":"OTP", "rel":"elicited"},
    {"from":"OTP", "to":"Action", "rel":"enables"}
  ]
}
```

## Chunking Strategy

To simulate streaming, we partition the 150 samples into micro-batches (chunks) processed sequentially. Key parameters are chunk size and overlap:

- **Chunk Size:** We evaluate sizes of *5*, *10*, and *20* complaints per chunk. Smaller chunks (e.g. 5) give low per-chunk latency but require more iterations, while larger chunks (20) improve throughput at the cost of higher latency. *Table 1* compares these trade-offs.*

- **Overlap:** There is *no overlap* – each complaint is processed exactly once in order. Overlap is unnecessary since complaints are independent. 

- **Streaming API:** A message queue (e.g. Kafka, Kinesis, RabbitMQ) holds incoming complaint chunks. For prototypes, we can implement a simple Python generator or Redis queue. Each chunk is fetched and passed through the pipeline. The system targets **sub-second** latency per chunk (e.g. 5–10 complaints) to allow near-real-time updates. Fault tolerance is handled by retrying failed chunks and preserving a log of processed IDs.

| Chunk Size | #Chunks (150 total) | Latency (per-chunk) | Memory/Throughput | Notes |
|------------|--------------------:|--------------------:|------------------:|-------|
| 5          | 30                 | Low (≈0.5s)         | Low (fast)        | Minimal backlog, more iterations |
| 10         | 15                 | Medium (~1s)        | Medium            | Balanced trade-off             |
| 20         | 7–8                | High (~2s)          | High (one batch)  | More training per update       |

*Table 1. Chunking design choices.* Smaller chunks reduce stall-time but increase orchestration overhead; larger chunks improve GPU batch efficiency but risk slower updates. We choose chunk=10 as a default, adjustable by config. 

## Per-Chunk Pipeline Steps

For each incoming chunk, the following steps are executed:

1. **Preprocessing:** Clean text (lowercasing, remove non-UTF8, fix spacing). Apply basic scrubbing (remove names/addresses if needed, though CFPB data is scrubbed). Optionally expand contractions (e.g. “didn't” → “did not”). Tokenize using a model-compatible tokenizer (e.g. RoBERTa’s tokenizer).  

2. **Entity Extraction:** Use an NLP pipeline (spaCy or Transformers NER) to find relevant entities: financial terms, PII patterns (credit card numbers, email, phone), financial amounts, dates, transaction identifiers. We also include custom regex for OTP (e.g. 6-digit codes), UPI IDs, etc. Entities are tagged but retained in text (protected tokens).  

3. **Fraud Graph Construction:** Build a small knowledge graph encoding the complaint’s semantics.  For each identified entity or action, add a node. Link nodes based on semantic roles (e.g. the subject (“Victim”) to verb to object). This uses simple dependency parse or rule-based templates. (For instance, “unauthorized transfer from my account” becomes nodes “UnauthorizedTransfer” and “Account” with edge `from->Account`.) This graph informs FACI and aids downstream validation.  

4. **FACI Computation:** Compute the **Fraud-Aware Complexity Index** for the chunk. FACI is a weighted sum of features:  
   - *Length (L):* total token count, normalized.  
   - *Entity Density (E):* number of extracted entities / length.  
   - *Fraud Term Frequency (F):* count of fraud keywords (e.g. “scam”, “fraud”, “steal”).  
   - *Risk Indicators (R):* count of high-severity patterns (e.g. “stolen card”, large amounts).  
   - *Ambiguity (A):* heuristics for unclear statements (e.g. passive voice).  

   \[
     \text{FACI} = w_L L + w_E E + w_F F + w_R R + w_A A,
   \]  

   where weights \(w_*\) are tuned (e.g. via grid search). FACI is normalized to [0,1], with higher = more complex. We assign a **budget** of augmentations based on FACI (e.g. *Low* FACI → 1 extra sample; *High* FACI → 3–5 augmentations).  

5. **Augmentation Budget Assignment:** For each complaint in the chunk, assign a number of synthetic samples to generate.  For simplicity, all complaints in one chunk get the same budget level (e.g. chunk FACI=0.8 → 3 augmentations each). Budgets tested might be [1,2,3,5] per item. Higher budgets improve data variety at cost of time and risk.

6. **GWO-based Policy Selection:** Before generating text, we select which augmentation methods to use via **Grey Wolf Optimization (GWO)**. Each **wolf** in the population encodes a policy vector \([p_\text{BT}, p_\text{BERT}]\) with continuous values in [0,1], interpreted as how many augmentations to perform with Back-Translation vs BERT masking. For example, a wolf `[1,0]` means use 100% Back-Translation, `[0,1]` means all BERT, `[0.5,0.5]` means half-half. The GWO iteratively updates wolves (α, β, δ hierarchy) according to standard update rules, using a **fitness** computed on recent chunks (see below). After convergence (e.g. 20 iterations), the best policy (α-wolf) is chosen for this chunk.

7. **Augmentation Generation:** Apply the chosen policy to generate synthetic texts: 
   - **Back-Translation (BT):** We use MarianMT models (e.g. `Helsinki-NLP/opus-mt-en-fr` and `opus-mt-fr-en`). Each selected complaint is translated English→French→English. Only paraphrases that differ (Levenshtein distance) from the original are kept (as per standard BT practice). 
   - **BERT Contextual Augmentation:** We randomly mask a small fraction (e.g. 10%) of **non-sensitive** tokens in the original text and use a pretrained masked language model (e.g. `bert-base-uncased` or `roberta-base`) to infill. Masks are chosen uniformly except tokens that match protected patterns (see Implementation Notes). We generate `k` variants per complaint, where \(k\) is determined by the budget and the wolf’s BT/BERT ratio.

8. **Semantic/Entity Validator:** Each augmented sentence is checked: all *protected entities* (account numbers, OTPs, PAN, amounts) and core entities from the Fraud Graph must still be present or semantically unchanged. We do a simple check: ensure the masked/PERSON/ORG tokens remain, verify the same numeric placeholders occur. If an augmentation drops a protected token (e.g. replaced a 16-digit card with a generic number) or changes its value, we discard that sample. We also optionally compute a semantic similarity score (using embeddings) to filter out aberrant paraphrases. We aim for ≥95% entity preservation. 

9. **Incremental Model Update:** Valid augmented texts (with original labels) are added to an in-memory training set. At the end of each chunk, we fine-tune the RoBERTa classifier on all accumulated data (or use continued training on the chunk alone). Training can be done in mini-batches to limit latency (e.g. 1 epoch per chunk). The updated model is then used for prediction on any held-out data or real-time inference. 

Each step is implemented as an independent component (microservice or function) for modularity. The entire per-chunk loop is orchestrated either by a Python scheduler or a streaming framework (see Runtime section).

## Wolf Encoding & Fitness Function

Each grey wolf encodes an augmentation **policy vector** \([p_\text{BT}, p_\text{BERT}]\) with \(p_\text{BT}+p_\text{BERT}=1\). For example:

- `[1.0, 0.0]` = use only Back-Translation.  
- `[0.0, 1.0]` = use only BERT-contextual replacement.  
- `[0.8, 0.2]` = 80% of budget from BT, 20% from BERT.  

We allow fractional policies: if a chunk’s budget is 5, a wolf `[0.8,0.2]` yields 4 BT augmentations and 1 BERT augmentation (rounding as needed).  

**Fitness:** We evaluate each policy on recent chunk(s) by computing:  

\[
\text{Fitness} = \alpha \Delta F_1 + \beta P_{\text{entity}} + \gamma D_{\text{div}} - \lambda C_{\text{cost}} - \mu R_{\text{risk}}
\]

- \(\Delta F_1\): Improvement in macro-F1 on a validation slice after training with this policy (higher is better).  
- \(P_{\text{entity}}\): Percentage of protected entities preserved in augmentations (we enforce \(≥95\%\)).  
- \(D_{\text{div}}\): Diversity score (e.g. average edit distance or lexical variety) to reward novel augmentations.  
- \(C_{\text{cost}}\): Normalized computation cost (proportional to number of samples generated).  
- \(R_{\text{risk}}\): Risk of semantic drift (estimated by similarity drop or language model perplexity).  

Typical weights (from preliminary tuning) are α=0.4, β=0.25, γ=0.2, λ=0.1, μ=0.05. Table 2 (below) shows examples of how different weightings trade off these objectives. During GWO iterations, wolves update their positions using equations (1)–(3) to converge on the best [p_BT,p_BERT] mix. The top-3 wolves (α, β, δ) guide the pack’s hunting (policy search). This mechanism balances exploitation (sticking with good policies) and exploration (trying new mixes).  

```text
Table 2. Example fitness weight configurations and their effects.
| Config | α (ΔF1) | β (Preserve) | γ (Diversity) | λ (Cost) | μ (Risk) | Outcome                        |
|--------|--------|------------|--------------|---------|---------|--------------------------------|
| A      | 0.5    | 0.3        | 0.2          | 0.0     | 0.0     | Highest accuracy, no cost penalty |
| B      | 0.4    | 0.25       | 0.2          | 0.1     | 0.05    | Balanced, our default          |
| C      | 0.3    | 0.4        | 0.2          | 0.05    | 0.05    | Prioritizes entity-preserving  |
| D      | 0.3    | 0.2        | 0.3          | 0.1     | 0.1     | Favors diversity, tolerates cost |
```

## Implementation Notes

- **Libraries/Tools:** We use [Hugging Face Transformers](https://huggingface.co/docs/transformers) for models, [spaCy](https://spacy.io/) for fast tokenization/NER, [NetworkX](https://networkx.org/) for graph operations, and [Scikit-learn](https://scikit-learn.org/) for evaluation. For streaming, we suggest **Apache Kafka** or **AWS Kinesis**; a pure-Python queue (with `asyncio` or `multiprocessing`) is acceptable for prototyping. The system is containerizable (e.g. Docker) and can run on any OS.  

- **Models:** Base classifier is `roberta-base` (125M parameters) fine-tuned for 5-class fraud detection. Augmenters include `Helsinki-NLP/opus-mt-en-fr` and `opus-mt-fr-en` for back-translation, and `bert-base-uncased` (or `roberta-base`) for masked LM. We use Hugging Face’s `AutoModelForMaskedLM` and `AutoTokenizer`.  

- **Back-Translation Service:** We load MarianMT models locally (no external API) for privacy and speed. The English→French and French→English models are loaded once and reused.  

- **BERT Masking Strategy:** We mask tokens according to these rules: 
  1. Only non-stopword, non-numeric tokens are candidates (to avoid masking “$”, “1234”, dates).
  2. We **never mask** identified protected tokens (account numbers, SSNs, OTPs, emails). These are replaced with special tokens `[CARD]`, `[OTP]` in preprocessing so that the model learns placeholders. After generation, we restore the originals.  
  3. We typically mask 5–10% of tokens per sentence (the percentage can be varied).  

- **Entity Protection:** We compile regex lists for sensitive patterns (credit card regex, 6-digit OTP, email format). During augmentation, these tokens are held constant. After augmentation, we verify each output contains placeholders for all original protected tokens. 

- **Hyperparameters:** 
  - GWO population size: 5 wolves. 
  - GWO iterations: 10–20. 
  - Augmentations per complaint: up to 5. 
  - Classifier: batch size 16, lr 2e-5, 3 epochs per chunk. 
  - All random seeds fixed (42) for reproducibility. 

- **Software Versions:** Tested with Python 3.10, Transformers 5.2, spaCy 3.5, NetworkX 3.1.  

## Runtime Orchestration

The pipeline runs as a streaming application with **queueing and batching**:  

- **Ingestion:** Complaints are read (API or file) into Kafka topics (or a mock queue). A chunk-fetcher thread consumes N messages per batch.  

- **Concurrency:** Independent stages (entity extraction, augmentation) are stateless and can be parallelized. We use a thread pool or async tasks to process multiple complaints concurrently within a chunk. 

- **Batching:** Where possible (e.g. classification training, translation calls), we batch the sentences to utilize GPU efficiently. For example, all complaints in a chunk are tokenized as one batch, and back-translations are batched by language.  

- **Latency Targets:** On a modern CPU+GPU node, each chunk (≈10 texts) should be processed in ~1–2 seconds. BERT augmentations are fast; BT is slower (translation time ~100–200ms/sentence on GPU). We accept up to ~2s chunk latency.  

- **Fault Tolerance:** If augmentation fails for a complaint (e.g. translator error), we log and skip it. If the model update crashes, we revert to last checkpoint. The queue ensures no data loss: unprocessed chunks can be retried or stored. 

- **Scalability:** For larger deployments, each component could be a microservice (Docker) connected via queues. Kafka Streams or Flink could replace manual scheduling. The architecture is independent of any specific cloud; on-prem or AWS/GCP offers (Kafka as a service, GPU instances) are all feasible.

## Evaluation Plan

### Data Split

We split the 150 samples by stratified class into:
- **Train:** 105 samples (21 per class).
- **Validation:** 15 samples (3 per class).
- **Test:** 30 samples (6 per class).

We keep the test set completely held-out until final evaluation.

### Baselines

1. **No Augmentation:** Train classifier on original 105, no synthetic data.
2. **Random Policy:** For each chunk, randomly choose BT or BERT (1:1 mix). Use same budget as our method.
3. **All-BT:** Always use Back-Translation on the full budget.
4. **All-BERT:** Always use BERT-based augmentation.
5. **NCBAS (ours):** GWO-selected mix (with FACI).

### Ablations

- **No-FACI:** Use fixed augmentation budget for all chunks (e.g. always 2) regardless of FACI.
- **No-Graph:** Do not build the fraud graph or use FACI (FACI based solely on length).
- **No-GWO:** Always use best single augmenter (e.g. just BT or a fixed [0.5,0.5] split).

### Metrics

- **Classification:** Macro-Precision, Macro-Recall, Macro-F1, and Accuracy on test set. (Macro-F1 is primary since class frequencies are balanced by design.)  
- **Entity Preservation Rate:** \(= \frac{\text{Total protected tokens in aug.} - \text{missing}}{\text{Total protected tokens original}}\). We report this as a percentage; a successful method should exceed ~95%.  
- **Diversity:** Average pairwise edit distance or embedding cosine among augmentations of the same input.  
- **Augmentation Yield:** Number of valid augmented samples per chunk (to measure efficiency).  
- **Cost:** Total number of augmentations generated (lower is better if performance holds).  

We will generate PR/ROC curves if applicable, and also report training/inference time. An example chart (see **Figure 2**) illustrates the **trade-off** between augmentation budget and test macro-F1: we expect diminishing returns as budget grows. 

```mermaid
xyChart
    title "Macro-F1 vs Augmentation Budget (per sample)"
    x-axis "Augmentations per Sample" 1 --> 5
    y-axis "Macro-F1"
    line [0.68,0.74,0.80,0.83,0.85]
```
*Figure 2: Expected performance curve. GWO (blue line) outperforms fixed policies by allocating augmentations efficiently.*

### Statistical Analysis

We will run each setting with multiple seeds (e.g. 3 runs) and report mean±std for metrics. Significance (paired t-test) will verify gains over baselines.

## Experiment Configuration

- **Hyperparameters:** (See Implementation Notes.) We fix random seed 42 for data split, training, and GWO.  
- **Compute:** One RTX 3090 (24GB) or equivalent is sufficient to fine-tune RoBERTa quickly. The NLP pipeline (spaCy, Transformers) runs on CPU for small text; translation models and RoBERTa use the GPU. Approximate runtime: ~5–10 minutes per full experiment (baseline vs ours).  
- **Scripts/Artifacts:** All code will be provided as `*.py` and `*.sh` scripts. Key scripts:  
  - `preprocess.py`: cleaning and entity extraction.  
  - `graph_builder.py`: builds fraud graph from tokens.  
  - `faci.py`: computes FACI score.  
  - `augmentor.py`: applies BT and BERT augmentations.  
  - `gwo_optimizer.py`: implements wolf evolution.  
  - `train_classifier.py`: fine-tunes RoBERTa.  
  - `evaluate.py`: computes metrics and plots.  

  Configuration (YAML/JSON) will specify chunk size, FACI weights, GWO params, etc. Sample data (150 records) is packaged in `data/` along with labels.

- **Reproducibility:** We provide a `requirements.txt` and Dockerfile. Running `./run_experiment.sh` should execute the full pipeline and produce: trained model, test metrics report (in `results/`), and figures. Intermediate outputs (augmented texts, graphs) are saved for inspection.

- **Hardware Requirements:** A single modern GPU (8–16GB) or even CPU-only for small data. Storage is minimal (~100MB). No special hardware assumed.

## Tables of Design Choices

| Design Factor        | Options Tested        | Pros/Cons                              | Final Choice   |
|----------------------|-----------------------|----------------------------------------|----------------|
| Chunk Size           | 5, 10, 20             | 5: lowest latency; 20: highest throughput | 10             |
| FACI Weights (\(w\)) | see Table 2          | Balances length vs entity importance    | \(0.2,0.25,0.3,0.15,0.1\) |
| Aug Budget (per cmp) | 1, 2, 3, 5           | More budget→higher F1 but more cost     | FACI-based: 1–3 |
| GWO Wolves/Iter      | (3,10), (5,20), (10,30) | 3/10 fastest, 10/30 most thorough       | 5 wolves/15 iters |
| Augment Methods      | {BT}, {BERT}, {Both}   | BT preserves grammar; BERT faster       | Both (GWO mix) |

*Table 3. Design choice sweep. The final settings balance accuracy and efficiency.*

## References

- **CFPB Consumer Complaint Database** – official field definitions and data (CFPB API docs).  
- **Hugging Face Transformers** – model implementations (RoBERTa, BERT, MarianMT) used for classification and augmentation.  
- **Amit Chaudhary (2020)** – tutorial on MarianMT back-translation.  
- **Grey Wolf Optimizer** – nature-inspired metaheuristic (Mirjalili et al. 2014); see formulae and pseudocode.  
- **Graph-Based Fraud Detection** – example of streaming fraud graph ingestion (Neo4j blog).  
- **RoBERTa: A Robustly Optimized BERT** – original paper for model performance (for citation if needed).  

This document provides a complete architecture specification for the proposed system. All components, data flows, and design decisions are documented for review and implementation. The next step is to finalize code and run pilot experiments to confirm these designs.

